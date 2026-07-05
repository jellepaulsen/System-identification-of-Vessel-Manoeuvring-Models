"""Vergleich von Krylov-Simulationen mit Messdaten und Parameter-Tuning.

Typischer Ablauf (Notebook):

    from Krylov.comparison_tuning import KrylovComparison, ParameterTuner

    comp = KrylovComparison(kf, data)      # kf: Krylov_forces, data: Trial-Parquet (Index in ns)
    comp.prepare_inputs()                  # rpm/azimuth -> kf.data (N0, N1, delta_r0, delta_r1)
    comp.run()                             # Simulation mit x0 aus der Messung
    print(comp.error_factors())            # Fehler als Faktor sim/real pro Kanal
    comp.plot_timeseries(); comp.plot_track()

    tuner = ParameterTuner(comp)
    print(tuner.tune())                    # iteriert k_x/k_n (und k_y aus der Querbilanz)
"""

import math

import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy.optimize import brentq


class KrylovComparison:
    """Vergleicht eine Krylov_forces-Simulation mit den Messdaten eines Trials.

    Args:
        kf: fertig konfigurierte Krylov_forces-Instanz
        data_measured: Roh-Segment (Parquet), Index Zeitstempel in ns
        fs: Zielrate fuer das gemeinsame Vergleichsraster [Hz]
        antenna_offset: (x_a, y_a) GPS-Antenne relativ zum Schwerpunkt [m];
            korrigiert die aus SOG/COG abgeleiteten u/v um den Hebelarm
        use_rudder_force: calc_pod_rud_force statt calc_pod_forces verwenden
        subtract_azimuth_offset: azimuth_offset_ps/stb von den Azimut-Kanaelen abziehen
    """

    INPUT_CHANNELS = {
        "N0": "rpm_response_ps",
        "N1": "rpm_response_stb",
        "delta_r0": "azimuth_response_ps",
        "delta_r1": "azimuth_response_stb",
    }

    def __init__(self, kf, data_measured: pd.DataFrame, fs: float = 10.0,
                 antenna_offset=(0.0, 0.0), use_rudder_force: bool = True,
                 subtract_azimuth_offset: bool = True):
        self.kf = kf
        self.raw = data_measured
        self.fs = fs
        self.antenna_offset = antenna_offset
        self.subtract_azimuth_offset = subtract_azimuth_offset
        if use_rudder_force:
            kf.calc_pod_forces = kf.calc_pod_rud_force
        self.sim = None
        self._prepare_measurement()

    # ------------------------------------------------------------------
    # Messdaten-Aufbereitung
    # ------------------------------------------------------------------
    def _prepare_measurement(self):
        raw = self.raw
        t = raw.index.to_numpy(dtype=float) / 1e9
        t = t - t[0]
        self.t = np.arange(0.0, t[-1], 1.0 / self.fs)

        def rs(col):
            return np.interp(self.t, t, raw[col].to_numpy(dtype=float))

        m = pd.DataFrame(index=self.t)
        m["psi"] = np.unwrap(rs("heading"))
        m["sog"] = rs("SOG")
        m["cog"] = np.unwrap(rs("COG"))

        # Drehrate aus geglaettetem Heading (robuster als rate_of_turn-Kanal)
        k = max(3, int(2.0 * self.fs) // 2 * 2 + 1)  # ~2 s Fenster, ungerade
        m["r"] = np.gradient(medfilt(m["psi"].to_numpy(), k), self.t)

        # u, v aus SOG/COG/heading (erdfest -> bodyfest), GPS-Hebelarm korrigiert
        u_ant = m["sog"] * np.cos(m["cog"] - m["psi"])
        v_ant = m["sog"] * np.sin(m["cog"] - m["psi"])
        x_a, y_a = self.antenna_offset
        m["u"] = u_ant + m["r"] * y_a
        m["v"] = v_ant - m["r"] * x_a
        m["U"] = np.hypot(m["u"], m["v"])

        for col in ("X", "Y"):
            if col in raw.columns:
                m[col.lower()] = rs(col) - raw[col].iloc[0]

        azi = 0.5 * (rs("azimuth_response_ps") + rs("azimuth_response_stb"))
        if self.subtract_azimuth_offset and "azimuth_offset_ps" in raw.columns:
            off = 0.5 * (rs("azimuth_offset_ps") + rs("azimuth_offset_stb"))
            azi = azi - np.where(np.isfinite(off), off, 0.0)  # Offset-Kanal kann NaN sein
        m["delta"] = azi

        self.meas = m
        self.legs = self._detect_legs()
        self.trial_type = "ZZ" if len(self.legs) >= 3 else "TC"

    def _detect_legs(self, thresh_deg: float = 8.0, min_dur: float = 6.0):
        """Phasen voller Ruderauslage, getrennt nach Vorzeichen (fuer Zigzag)."""
        adeg = np.degrees(self.meas["delta"].to_numpy())
        state = np.where(adeg > thresh_deg, 1, np.where(adeg < -thresh_deg, -1, 0))
        legs, start = [], 0
        for i in range(1, len(state)):
            if state[i] != state[i - 1]:
                if state[i - 1] != 0 and self.t[i - 1] - self.t[start] > min_dur:
                    legs.append((start, i - 1, state[i - 1]))
                start = i
        return legs

    def steady_window(self, frac: float = 0.4):
        """Index-Slice des stationaeren Teils (letzter Anteil des Segments)."""
        return slice(int(frac * len(self.t)), len(self.t))

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def prepare_inputs(self, medfilt_kernel: int = 15):
        """Baut kf.data (N0, N1, delta_r0, delta_r1; Index in s) aus den Messkanaelen."""
        raw = self.raw
        t = raw.index.to_numpy(dtype=float) / 1e9
        t = t - t[0]
        inp = {}
        for sim_col, meas_col in self.INPUT_CHANNELS.items():
            x = raw[meas_col].to_numpy(dtype=float)
            if sim_col.startswith("delta") and self.subtract_azimuth_offset:
                off_col = "azimuth_offset_ps" if sim_col.endswith("0") else "azimuth_offset_stb"
                if off_col in raw.columns:
                    off = raw[off_col].to_numpy(dtype=float)
                    x = x - np.where(np.isfinite(off), off, 0.0)  # Offset-Kanal kann NaN sein
            inp[sim_col] = medfilt(x, medfilt_kernel)
        self.kf.data = pd.DataFrame(inp, index=t)
        return self.kf.data

    def x0_from_measurement(self):
        m = self.meas
        return [0.0, 0.0, float(m["psi"].iloc[0]),
                float(m["u"].iloc[0]), float(m["v"].iloc[0]), float(m["r"].iloc[0])]

    def run(self, x0=None):
        if x0 is None:
            x0 = self.x0_from_measurement()
        sim = self.kf.simulate(x0)
        # auf das Vergleichsraster interpolieren
        ts = sim.index.to_numpy()
        s = pd.DataFrame(index=self.t)
        for col in ("psi", "u", "v", "r", "x0", "y0"):
            s[col] = np.interp(self.t, ts, sim[col].to_numpy(dtype=float))
        s["U"] = np.hypot(s["u"], s["v"])
        self.sim_raw = sim
        self.sim = s
        self._check_validity()
        return sim

    def _check_validity(self):
        """Warnt, wenn die Simulation den Gueltigkeitsbereich des Modells verlaesst."""
        u, v = self.sim["u"].to_numpy(), self.sim["v"].to_numpy()
        beta = np.degrees(np.arctan2(v, np.where(np.abs(u) < 1e-9, 1e-9, u)))
        psix = float(self.kf.kp.get("psix", 90))
        if (u < 0).any():
            print("WARNUNG: u < 0 in der Simulation - Modell ausserhalb des Gueltigkeitsbereichs!")
        if (np.abs(beta) > psix).any():
            print(f"WARNUNG: |beta| > {psix:.0f} deg (max {np.abs(beta).max():.0f} deg) - "
                  "Faktoren/Metriken sind in diesem Bereich Artefakte!")

    # ------------------------------------------------------------------
    # Metriken
    # ------------------------------------------------------------------
    def error_factors(self) -> pd.DataFrame:
        """Fehler pro Kanal: Faktor sim/real (Amplituden bzw. stationaere Mittel),
        dazu RMSE und Bias auf dem gemeinsamen Zeitraster."""
        if self.sim is None:
            raise RuntimeError("Erst run() aufrufen.")
        rows = {}
        for col in ("psi", "r", "u", "v", "U"):
            sim_y = self.sim[col].to_numpy()
            real_y = self.meas[col].to_numpy()
            rows[col] = {
                "factor": self._amplitude_factor(col, sim_y, real_y),
                "rmse": float(np.sqrt(np.mean((sim_y - real_y) ** 2))),
                "bias": float(np.mean(sim_y - real_y)),
            }
        return pd.DataFrame(rows).T

    def _amplitude_factor(self, col, sim_y, real_y):
        if self.trial_type == "ZZ" and self.legs:
            # Verhaeltnis der mittleren Auslenkung pro Leg (psi: Kursaenderung)
            if col == "psi":
                sw_s = [sim_y[b] - sim_y[a] for a, b, _ in self.legs]
                sw_r = [real_y[b] - real_y[a] for a, b, _ in self.legs]
            else:
                sw_s = [np.mean(sim_y[a:b]) for a, b, _ in self.legs]
                sw_r = [np.mean(real_y[a:b]) for a, b, _ in self.legs]
            num = np.mean(np.abs(sw_s))
            den = np.mean(np.abs(sw_r))
        else:
            s = self.steady_window()
            if col == "psi":  # Gesamtdrehung im Fenster
                num = abs(sim_y[s][-1] - sim_y[s][0])
                den = abs(real_y[s][-1] - real_y[s][0])
            else:
                num = abs(np.mean(sim_y[s]))
                den = abs(np.mean(real_y[s]))
        return float(num / den) if den > 1e-9 else float("nan")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_timeseries(self):
        import matplotlib.pyplot as plt
        rows = [("psi [deg]", "psi", np.degrees), ("r [deg/s]", "r", np.degrees),
                ("u [m/s]", "u", None), ("v [m/s]", "v", None), ("U [m/s]", "U", None)]
        fig, axes = plt.subplots(len(rows) + 1, 1, figsize=(12, 14), sharex=True)
        axes[0].plot(self.t, np.degrees(self.meas["delta"]), color="C2")
        axes[0].set_ylabel("delta [deg]")
        for ax, (label, col, conv) in zip(axes[1:], rows):
            sim_y, real_y = self.sim[col], self.meas[col]
            if conv is not None:
                sim_y, real_y = conv(sim_y), conv(real_y)
            ax.plot(self.t, sim_y, label="sim", color="C0")
            ax.plot(self.t, real_y, label="real", color="C1", alpha=0.7)
            ax.set_ylabel(label)
            ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("t [s]")
        fig.tight_layout()
        return fig

    def plot_track(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.sim["y0"], self.sim["x0"], label="sim", color="C0")
        if "x" in self.meas.columns:
            ax.plot(self.meas["y"], self.meas["x"], label="real", color="C1", alpha=0.7)
        ax.plot(0, 0, "ko", ms=6, label="Start")
        ax.set_xlabel("y / Ost [m]")
        ax.set_ylabel("x / Nord [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"Trajektorie ({self.trial_type})")
        ax.legend()
        fig.tight_layout()
        return fig


class ParameterTuner:
    """Fixpunkt-Iteration der Kalibrierungsfaktoren k_x, k_y, k_n gegen einen Trial.

    Update-Regeln pro Iteration (Faktor = sim/real aus KrylovComparison):
        k_n *= factor(r)     mehr Drehrate -> mehr Daempfung noetig
        k_x *= factor(U)**2  zu schnell -> mehr Widerstand (Widerstand ~ U^2)
        k_y   aus der stationaeren Querbilanz (nur Drehkreis, einmalig vorab)
    """

    def __init__(self, comp: KrylovComparison):
        self.comp = comp
        self.kf = comp.kf
        self.history = []

    # -- k_y analytisch aus der stationaeren Querbilanz (Drehkreis) ----------
    def solve_k_y_from_balance(self):
        """Loest die Laengsbilanz nach der (nicht messbaren) Drift v auf und
        bestimmt daraus k_y ueber die Querbilanz. Nur fuer quasi-stationaere
        Drehkreise sinnvoll."""
        comp, kf = self.comp, self.kf
        s = comp.steady_window()
        m = comp.meas
        r = float(np.polyfit(comp.t[s], m["psi"].to_numpy()[s], 1)[0])
        U = float(np.mean(m["sog"].to_numpy()[s]))
        d = float(np.mean(m["delta"].to_numpy()[s]))
        t_inp = kf.data.index.to_numpy(dtype=float)
        n = float(np.mean(0.5 * (np.interp(comp.t, t_inp, kf.data["N0"])
                                 + np.interp(comp.t, t_inp, kf.data["N1"]))[s]))
        m_x = kf.hydro_mass_dict["m11"] + kf.sd.m
        m_y = kf.hydro_mass_dict["m22"] + kf.sd.m
        k_x_saved, k_y_saved, k_n_saved = kf.k_x, kf.k_y, kf.k_n

        def surge_residual(v):
            u = math.sqrt(max(U ** 2 - v ** 2, 1e-6))
            x0 = [0, 0, 0, u, v, r]
            be, bs = kf.eff_drift_angle(x0, kf.eps)
            kf.k_x, kf.k_y, kf.k_n = 1.0, 1.0, 1.0  # rohe Rumpfkraefte
            try:
                kX, kY, kN, _ = kf.krylov_force(x0, kf.sd, be, bs, kf.eps)
                pX, pY, pN = kf.calc_pod_rud_force(
                    input=np.array([n, n, d, d]), input_columns=kf.input_columns,
                    sd=kf.sd, urx=u)
            finally:
                kf.k_x, kf.k_y, kf.k_n = k_x_saved, k_y_saved, k_n_saved
            return k_x_saved * kX + pX + m_y * r * v, kY, pY

        vs = np.linspace(-0.9, 0.9, 181)
        res = np.array([surge_residual(v)[0] for v in vs])
        i0 = np.where(np.diff(np.sign(res)))[0]
        if not len(i0):
            print("solve_k_y_from_balance: keine Loesung der Laengsbilanz gefunden, k_y unveraendert.")
            return kf.k_y
        v_star = brentq(lambda v: surge_residual(v)[0], vs[i0[0]], vs[i0[0] + 1])
        _, kY, pY = surge_residual(v_star)
        u_star = math.sqrt(max(U ** 2 - v_star ** 2, 1e-6))
        k_y = (m_x * r * u_star - pY) / kY
        print(f"solve_k_y_from_balance: v*={v_star:+.3f} m/s -> k_y={k_y:.2f}")
        return float(k_y)

    # -- Fixpunkt-Iteration ---------------------------------------------------
    def tune(self, max_iter: int = 6, tol: float = 0.05, solve_k_y: bool = None):
        comp, kf = self.comp, self.kf
        if solve_k_y is None:
            solve_k_y = comp.trial_type == "TC"
        if solve_k_y:
            kf.k_y = self.solve_k_y_from_balance()

        for it in range(max_iter):
            comp.run()
            ef = comp.error_factors()
            f_r, f_U = ef.loc["r", "factor"], ef.loc["U", "factor"]
            self.history.append({"iter": it, "k_x": kf.k_x, "k_y": kf.k_y,
                                 "k_n": kf.k_n, "factor_r": f_r, "factor_U": f_U})
            print(f"it{it}: k_x={kf.k_x:.3f} k_y={kf.k_y:.2f} k_n={kf.k_n:.3f} | "
                  f"factor r={f_r:.2f}, U={f_U:.2f}")
            if abs(f_r - 1) < tol and abs(f_U - 1) < tol:
                print("konvergiert.")
                break
            kf.k_n *= f_r
            kf.k_x *= f_U ** 2
        return self.result()

    def result(self):
        return {"k_x": round(self.kf.k_x, 3), "k_y": round(self.kf.k_y, 3),
                "k_n": round(self.kf.k_n, 3),
                "history": pd.DataFrame(self.history)}

    def to_yaml_block(self, trial_name: str = "") -> str:
        """Gibt den fertigen Block fuer die krylov.yml aus."""
        ef = self.comp.error_factors()
        return (
            f"# Kalibrierungsfaktoren aus {trial_name or 'Trial'} "
            f"({self.comp.trial_type}), factor r={ef.loc['r', 'factor']:.2f}, "
            f"U={ef.loc['U', 'factor']:.2f}\n"
            f"# Nur gueltig zusammen mit calc_pod_rud_force!\n"
            f"k_x: {self.kf.k_x:.3f}\n"
            f"k_y: {self.kf.k_y:.3f}\n"
            f"k_n: {self.kf.k_n:.3f}\n"
        )
