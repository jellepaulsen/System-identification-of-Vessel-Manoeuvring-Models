"""Modell-agnostischer Simulator fuer Manoevriermodelle.

Der Simulator kennt kein konkretes Kraftmodell. Modelle werden per Name
registriert und muessen nur drei Dinge mitbringen (Duck-Typing):

    derivatives(t, y, inp) -> array(6)   # [dx0, dy0, dpsi, du, dv, dr]
    forces(t, y, inp)      -> tuple       # Kraefte fuer das Ausgabe-DataFrame
    force_columns          -> list[str]   # Spaltennamen dazu (F_X, F_Y, M_N immer dabei)

y ist der Zustand [x0, y0, psi, u, v, r], inp die aktuelle Input-Zeile
(N0, N1, delta_r0, delta_r1). Siehe SIMULATOR.md.
"""

import math
from dataclasses import dataclass
from time import time

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass
class ZigzagConfig:
    N: float          # propeller RPM
    delta_deg: float  # rudder angle magnitude [deg]
    psi_des: float    # heading threshold per side [deg]
    n_switches: int   # number of rudder reversals
    t0: float         # pre-maneuver acceleration time [s]
    t_aft: float      # coast-down duration after last switch [s]
    dt: float = 0.5   # output timestep [s]


class Simulator:
    def __init__(self, data: pd.DataFrame = None,
                 state_columns: list = ["x0", "y0", "psi", "u", "v", "r"],
                 input_columns: list = ["N0", "N1", "delta_r0", "delta_r1"],
                 dt_out: float = 0.01, method: str = "Radau"):
        self.data = data
        self.state_columns = state_columns
        self.input_columns = input_columns
        self.dt_out = dt_out
        self.method = method
        self.models = {}

    # ------------------------------------------------------------------
    # Modell-Verwaltung
    # ------------------------------------------------------------------
    def add_model(self, name: str, model):
        for attr in ("derivatives", "forces", "force_columns"):
            if not hasattr(model, attr):
                raise TypeError(f"Modell '{name}' hat kein '{attr}' (siehe SIMULATOR.md)")
        self.models[name] = model
        return model

    def __getitem__(self, name: str):
        return self._get_model(name)

    @property
    def model_names(self):
        return list(self.models)

    def _get_model(self, name: str):
        if name not in self.models:
            raise KeyError(f"Unbekanntes Modell '{name}', registriert: {self.model_names}")
        return self.models[name]

    # ------------------------------------------------------------------
    # Simulation mit Input-Zeitreihe (Zero-Order-Hold)
    # ------------------------------------------------------------------
    def simulate(self, name: str, x0_) -> pd.DataFrame:
        """Simulation mit den Inputs aus self.data (Index = Zeit [s], ZOH)."""
        model = self._get_model(name)
        if self.data is None:
            raise RuntimeError("Simulator.data ist nicht gesetzt")
        print(f"Initializing simulation with model '{name}'...")

        t_input = self.data.index.to_numpy(dtype=float)
        input_data = self.data[self.input_columns].to_numpy(dtype=float)

        def input_at(t):
            i = np.searchsorted(t_input, t, side="right") - 1
            return input_data[max(i, 0)]

        def rhs(t, y):
            return model.derivatives(t, y, input_at(t))

        t_span = [t_input.min(), t_input.max()]
        n_out = max(2, int(round((t_span[1] - t_span[0]) / self.dt_out)) + 1)
        t_eval = np.linspace(t_span[0], t_span[1], n_out)

        print("Starting integration...")
        start = time()
        sol = solve_ivp(rhs, t_span, x0_, t_eval=t_eval, method=self.method)
        print(f"Integration completed in {time() - start:.2f} s.")
        if not sol.success:
            print("Integration failed:", sol.message)

        input_idx = np.clip(np.searchsorted(t_input, t_eval, side="right") - 1, 0, None)
        df_input = pd.DataFrame(input_data[input_idx], columns=self.input_columns, index=t_eval)
        return self._assemble(model, sol.t, sol.y, df_input)

    # ------------------------------------------------------------------
    # IMO-Zigzag (synthetische, stueckweise konstante Inputs)
    # ------------------------------------------------------------------
    def simulate_zigzag(self, name: str, x0_, config: ZigzagConfig) -> pd.DataFrame:
        """IMO zigzag manoeuvre.

        Phases:
          1. ACCEL    - rudder=0, N=config.N for config.t0 seconds
          2. MANEUVER - alternating +-delta until heading threshold +-psi_des,
                        repeated config.n_switches times
          3. COAST    - rudder=0, N=0 for config.t_aft seconds
        """
        model = self._get_model(name)
        psi_des_rad = math.radians(config.psi_des)
        delta_rad = math.radians(config.delta_deg)

        def const_rhs(N, delta):
            inp = np.array([N, N, delta, delta], dtype=float)
            return lambda t, y: model.derivatives(t, y, inp)

        def _integrate(rhs, t_start, duration, y0, event=None):
            t_end = t_start + duration
            t_eval = np.arange(t_start, t_end + config.dt, config.dt)
            t_eval = np.clip(t_eval, t_start, t_end)
            evs = [event] if event is not None else []
            return solve_ivp(rhs, [t_start, t_end], y0,
                             t_eval=t_eval, events=evs, method=self.method)

        segments = []  # (sol, N_seg, delta_seg, phase_label)
        t_now = 0.0
        y_now = np.array(x0_, dtype=float)

        # --- Phase 1: ACCEL ---
        print(f"Zigzag [{name}]: acceleration phase (t0={config.t0}s, N={config.N} RPM)...")
        sol = _integrate(const_rhs(config.N, 0.0), t_now, config.t0, y_now)
        segments.append((sol, config.N, 0.0, "accel"))
        t_now, y_now = sol.t[-1], sol.y[:, -1]

        # --- Phase 2: MANEUVER ---
        print(f"Zigzag [{name}]: manoeuvre phase ({config.n_switches} switches, "
              f"delta={config.delta_deg} deg, psi_des={config.psi_des} deg)...")
        sign = 1
        for i in range(config.n_switches):
            rudder = sign * delta_rad
            psi_ref = float(y_now[2])

            def heading_event(t, y, s=sign, ref=psi_ref, des=psi_des_rad):
                return s * (y[2] - ref) - des
            heading_event.terminal = True
            heading_event.direction = 1  # fires only when value crosses 0 upward

            sol = _integrate(const_rhs(config.N, rudder), t_now, 600.0, y_now,
                             event=heading_event)
            segments.append((sol, config.N, rudder, "maneuver"))
            t_now, y_now = sol.t[-1], sol.y[:, -1]
            print(f"  switch {i+1}/{config.n_switches}: t={t_now:.1f}s  "
                  f"psi={math.degrees(float(y_now[2])):.1f} deg  "
                  f"next rudder={'port' if sign < 0 else 'stbd'}")
            sign *= -1

        # --- Phase 3: COAST ---
        print(f"Zigzag [{name}]: coast phase (t_aft={config.t_aft}s)...")
        sol = _integrate(const_rhs(0.0, 0.0), t_now, config.t_aft, y_now)
        segments.append((sol, 0.0, 0.0, "coast"))

        # --- Combine segments (drop duplicate boundary point between segments) ---
        parts_t, parts_y, parts_inp, parts_phase = [], [], [], []
        for k, (s, N_seg, delta_seg, phase) in enumerate(segments):
            sl = slice(1, None) if k > 0 else slice(None)
            parts_t.append(s.t[sl])
            parts_y.append(s.y[:, sl])
            n = s.t[sl].shape[0]
            parts_inp.append(np.tile([N_seg, N_seg, delta_seg, delta_seg], (n, 1)))
            parts_phase.extend([phase] * n)

        t_all = np.concatenate(parts_t)
        y_all = np.hstack(parts_y)
        df_input = pd.DataFrame(np.vstack(parts_inp), columns=self.input_columns, index=t_all)
        df_phase = pd.DataFrame({"phase": parts_phase}, index=t_all)

        df = self._assemble(model, t_all, y_all, df_input)
        print("Zigzag simulation completed.")
        return pd.concat([df, df_phase], axis=1)

    # ------------------------------------------------------------------
    # Ergebnis-DataFrame: states + inputs + Kraftspalten des Modells
    # ------------------------------------------------------------------
    def _assemble(self, model, t_all, y_all, df_input: pd.DataFrame) -> pd.DataFrame:
        df_sim = pd.DataFrame(y_all.T, columns=self.state_columns, index=t_all)
        inp_arr = df_input.to_numpy()
        forces_list = [model.forces(t_all[k], y_all[:, k], inp_arr[k])
                       for k in range(len(t_all))]
        df_forces = pd.DataFrame(forces_list, columns=model.force_columns, index=t_all)
        return pd.concat([df_sim, df_input, df_forces], axis=1)
