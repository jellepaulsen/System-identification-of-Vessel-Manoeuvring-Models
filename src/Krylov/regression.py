"""ForceRegression: hydrodynamische Kraftkoeffizienten aus Kalman-gefilterten
Trials schaetzen. Siehe REGRESSION.md fuer das Modell-Protokoll und den
vollen Datenfluss.

Ablauf pro Trial (ein EKF-Ergebnis aus kalman_filter.py + die zugehoerige
Krylov_forces-Instanz `kf`, die den Input und Pod-/Windkraefte liefert):
    1. Input (N0,N1,delta_r0,delta_r1) auf das EKF-Zeitraster interpolieren (ZOH)
    2. model.rigid_body_forces(y, acc) -> Kraft, die die gemessene Bewegung
       erfordert (reine Traegheitsseite, siehe abkowitz.py/krylov.py)
    3. kf.calc_pod_forces / kf.calc_windforces abziehen -> X_hydro, Y_hydro, N_hydro

Alle Trials werden danach aneinandergehaengt (Markov-Zustand: die Zeitordnung
zwischen Trials ist fuer die Regression irrelevant) und mit einem
statsmodels-OLS je Gleichung (X/Y/N) gegen ein Set von Termen regressiert.
Das Term-Format ist dasselbe wie in der Abkowitz-Koeffizienten-YAML
(Name -> [wert, [variablen]]), der Wert wird durch die Regression ersetzt.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

from Krylov.abkowitz import VARIABLES, q_scale


def variable_values(df: pd.DataFrame, sd) -> dict:
    """Wertet alle VARIABLES-Funktionen aus abkowitz.py vektorisiert auf einem
    Datensatz aus (Spalten u,v,r,N0,N1,delta_r0,delta_r1,U). Gleiche
    Definitionen wie im Simulator - eine neue Variable dort ergaenzen reicht.

    Zusaetzlich werden udot',vdot',rdot' (primed, aus den gemessenen
    Beschleunigungen) angeboten, damit added-mass-Terme wie Xudot/Yvdot/
    Nrdot (siehe abkowitz_coefficients.yml) mitregressiert werden koennen -
    das sind hydrodynamische Kraefte (Reaktion des Wassers auf die
    Beschleunigung), keine Schiffs-Massenterme, und gehoeren daher zu
    X_hydro/Y_hydro/N_hydro, nicht zu rigid_body_forces.
    """
    y = [None, None, None, df["u"].to_numpy(), df["v"].to_numpy(), df["r"].to_numpy()]
    inp = [df[c].to_numpy() for c in ("N0", "N1", "delta_r0", "delta_r1")]
    U = df["U"].to_numpy()
    values = {name: f(y, inp, U, sd) for name, f in VARIABLES.items()}
    values["udot"] = df["udot"].to_numpy() * sd.L / U**2
    values["vdot"] = df["vdot"].to_numpy() * sd.L / U**2
    values["rdot"] = df["rdot"].to_numpy() * sd.L**2 / U**2
    return values


def term_column(values: dict, mult: list) -> np.ndarray:
    """Spalte fuer einen Term wert*var1'*var2'*... ; leere Liste = Bias (Einsen)."""
    col = np.ones_like(next(iter(values.values())))
    for m in mult:
        col = col * values[m]
    return col


class ForceRegression:
    """Regressiert hydrodynamische X'/Y'/N'-Koeffizienten im Prime-System aus
    einem oder mehreren Kalman-gefilterten Trials (siehe REGRESSION.md).
    """

    def __init__(self, model, sd, t_settle: float = 30.0, eps: float = 0.5):
        """
        model:    Objekt mit rigid_body_forces(y, acc) -> (X, Y, N) dimensional
                  (AbkowitzModel, KrylovModel, oder ein eigenes Modell)
        sd:       ShipConfig fuer die Prime-Normierung (q_scale)
        t_settle: erste t_settle Sekunden je Trial verwerfen (EKF-Einschwingphase)
        eps:      minimale Geschwindigkeit U fuer die Prime-Normierung
        """
        self.model = model
        self.sd = sd
        self.t_settle = t_settle
        self.eps = eps
        self.runs = []
        self.dataset = None
        self.term_spec = None
        self.results = {}

    # ------------------------------------------------------------------
    # Trial hinzufuegen
    # ------------------------------------------------------------------
    def add_run(self, kalman_result: pd.DataFrame, kf, name: str = None) -> pd.DataFrame:
        """Bereitet ein Trial vor und haengt es an self.runs an.

        kalman_result: Rueckgabe von ExtendedKalmanFilter.filter() (mit
                       udot,vdot,rdot-Spalten)
        kf:            zugehoerige Krylov_forces-Instanz (Input, Wind, Pod)
        """
        df = kalman_result.loc[kalman_result.index[0] + self.t_settle:].copy()
        df["U"] = np.sqrt(df["u"] ** 2 + df["v"] ** 2)

        t = df.index.to_numpy(dtype=float)
        t_inp = kf.data.index.to_numpy(dtype=float)
        inp_arr = kf.data[kf.input_columns].to_numpy(dtype=float)
        idx = np.clip(np.searchsorted(t_inp, t, side="right") - 1, 0, None)
        inp_at_t = inp_arr[idx]
        for col, vals in zip(kf.input_columns, inp_at_t.T):
            df[col] = vals

        n = len(df)
        X_tot, Y_tot, N_tot = np.empty(n), np.empty(n), np.empty(n)
        pod_X, pod_Y, pod_N = np.empty(n), np.empty(n), np.empty(n)
        wind_X, wind_Y, wind_N = np.empty(n), np.empty(n), np.empty(n)

        for i, (row, inp) in enumerate(zip(df.itertuples(), inp_at_t)):
            y = [row.x0, row.y0, row.psi, row.u, row.v, row.r]
            acc = [row.udot, row.vdot, row.rdot]
            X_tot[i], Y_tot[i], N_tot[i] = self.model.rigid_body_forces(y, acc)
            pod_X[i], pod_Y[i], pod_N[i] = kf.calc_pod_forces(
                input=inp, input_columns=kf.input_columns, sd=kf.sd, urx=row.u)
            wind_X[i], wind_Y[i], wind_N[i] = kf.calc_windforces(
                uwind=kf.uwind, wind_dir=kf.wind_dir, rho_air=kf.rho_air,
                Ax=kf.sd.A_x, Ay=kf.sd.A_y, L=kf.sd.L, x0_=y,
                cxw=kf.sd.C_x_w, cyw=kf.sd.C_y_w, cnw=kf.sd.C_n_w)

        df["X_total"], df["Y_total"], df["N_total"] = X_tot, Y_tot, N_tot
        df["pod_X"], df["pod_Y"], df["pod_N"] = pod_X, pod_Y, pod_N
        df["wind_X"], df["wind_Y"], df["wind_N"] = wind_X, wind_Y, wind_N
        # bereinigte hydrodynamische Kraft (Rumpf) - das ist es, was regressiert wird
        df["X_hydro"] = X_tot - pod_X - wind_X
        df["Y_hydro"] = Y_tot - pod_Y - wind_Y
        df["N_hydro"] = N_tot - pod_N - wind_N

        df["run"] = name if name is not None else f"run{len(self.runs)}"
        self.runs.append(df)
        return df

    # ------------------------------------------------------------------
    # Trials zu einem Datensatz zusammenfassen (kein Zeitsortieren: Markov-Zustand)
    # ------------------------------------------------------------------
    def build_dataset(self) -> pd.DataFrame:
        if not self.runs:
            raise RuntimeError("Keine Trials hinzugefuegt (add_run)")
        self.dataset = pd.concat(self.runs, ignore_index=True)
        return self.dataset

    # ------------------------------------------------------------------
    # Fit je Gleichung (X, Y, N) im Prime-System
    # ------------------------------------------------------------------
    def fit(self, term_spec: dict) -> dict:
        """Regressiert X'_hydro, Y'_hydro, N'_hydro auf die Terme aus term_spec.

        term_spec: gleiches Format wie die Abkowitz-Koeffizienten-YAML,
                   z.B. {"Xu": [None, ["u"]], "Xvv": [None, ["v", "v"]], ...}.
                   Der erste Buchstabe entscheidet die Gleichung, der Wert
                   wird ignoriert (die Regression bestimmt ihn).

        Returns: {"X": RegressionResultsWrapper, "Y": ..., "N": ...}
        """
        if self.dataset is None:
            self.build_dataset()
        df = self.dataset
        self.term_spec = term_spec

        values = variable_values(df, self.sd)
        U = df["U"].to_numpy()
        q = q_scale(self.sd, U)
        mask = U > self.eps

        self.results = {}
        for eq in ("X", "Y", "N"):
            terms = {name: spec for name, spec in term_spec.items() if name[0].upper() == eq}
            if not terms:
                continue
            design = pd.DataFrame(
                {name: term_column(values, spec[1]) for name, spec in terms.items()})
            norm = q * self.sd.L if eq == "N" else q
            target = df[f"{eq}_hydro"].to_numpy() / norm
            self.results[eq] = sm.OLS(target[mask], design[mask]).fit()
        return self.results

    def summaries(self) -> dict:
        """Diagnostik je Gleichung (R^2, p-Werte, Konfidenzintervalle)."""
        return {eq: res.summary() for eq, res in self.results.items()}

    # ------------------------------------------------------------------
    # Export im abkowitz.py-YAML-Format
    # ------------------------------------------------------------------
    def to_yaml(self, path: str = None) -> str:
        if not self.results:
            raise RuntimeError("Noch nichts gefittet (fit)")
        coeffs = {}
        for eq, res in self.results.items():
            for name, value in res.params.items():
                _, mult = self.term_spec[name]
                coeffs[name] = [float(value), mult]

        if path is None:
            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            path = f"abkowitz_coefficients_regressed_{ts}.yml"

        with open(path, "w") as f:
            f.write("# auto-generated by ForceRegression.to_yaml - "
                    "values are regressed, not measured\n")
            yaml.safe_dump(coeffs, f, sort_keys=False)
        return path


if __name__ == "__main__":
    # Beispiel: ForceRegression auf einem echten Drehkreis-Trial, Kraftmodell
    # fuer die Starrkoerperseite (rigid_body_forces) = AbkowitzModel.
    # Vom Repo-Root starten: PYTHONPATH=src python src/Krylov/regression.py
    from Krylov.krylov import Krylov_forces, KrylovModel
    from Krylov.abkowitz import AbkowitzModel
    from Krylov.kalman_filter import ExtendedKalmanFilter

    # --- Parameter und Trial-Daten (wie in kalman_filter.py) ---
    with open("conf/base/parameters/krylov.yml") as f:
        kry_p = yaml.safe_load(f)
    with open("data/01_raw/wlfa/ship_data.yml") as f:
        ship_parameters = yaml.safe_load(f)
    ship_resistance = pd.read_csv("data/01_raw/wlfa/resistance_curve_HM.csv")
    prop_openwater = pd.read_csv("data/01_raw/wlfa/freif.inp", sep=r"\s+",
                                 header=None, names=["J", "KT", "KQ"])
    data = pd.read_parquet(
        "data/03_primary/wlfa/TC_20.0_nan_b_10.0_35.0_rpm_stb_turn.parquet")

    kf = Krylov_forces(ship_parameters=ship_parameters, krylov_parameters=kry_p,
                       ship_resistance=ship_resistance, prop_openwater=prop_openwater,
                       data=data)
    kf.data = kf.map_column_names_for_simulation(
        input_columns_data=["rpm_response_ps", "rpm_response_stb",
                            "azimuth_response_ps", "azimuth_response_stb"],
        geopos=["latitude", "longitude"])
    kf.data["delta_r1"] = kf.data["delta_r0"]
    kf.data.index = kf.data.index / 1e9

    # --- EKF: Zustandsschaetzung aus GPS/Kompass (Transitionsmodell = Krylov) ---
    state_columns = ["x0", "y0", "psi", "u", "v", "r"]
    meas_columns = ["X", "Y", "heading"]
    input_columns = ["N0", "N1", "delta_r0", "delta_r1"]
    P_0 = np.diag([1.0, 1.0, np.radians(2.0)**2, 0.1**2, 0.1**2, np.radians(0.5)**2])
    Q = np.diag([0.1**2, 0.1**2, np.radians(0.1)**2, 0.05**2, 0.05**2, np.radians(0.1)**2])
    R = np.diag([1.0**2, 1.0**2, np.radians(1.0)**2])
    H = np.eye(3, 6)

    ekf = ExtendedKalmanFilter(data=kf.data, model=KrylovModel(kf),
                               state_columns=state_columns, meas_coulumns=meas_columns,
                               input_columns=input_columns, P_0=P_0, Q=Q, R=R, H=H)
    result = ekf.filter(dt=1.0)
    accel_cols = [c for c in result.columns if c.endswith("dot")]
    print(f"EKF: {len(result)} Zeitschritte, Beschleunigungen dabei: {accel_cols}")

    # --- Regression: Starrkoerperseite kommt aus dem AbkowitzModel ---
    abk_model = AbkowitzModel.from_yaml(
        "data/03_primary/wlfa/abkowitz_coefficients.yml", kf.sd, eps=0.5)

    reg = ForceRegression(model=abk_model, sd=kf.sd, t_settle=30.0)
    reg.add_run(result, kf, name="TC_20.0_stb_turn")
    reg.build_dataset()
    print(f"Datensatz: {len(reg.dataset)} Zeilen nach Ausschluss der Einschwingphase")

    # Term-Set: Abkowitz-Kernterme + added mass (udot/vdot sind primed
    # Beschleunigungen, siehe REGRESSION.md - "Added mass")
    term_spec = {
        "Xu":    [None, ["u"]],
        "Xvv":   [None, ["v", "v"]],
        "Xrr":   [None, ["r", "r"]],
        "Xudot": [None, ["udot"]],
        "Yv":    [None, ["v"]],
        "Yr":    [None, ["r"]],
        "Yvdot": [None, ["vdot"]],
        "Nv":    [None, ["v"]],
        "Nr":    [None, ["r"]],
        "Nvdot": [None, ["vdot"]],
    }
    results = reg.fit(term_spec)
    for eq, res in results.items():
        print(f"\n--- Gleichung {eq}: R^2={res.rsquared:.3f} ---")
        print(res.params.round(5))

    print("\nDiagnose (Gleichung X):")
    print(reg.summaries()["X"])

    path = reg.to_yaml()
    print(f"\nRegressierte Koeffizienten gespeichert: {path}")
