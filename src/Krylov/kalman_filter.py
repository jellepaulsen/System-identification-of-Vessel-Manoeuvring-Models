import numpy as np
import pandas as pd
from sympy import symbols, Matrix, sin, cos, simplify
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter






class ExtendedKalmanFilter:
    """
    Kalman-Filter für die Zustandsabschätzung eines Schiffes.

    Attributes
    ----------
    model : object
        Das Schiffmodell, das die Systemdynamik beschreibt.
    state_estimate : np.ndarray
        Der aktuelle Schätzwert des Zustandsvektors.
    covariance_estimate : np.ndarray
        Die aktuelle Schätzung der Kovarianzmatrix.
    process_noise_cov : np.ndarray
        Die Kovarianzmatrix des Prozessrauschens.
    measurement_noise_cov : np.ndarray
        Die Kovarianzmatrix des Messrauschens.
    """

    def __init__(self, data: pd.DataFrame, model: object, state_columns: list, meas_coulumns: list, input_columns: list, P_0: np.ndarray, Q: np.ndarray, R: np.ndarray, H: np.ndarray):
        self.data = data
        self.model = model
        self.state_columns = state_columns
        self.meas_coulumns = meas_coulumns
        self.input_columns = input_columns
        self.P_0 = P_0                      # initial covariance matrix
        self.Q = Q                          # process noise covariance matrix
        self.R = R                          # measurement noise covariance matrix
        self.H = H                          # measurement matrix

        self.meas = data[meas_coulumns]
        self.u_inp = data[input_columns]


    def transition(self, x, t0, t1, u_inp):
        """Praediktion: Zustand von t0 nach t1 integrieren.
        x:     np.ndarray mit den Zustandsvariablen (x0, y0, psi, u, v, r)
        u_inp: DataFrame mit den Input-Spalten (N0, N1, delta_r0, delta_r1),
        Index = Zeit [s]. Der Input wird fuer das Intervall konstant gehalten
        (Zero-Order-Hold auf den Wert bei t0).
        """
        x = np.asarray(x, dtype=float)

        i = u_inp.index.searchsorted(t0, side="right") - 1
        inp = u_inp.iloc[max(i, 0)].to_numpy(dtype=float)

        def rhs(t, y):
            return self.model.derivatives(t, y, inp)

        sol = solve_ivp(rhs, (t0, t1), x, method='Radau', t_eval=[t1])

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        return sol.y[:, -1]  # Rückgabe des Zustands am Ende des Integrationsintervalls


    
    def numerical_jacobian(self, x, f_x, t0, t1, u_inp, steps=None):
        """Numerische Berechnung der Jacobi-Matrix der Transition an der Stelle x.

        x: Punkt, an dem die Jacobi-Matrix berechnet werden soll.
        f_x: Wert von transition(x, t0, t1, u_inp).
        steps: Schrittweite je Zustandskomponente für die numerische Ableitung
               (Skalar oder Array der Länge len(x); Default 1e-4 für alle).
        """
        n = len(x)
        m = len(f_x)
        if steps is None:
            steps = np.full(n, 1e-4)
        else:
            steps = np.broadcast_to(np.asarray(steps, dtype=float), (n,))
        J = np.zeros((m, n))

        for i in range(n):
            x_step = np.array(x, dtype=float)
            x_step[i] += steps[i]
            f_step = self.transition(x_step, t0, t1, u_inp)
            J[:, i] = (f_step - f_x) / steps[i]

        return J

    def predict(self, x, P, t0, t1, u_inp):

        x_pred = self.transition(x, t0, t1, u_inp)
        f_x = x_pred
        F = self.numerical_jacobian(x, f_x, t0, t1, u_inp)
        P_pred = F @ P @ F.T + self.Q
        
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z):
        '''
        Update-Schritt des Kalman-Filters.
        x_pred: Vorhersage des Zustandsvektors.
        P_pred: Vorhersage der Kovarianzmatrix.
        z: Messung.
        '''

        y = z -self.H @ x_pred  # Innovationsvektor 
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi  # Winkelnormalisierung für psi
        S = self.H @ P_pred @ self.H.T + self.R  # Innovationskovarianz 
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman-Gain 
        x = x_pred + K @ y  # Aktualisierter Zustandsvektor
        # simple update of the covariance matrix
        # P = (np.eye(len(x)) - K @ self.H) @ P_pred  # Aktualisierte Kovarianzmatrix 1-K faktor wie sicher die messung war @H nur anwenden auf gemessene states @pred Faktor anwenden auf den alten P_pred
        
        # Joseph-Formel für die Kovarianzmatrix (numerisch stabiler)
        P = (np.eye(len(x)) - K @ self.H) @ P_pred @ (np.eye(len(x)) - K @ self.H).T + K @ self.R @ K.T
        return x, P, y
    
    def filter(self, x0: np.ndarray = None, dt: float = 0.1,
              accel_columns: tuple = ("u", "v", "r"),
              savgol_window: int = 11, savgol_polyorder: int = 3):
        """Filtert die Messreihe und haengt zusaetzlich udot/vdot/rdot an.

        Die Beschleunigungen werden fuer ForceRegression (regression.py,
        siehe REGRESSION.md) gebraucht: u,v,r vor dem Differenzieren mit
        Savitzky-Golay glaetten (sonst verstaerkt np.gradient das Rauschen
        im EKF-Output), dann zentrale Differenzen mit np.gradient(., dt).
        """
        inp = self.u_inp
        meas = self.meas
        start_time = self.data.index[0]
        end_time = self.data.index[-1]
        t_grid = np.arange(start_time, end_time, dt)

        P = self.P_0.copy()

        if x0 is None:
            # Anfangszustand aus der ersten Messzeile (SOG/COG/heading/ROT -> Body-Frame)
            x0 = self.model.kf.get_x0(sog_col="SOG", cog_col="COG",
                                      heading_col="heading", rot_col="rate_of_turn")
        x0 = np.asarray(x0, dtype=float).copy()

        records = []  # Überwachung: Zeit, Zustand, Unsicherheit und Innovation je Zeitschritt

        for t0, t1 in zip(t_grid[:-1], t_grid[1:]):
            x0, P = self.predict(x0, P, t0, t1, inp)
            i = meas.index.searchsorted(t1, side="right") - 1 # index für den measured state t1
            z = meas.iloc[max(i, 0)].to_numpy(dtype=float)
            x0, P, y = self.update(x0, P, z)

            sigma = np.sqrt(np.diag(P))
            record = {"t": t1}
            record.update({name: val for name, val in zip(self.state_columns, x0)})
            record.update({f"sigma_{name}": val for name, val in zip(self.state_columns, sigma)})
            record.update({f"y_{name}": val for name, val in zip(self.meas_coulumns, y)})
            records.append(record)

        df = pd.DataFrame(records).set_index("t")

        # ungerades Fenster, nicht groesser als die Datenreihe
        window = min(savgol_window, len(df) if len(df) % 2 else len(df) - 1)
        smooth = window >= 3 and window > savgol_polyorder
        for name in accel_columns:
            series = df[name].to_numpy()
            if smooth:
                series = savgol_filter(series, window, savgol_polyorder)
            df[f"{name}dot"] = np.gradient(series, dt)

        return df


if __name__ == "__main__":
    # Beispiel: vollstaendiges Krylov-Modell mit dem TC-Trial aus dem
    # Simulate-Notebook (J_10.01_simulate.ipynb), vom Repo-Root starten:
    #   PYTHONPATH=src python src/Krylov/kalman_filter.py
    import yaml
    from Krylov.krylov import Krylov_forces, KrylovModel

    # --- Parameter und Daten (wie im Notebook ueber den Kedro-Katalog) ---
    with open("conf/base/parameters/krylov.yml") as f:
        kry_p = yaml.safe_load(f)
    with open("data/01_raw/wlfa/ship_data.yml") as f:
        ship_parameters = yaml.safe_load(f)
    ship_resistance = pd.read_csv("data/01_raw/wlfa/resistance_curve_HM.csv")
    prop_openwater = pd.read_csv("data/01_raw/wlfa/freif.inp", sep=r"\s+",
                                 header=None, names=["J", "KT", "KQ"])
    data = pd.read_parquet(
        "data/03_primary/wlfa/TC_20.0_nan_b_10.0_35.0_rpm_stb_turn.parquet")

    # --- Krylov_forces aufsetzen (wie Notebook-Zelle: mappen, delta_r1, Index in s) ---
    kf_forces = Krylov_forces(
        ship_parameters=ship_parameters,
        krylov_parameters=kry_p,
        ship_resistance=ship_resistance,
        prop_openwater=prop_openwater,
        data=data,
    )
    kf_forces.data = kf_forces.map_column_names_for_simulation(
        input_columns_data=["rpm_response_ps", "rpm_response_stb",
                            "azimuth_response_ps", "azimuth_response_stb"],
        geopos=["latitude", "longitude"],
    )
    kf_forces.data["delta_r1"] = kf_forces.data["delta_r0"]
    kf_forces.data.index = kf_forces.data.index / 1e9

    # Initialisierung des Modells und des Kalman-Filters
    model = KrylovModel(kf_forces)
    state_columns = ["x0", "y0", "psi", "u", "v", "r"]
    meas_coulumns = ["X", "Y", "heading"]  # NED-Position [m] und Kurs [rad] aus dem Parquet
    input_columns = ["N0", "N1", "delta_r0", "delta_r1"]

    # P_0: Unsicherheit des Anfangszustands (Varianzen)
    P_0 = np.diag([1.0, 1.0,               # x0, y0 [m^2]
                   np.radians(2.0)**2,     # psi [rad^2]
                   0.1**2, 0.1**2,         # u, v [(m/s)^2]
                   np.radians(0.5)**2])    # r [(rad/s)^2]
    # Q: Modellfehler pro Filterschritt (Tuning-Knopf: gross = Messung dominiert)
    Q = np.diag([0.1**2, 0.1**2,
                 np.radians(0.1)**2,
                 0.05**2, 0.05**2,
                 np.radians(0.1)**2])
    # R: Messrauschen von GPS (~1 m) und Kompass (~1 grad)
    R = np.diag([1.0**2, 1.0**2, np.radians(1.0)**2])
    H = np.eye(3, 6)  # gemessen werden die ersten 3 Zustaende (x0, y0, psi)

    kalfil = ExtendedKalmanFilter(data=kf_forces.data, model=model,
                                  state_columns=state_columns,
                                  meas_coulumns=meas_coulumns,
                                  input_columns=input_columns,
                                  P_0=P_0, Q=Q, R=R, H=H)

    result = kalfil.filter(dt=1.0)
    print(result[state_columns].describe().round(3))

    # --- Vergleichsplot: EKF-Schaetzung vs. gemessene Trajektorie ---
    import matplotlib.pyplot as plt

    meas = kf_forces.data
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    ax1.plot(meas["Y"], meas["X"], "k-", label="Messung (GPS)")
    ax1.plot(result["y0"], result["x0"], "r--", label="EKF")
    ax1.set_xlabel("Ost y0 [m]"); ax1.set_ylabel("Nord x0 [m]")
    ax1.axis("equal"); ax1.legend(); ax1.set_title("Trajektorie")

    ax2.plot(meas.index, meas["heading"], "k-", label="Messung")
    psi_wrapped = np.mod(result["psi"] + np.pi, 2 * np.pi) - np.pi  # auf [-pi, pi] wie die Messung
    ax2.plot(result.index, psi_wrapped, "r--", label="EKF")
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("psi [rad]")
    ax2.legend(); ax2.set_title("Kurs")

    ax3.plot(result.index, result["y_X"], label="Innovation X [m]")
    ax3.plot(result.index, result["y_Y"], label="Innovation Y [m]")
    ax3.plot(result.index, np.degrees(result["y_heading"]), label="Innovation psi [deg]")
    ax3.axhline(0, color="k", lw=0.5)
    ax3.set_xlabel("t [s]"); ax3.legend(); ax3.set_title("Innovationen (sollten um 0 pendeln)")

    fig.tight_layout()
    fig.savefig("ekf_trial.png", dpi=150)
    print("Plot gespeichert: ekf_trial.png")
    plt.show()