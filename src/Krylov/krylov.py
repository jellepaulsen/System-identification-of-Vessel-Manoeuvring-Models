import pstats
from time import time
from plot.plot import *
import numpy as np
import math
import pandas as pd
import sympy as sp
import yaml 
import matplotlib
matplotlib.use('QtAgg')  # kein GUI-Fenster, nur Dateien speichern
import matplotlib.pyplot as plt
import cProfile

from dataclasses import dataclass
from Krylov.krylov_data import ShipConfig
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def calc_helper():
    pass

def coeffs(value, a,b,c):
    return a*value**2 + b*value + c

def clean_input_vars(list_of_vars: list, state_columns: list, input_vars: list):
    return [x for i,x in enumerate(list_of_vars) if state_columns[i] in input_vars]


def get_test_input(
    h: float,
    n: int,
    N: float,
    dt: float = 1.0,
    delta_deg: float = 10.0,
) -> pd.DataFrame:
    """
    Build simple zigzag test input with alternating rudder angles.

    The rudder angle alternates between +delta_deg and -delta_deg every h seconds,
    repeated n times. Both propellers use the same revolution N and both rudders
    use the same deflection delta.
    """
    if h <= 0:
        raise ValueError("h must be > 0")
    if n <= 0:
        raise ValueError("n must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    rows = []
    delta_rad = math.radians(delta_deg)
    steps_per_leg = max(1, int(round(h / dt)))

    for leg in range(n):
        delta = delta_rad if leg % 2 == 0 else -delta_rad
        for step in range(steps_per_leg):
            t = leg * h + step * dt
            rows.append(
                {
                    "time": t,
                    "N0": N,
                    "N1": N,
                    "delta_r0": delta,
                    "delta_r1": delta,
                }
            )

    df = pd.DataFrame(rows).set_index("time")
    df.index.name = "time"
    return df





@dataclass
class ZigzagConfig:
    N: float          # propeller RPM
    delta_deg: float  # rudder angle magnitude [deg]
    psi_des: float    # heading threshold per side [deg]
    n_switches: int   # number of rudder reversals
    t0: float         # pre-maneuver acceleration time [s]
    t_aft: float      # coast-down duration after last switch [s]
    dt: float = 0.5   # output timestep [s]


class Krylov_pre_calc:
    def __init__(self, ship_parameters: dict, krylov_parameters: dict):
        self.sd = ShipConfig(**ship_parameters)
        self.kp = krylov_parameters
        self.eps = self.kp["eps"]
        self.psi2p = self.kp["psi_2"] # psi_2 parameters
        self.c2p = self.kp["c_2"] # c_2 parameters
        self.cy_betap = self.kp["cy_beta"] # cy_beta parameters
        # Kalibrierungsfaktoren auf die Rumpfkraefte (aus Trial-Abgleich, siehe krylov.yml)
        self.k_x = self.kp.get("k_x", 1.0)
        self.k_y = self.kp.get("k_y", 1.0)
        self.k_n = self.kp.get("k_n", 1.0)
        self.hydro_mass()

    def hydro_mass(self):
        # calculation of hydrodynamic mass using the Krylov code
        R1_munk=self.kp["R1_munk"]      
        R2_munk=self.kp["R2_munk"]        
        R3_munk=self.kp["R3_munk"]        
        C=self.kp["C"]	           #Lewis coefficient

        self.hydro_mass_dict = {}
        m11 = np.pi * self.sd.rho* self.sd.Tm**2 * C *R1_munk * self.sd.L/2.
        m22 = np.pi * self.sd.rho* self.sd.Tm**2 * self.sd.L/2. * C *R2_munk* 1/2
        m66 = np.pi * self.sd.rho* self.sd.Tm**2 * self.sd.L**2. * C *R3_munk / 24  * self.sd.L
        print(f"m11: {m11:.2f}, m22: {m22:.2f}, m66: {m66:.2f}")
        

        if self.sd.noh > 2:
            print("Krylov code only implemented for monohulls and catamarans, not for trimarans or higher")
        
        elif self.sd.noh == 2:
            qqq = -1.0 + self.sd.dbh / self.sd.B
            Akxx = 2. + np.exp(-qqq)
            Akyy = 2. -0.8*np.exp(-2.*qqq)
            print(f"Akxx: {Akxx:.2f}, Akyy: {Akyy:.2f}")
            self.Corr_x=2.	
            self.Corr_y = 2.-0.5*np.exp(-2.*qqq)
            self.Corr_n=2.-0.65*np.exp(-2.*qqq)
            m = self.sd.CB*self.sd.rho*self.sd.L*self.sd.B*self.sd.Tm*2
            volume = self.sd.L*self.sd.B*self.sd.Tm*self.sd.CB
            self.hydro_mass_dict["m11"] = m11 * Akxx
            self.hydro_mass_dict["m22"] = m22 * Akyy
            self.hydro_mass_dict["m66"] = Akyy*(m66+((self.sd.dbh/2.0)**2)*m22)+Akxx*((self.sd.dbh/2.0)**2)*m11 
            
            self.hydro_mass_dict["izz"] = 409355 # 11115 fixed value for catamaran no idea about dimension 
            # self.hydro_mass_dict["m66"] = self.hydro_mass_dict["izz"] *0.7
            print(f"m11: {self.hydro_mass_dict['m11']:.2f}, m22: {self.hydro_mass_dict['m22']:.2f}, m66: {self.hydro_mass_dict['m66']:.2f}, izz: {self.hydro_mass_dict['izz']:.2f}")
        else: 
            volume = self.sd.CB * self.sd.L * self.sd.B * self.sd.Tm
            m = volume * self.sd.rho
            C = 1.0

            izz = m66 / 1.5
            self.Corr_x = 1.0
            self.Corr_y = 1.0
            self.Corr_n = 1.0

            self.hydro_mass_dict["m11"] = m11 * self.Corr_x
            self.hydro_mass_dict["m22"] = m22 * self.Corr_y
            self.hydro_mass_dict["m66"] = m66 * self.Corr_n
            self.hydro_mass_dict["izz"] = izz


class Krylov_forces(Krylov_pre_calc):
    def __init__(self, 
                 ship_parameters: dict,
                 krylov_parameters: dict,
                 ship_resistance: pd.DataFrame = None,
                 prop_openwater: pd.DataFrame = None,
                 data: pd.DataFrame = None, 
                #  x0: np.array = None, 
                 state_columns: list =["x0", "y0", "psi", "u", "v", "r"],
                 input_columns: list  = ["N0", "N1", "delta_r0", "delta_r1"]
                 ):
        
        super().__init__(ship_parameters, krylov_parameters)
        self.debug = False

        # states = data[state_columns]
        # input = data[input_columns]
        self.data = data
        self.input_columns = input_columns
        self.state_columns = state_columns
        sd = self.sd
        sd.prop_openwater = prop_openwater
        if "m/s" not in ship_resistance.columns:
            ship_resistance = self.add_column(ship_resistance, 0.514444, "kn", "m/s")  # Convert kn to m/s
        sd.ship_resistance = self.add_ct(ship_resistance, sd)
        self.cns = []
        
        # try:
        #     missing = [col for col in input_columns if col not in data.columns]
        # except KeyError as e:
        #     raise KeyError(f"columns not found in df: {missing}")





        # self.test(x0 = states.iloc[0].values, input = input, input_columns = input_columns, sd = sd, eps = self.eps)

    # def test(self, x0 = None, input = None, input_columns = None, sd: ShipConfig = None, eps = None):
    #     self.forces(x0 = x0, input = input, input_columns = input_columns, sd = sd, eps = eps)


    def map_column_names_for_simulation(
        self,
        input_columns_data: list,
        geopos: list,
    ) -> pd.DataFrame:
        """Map raw measurement data to simulation input format.

        Args:
            df: input DataFrame
            columns: input column names to rename
            geopos: [lat_col, lon_col] column names containing position in degrees
            output_columns: output column names, 1:1 with `columns`

        Returns:
            DataFrame with renamed columns and geopos converted to NED metres
            (x+ North, y+ East), origin at first row.
        """
        output_columns = self.input_columns  # For now, we assume the output columns are the same as input columns
        if len(input_columns_data) != len(output_columns):
            raise ValueError("input_columns_data and output_columns must have the same length")

        out = self.data.copy()


        # rename columns 1:1
        rename_map = dict(zip(input_columns_data, output_columns))
        print(f"Renaming columns: {rename_map}")
        out = out.rename(columns=rename_map)

        # convert lat/lon [deg] → NED [m] relative to first position
        # lat_col, lon_col = geopos
        # # use renamed names if the geopos cols were in `columns`
        # lat_out = rename_map.get(lat_col, lat_col)
        # lon_out = rename_map.get(lon_col, lon_col)

        # R = 6_371_000.0  # mean Earth radius [m]
        # lat0 = math.radians(out[lat_out].iloc[0])
        # lon0 = math.radians(out[lon_out].iloc[0])

        # lat_rad = out[lat_out].apply(math.radians)
        # lon_rad = out[lon_out].apply(math.radians)

        # # x+ North, y+ East
        # out[lat_out] = (lat_rad - lat0) * R
        # out[lon_out] = (lon_rad - lon0) * math.cos(lat0) * R

        # self.data = out.rename(columns={lat_out: "x0", lon_out: "y0"})
        
        return out

    def get_x0(self, sog_col: str, cog_col: str, heading_col: str, rot_col: str) -> list:
        """Compute initial state [x0, y0, psi, u, v, r] from first row of self.data.

        SOG/COG are in the earth frame; u/v are resolved into the body frame via psi.
        """
        row = self.data.iloc[0]

        psi = math.radians(row[heading_col])
        sog = row[sog_col]
        cog = math.radians(row[cog_col])
        r   = math.radians(row[rot_col])

        # earth-frame velocity components (NED)
        v_north = sog * math.cos(cog)
        v_east  = sog * math.sin(cog)

        # rotate into body frame
        u =  v_north * math.cos(psi) + v_east * math.sin(psi)
        v = -v_north * math.sin(psi) + v_east * math.cos(psi)

        return [0.0, 0.0, psi, u, v, r]

    def add_ct(self, df: pd.DataFrame, sd: ShipConfig):
        psi1 = (sd.Ta - sd.Tf) /sd.L 
        m = 1
        if "kN" in df.columns:
            m = 1000
        


        for idx, row in df.iterrows():

            fn = row["m/s"] / np.sqrt(sd.L * 9.81)
            psi2 = self.calc_psi2(sd = sd, Fn = fn, psi2p = self.psi2p)
            x, asigma = self.calc_sigma(sd = sd,psi1=psi1, psi2=psi2, Tm=sd.Tm, L=sd.L, askeg=sd.askeg, fr_i=sd.fr_i)

            if sd.noh == 2:
                df.loc[idx, "ct"] = row["kN"]*m / (0.5*sd.rho * row["m/s"]**2 * sd.S)
                df.loc[idx, "ct_kry"] = row["kN"]*m / (0.5*sd.rho * asigma * row["m/s"]**2)
            else: 
                df.loc[idx, "ct"] = row["kN"]*m / (0.5 * sd.rho * sd.S * row["m/s"]**2)
                df.loc[idx, "ct_kry"] = row["kN"]*m / (0.5 * sd.rho * asigma * row["m/s"]**2)

        # print(df)
        return df

    def add_column(self, df: pd.DataFrame, factor: float, column: str, new_column: str):
        df[new_column] = df[column] * factor
        return df

    def timer(func):
        def wrapper(*args, **kwargs):
            start_time = time()
            result = func(*args, **kwargs)
            end_time = time()
            print(f"Execution time: {end_time - start_time} seconds")
            return result

        return wrapper
    
    @timer
    def simulate(self, x0_):
        '''
        input columns: N und Delta_r
        state columns: x0, y0, psi, u, v, r
        x0_: initial state for the integration
        data: DataFrame with the input data, index should be time and columns should include the input_columns and state_columns
        '''
        print("Initializing simulation with Krylov forces...")
        sd = self.sd
        eps = self.eps
        input_columns = self.input_columns
        state_columns = self.state_columns
        # translate to numpy to gain performance
        t_input = self.data.index.to_numpy()
        input_data = self.data[input_columns].to_numpy()


        n_y = len(state_columns) + 3 # length of input_array for rhs
        y_ = np.empty(n_y) # state + forces

        rhs_sym = self.equations()

        rhs_input = state_columns + ["F_X", "F_Y", "M_N"]

        rhs_func = sp.lambdify(rhs_input, rhs_sym, modules="numpy")

        # build interpolation fuctions 
        # RT_interp = interp1d(sd.ship_resistance["m/s"], sd.ship_resistance["kN"], kind="cubic", fill_value="extrapolate")

        rhs = self.make_rhs(rhs_func = rhs_func, 
                            input = input_data, 
                            input_id = t_input, 
                            input_columns = input_columns, 
                            state_columns = state_columns, 
                            y_array = y_, 
                            sd = sd, 
                            eps = eps)



        

        # here the integration of the rhs function needs to be implemented, e.g. with scipy solve_ivp or a custom implementation
        # the output should be a DataFrame with the same columns as state_columns and the same index as data
        
        t = self.data.index
        t_span = [t.min(), t.max()]
        dt_out = 0.01  # fixed 100 Hz output, independent of input sampling rate
        n_out = max(2, int(round((t_span[1] - t_span[0]) / dt_out)) + 1)
        t_eval = np.linspace(t_span[0], t_span[1], n_out)

        # sol = solve_ivp(rhs, t_span, data[state_columns].iloc[0].values, t_eval=t_eval, method='RK45')

        # profiler = cProfile.Profile()
        # profiler.enable()
        
        print("Starting integration...")
        sol = solve_ivp(rhs, t_span, x0_, t_eval=t_eval, method='Radau')
        print("Integration completed.")

        # profiler.disable()
        # stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.print_stats(10)  # Print top 10 functions by cumulative time

        if not sol.success:
            print("Integration failed:", sol.message)


        # print(F"psi: {sol.y[2, -20:]}\n u: {sol.y[3, -20:]}\n v: {sol.y[4, -20:]}\n r: {sol.y[5, -20:]}\n")
        # zero-order-hold resample of the inputs onto the output grid,
        # same rule the rhs applies during integration
        input_idx = np.clip(np.searchsorted(t_input, t_eval, side="right") - 1, 0, None)

        df_sim = pd.DataFrame(sol.y.T, columns=state_columns, index=t_eval)
        df_input = pd.DataFrame(input_data[input_idx], columns=input_columns, index=t_eval)

        forces_list = []
        for k, t_i in enumerate(t_eval):
            idx = input_idx[k]
            x0_k = sol.y[:, k]
            inp_k = input_data[idx]
            beta_eff, beta_eff_sign = self.eff_drift_angle(x0_k, eps)
            kr_X, kr_Y, kr_N, cns = self.krylov_force(x0_k, sd, beta_eff, beta_eff_sign, eps)
            pod_X, pod_Y, pod_N = self.calc_pod_forces(
                input=inp_k, input_columns=input_columns, sd=sd, urx=x0_k[3]
            )
            forces_list.append((kr_X, kr_Y, kr_N, pod_X, pod_Y, pod_N,
                                 kr_X + pod_X, kr_Y + pod_Y, kr_N + pod_N, 
                                 cns))

        df_forces = pd.DataFrame(forces_list,
                                 columns=["kr_X", "kr_Y", "kr_N",
                                          "pod_X", "pod_Y", "pod_N",
                                          "F_X", "F_Y", "M_N", "cns"],
                                 index=t_eval)

        return pd.concat([df_sim, df_input, df_forces], axis=1)
    

    def _make_rhs_with_input(self, N: float, delta_r: float):
        """Build an rhs(t, y) closure for a fixed propeller RPM and rudder angle."""
        rhs_sym = self.equations()
        rhs_input = self.state_columns + ["F_X", "F_Y", "M_N"]
        rhs_func = sp.lambdify(rhs_input, rhs_sym, modules="numpy")
        n_y = len(self.state_columns) + 3
        y_ = np.empty(n_y)
        input_arr = np.array([N, N, delta_r, delta_r], dtype=float)
        sd = self.sd
        eps = self.eps
        input_columns = self.input_columns
        state_columns = self.state_columns

        def rhs(t, y):
            forces, cns = self.forces(x0_=y, input=input_arr,
                                 input_columns=input_columns, sd=sd, eps=eps)
            y_[:len(state_columns)] = y
            y_[len(state_columns):] = forces
            return np.asanyarray(rhs_func(*y_)).ravel()

        return rhs

    @timer
    def simulate_zigzag(self, x0_: list, config: ZigzagConfig) -> pd.DataFrame:
        """IMO zigzag manoeuvre simulation.

        Phases:
          1. ACCEL  – rudder=0, N=config.N for config.t0 seconds
          2. MANEUVER – alternating ±delta_r until heading threshold ±psi_des is
                        reached; repeated config.n_switches times
          3. COAST  – rudder=0, N=0 for config.t_aft seconds
        """
        psi_des_rad = math.radians(config.psi_des)
        delta_rad   = math.radians(config.delta_deg)
        state_columns  = self.state_columns
        input_columns  = self.input_columns
        sd             = self.sd

        # Each entry: (sol, N_seg, delta_seg, phase_label)
        segments = []

        t_now = 0.0
        y_now = np.array(x0_, dtype=float)

        def _integrate(rhs, t_start, duration, y0, event=None):
            t_end  = t_start + duration
            t_eval = np.arange(t_start, t_end + config.dt, config.dt)
            t_eval = np.clip(t_eval, t_start, t_end)
            evs = [event] if event is not None else []
            return solve_ivp(rhs, [t_start, t_end], y0,
                             t_eval=t_eval, events=evs, method='Radau')

        # --- Phase 1: ACCEL ---
        print(f"Zigzag: acceleration phase (t0={config.t0}s, N={config.N} RPM)...")
        rhs = self._make_rhs_with_input(config.N, 0.0)
        sol = _integrate(rhs, t_now, config.t0, y_now)
        segments.append((sol, config.N, 0.0, 'accel'))
        t_now = sol.t[-1]
        y_now = sol.y[:, -1]

        # --- Phase 2: MANEUVER ---
        print(f"Zigzag: manoeuvre phase ({config.n_switches} switches, "
              f"delta={config.delta_deg}°, psi_des={config.psi_des}°)...")
        sign = 1
        for i in range(config.n_switches):
            rudder  = sign * delta_rad
            rhs     = self._make_rhs_with_input(config.N, rudder)
            psi_ref = float(y_now[2])

            def heading_event(t, y, s=sign, ref=psi_ref, des=psi_des_rad):
                return s * (y[2] - ref) - des
            heading_event.terminal  = True
            heading_event.direction = 1   # fires only when value crosses 0 upward

            sol = _integrate(rhs, t_now, 600.0, y_now, event=heading_event)
            segments.append((sol, config.N, rudder, 'maneuver'))
            t_now = sol.t[-1]
            y_now = sol.y[:, -1]
            print(f"  switch {i+1}/{config.n_switches}: "
                  f"t={t_now:.1f}s  psi={math.degrees(float(y_now[2])):.1f}°  "
                  f"next rudder={'port' if sign < 0 else 'stbd'}")
            sign *= -1

        # --- Phase 3: COAST ---
        print(f"Zigzag: coast phase (t_aft={config.t_aft}s)...")
        rhs = self._make_rhs_with_input(0.0, 0.0)
        sol = _integrate(rhs, t_now, config.t_aft, y_now)
        segments.append((sol, 0.0, 0.0, 'coast'))

        # --- Combine segments (drop duplicate boundary point between segments) ---
        parts_t, parts_y, parts_inp, parts_phase = [], [], [], []
        for k, (s, N_seg, delta_seg, phase) in enumerate(segments):
            sl = slice(1, None) if k > 0 else slice(None)
            parts_t.append(s.t[sl])
            parts_y.append(s.y[:, sl])
            n = s.t[sl].shape[0]
            parts_inp.append(
                np.tile([N_seg, N_seg, delta_seg, delta_seg], (n, 1))
            )
            parts_phase.extend([phase] * n)

        t_all   = np.concatenate(parts_t)
        y_all   = np.hstack(parts_y)
        inp_all = np.vstack(parts_inp)

        df_sim   = pd.DataFrame(y_all.T,  columns=state_columns,  index=t_all)
        df_input = pd.DataFrame(inp_all,  columns=input_columns,  index=t_all)
        df_phase = pd.DataFrame({'phase': parts_phase},            index=t_all)

        # Recompute forces at every output point (same approach as simulate)
        forces_list = []
        for k in range(len(t_all)):
            x0_k = y_all[:, k]
            inp_k = inp_all[k]
            beta_eff, beta_eff_sign = self.eff_drift_angle(x0_k, self.eps)
            kr_X, kr_Y, kr_N = self.krylov_force(x0_k, sd, beta_eff, beta_eff_sign, self.eps)
            pod_X, pod_Y, pod_N = self.calc_pod_forces(
                input=inp_k, input_columns=input_columns, sd=sd, urx=x0_k[3]
            )
            forces_list.append((kr_X, kr_Y, kr_N, pod_X, pod_Y, pod_N,
                                 kr_X + pod_X, kr_Y + pod_Y, kr_N + pod_N))

        df_forces = pd.DataFrame(forces_list,
                                 columns=["kr_X", "kr_Y", "kr_N",
                                          "pod_X", "pod_Y", "pod_N",
                                          "F_X", "F_Y", "M_N"],
                                 index=t_all)

        print("Zigzag simulation completed.")
        return pd.concat([df_sim, df_input, df_forces, df_phase], axis=1)

    def equations(self, name: str= "krylov"):
        # movement equations for the forces, to be simplified and lambdified with sympy

        F_X, F_Y, M_N = sp.symbols('F_X F_Y M_N')
        m_x, m_y, m_n = sp.symbols('m_x m_y m_n')
        # dt = sp.symbols('dt')
        x0, y0, psi, u, v, r = sp.symbols('x0 y0 psi u v r')
        if name == "krylov":
            rhs_sym = sp.Matrix([
                (u * sp.cos(psi) - v * sp.sin(psi)),
                (u* sp.sin(psi) + v * sp.cos(psi)),
                r,
                (F_X + m_y*r*v)/ m_x,
                (-m_x*r*u + F_Y)/m_y,
                M_N / m_n
            ])
        elif name == "Winkler":
            rhs_sym = sp.Matrix([
                (u * sp.cos(psi) - v * sp.sin(psi)),
                (u* sp.sin(psi) + v * sp.cos(psi)),
                r,
                (F_X + m_y*r*v)/ m_x,
                (-m_x*r*u + F_Y)/m_y,
                M_N / m_n
            ])

        rhs_sym = rhs_sym.subs({
            m_x: self.hydro_mass_dict["m11"] + self.sd.m,
            m_y: self.hydro_mass_dict["m22"] + self.sd.m,
            m_n: self.hydro_mass_dict["m66"] + self.hydro_mass_dict["izz"]
        })

        return rhs_sym
    
    def make_rhs(self, rhs_func, input, input_id, input_columns, state_columns, y_array, sd, eps):

        def rhs(t, y):
            x0_ = y
            
            # self.data.iloc[t][self.state_columns].values()
            # Zero-Order-Hold -> 
            i = np.searchsorted(input_id, t, side="right") - 1 # finde the corresponding index from input data. -1 take last step not next step
            X, Y, N, cns = self.forces(x0_ = x0_, input = input[i], input_columns = input_columns, sd = sd, eps = eps)
            y_array[:len(state_columns)] = x0_
            y_array[len(state_columns):] = [X, Y, N]

            return np.asanyarray(rhs_func(*y_array)).ravel()
        return rhs

    def forces(self,
                x0_ = None,
                input: pd.DataFrame = None,
                input_columns = None,
                sd: ShipConfig = None,
                eps = None
                ):
        # adding force equations to be simpified and lambdified with sympy
        # If None take value from object:
        if x0_ is None:
            x0_ = input.iloc[0][input_columns].values()

        # if x0_[3] > 20 or x0_[4] > 20:
        #     print(f"blitzmeister")
        #     return None, None, None
        # adding wave induced velocities to 
        # need to be added

        
        beta_eff, beta_eff_sign = self.eff_drift_angle(x0_, eps) # self.beta_eff
        # print(f"u: {x0_[3]}, v: {x0_[4]}, r: {x0_[5]}, beta_eff: {beta_eff}")

        # print(f"Calculating forces for state: {x0_}, beta_eff: {beta_eff}, beta_eff_sign: {beta_eff_sign}")
        # x0_ = self.wave_induced_velocities(x0_)
        kr_X, kr_Y, kr_N, cns = self.krylov_force(x0_, sd, beta_eff, beta_eff_sign, eps)
        pod_X, pod_Y, pod_N = self.calc_pod_forces(
        # pod_X, pod_Y, pod_N = self.calc_pod_rud_force(
                        input = input,
                        input_columns = input_columns,
                        sd = sd,
                        # N =[input.loc[:, input_columns].values[0]],  # pod forces need to be time dependent, here only the first value is taken for testing
                        urx = x0_[3]
                    )
        # print(f"Krylov forces: X_kr={kr_X:.2f} N, Y_kr={kr_Y:.2f} N, N_kr={kr_N:.2f} Nm")
        # print(f"Pod forces: X_pod={pox_X:.2f} N, Y_pod={pod_Y:.2f} N, N_pod={pod_N:.2f} Nm")

        # print(f"X: {kr_X}, {pox_X:.2f} N, Y: {kr_Y}, {pod_Y:.2f} N, N: {kr_N}, {pod_N:.2f} Nm")



        # print(f"Total forces: X= kr:{kr_X} + {pox_X:.2f} N, Y= kr:{kr_Y } + { pod_Y:.2f} N, N= kr:{kr_N } + { pod_N:.2f} Nm")
        # print(f"urx: {x0_[3]}")
        
        return kr_X + pod_X, kr_Y +pod_Y, kr_N +pod_N, cns

    
    #------------------------------------------------------
    # wave induced velocities
    def wave_induced_velocities(self, x0_):
        # need to be added
        ucx = 0
        ucy = 0
        x0_[3] = x0_[3] + ucx 
        x0_[4] = x0_[4] + ucy 
        return x0_



    #------------------------------------------------------
    # Krylov Force
    
    def krylov_force(self, x0_= None, sd: ShipConfig = None, beta_eff: float = None, beta_eff_sign: float = None, eps = None):
        '''
        Ta = t aft
        Tf = t fore

        '''    
        # not sure if needed, i suppose it needs to be only symbolic (sympy) for the lambdification
        if x0_ is None:
            x0_ = self.x0
        # If no initial values for Ta and Tf, take mean draft as inital value for both
        if sd.Ta is None or sd.Tf is None:
            sd.Ta, sd.Tf = sd.Tm, sd.Tm 
            # später kann es auch aus dem mittleren Tiefgang und dem Trim aus dem IMU berechnet werden zu Fahrtantritt


        L, LB,Tm, TmL, cp, xtg, rho = sd.L, sd.LB, sd.Tm, sd.TmL, sd.cp, sd.xtg, sd.rho

        Uchar = np.sqrt(x0_[3]**2 + x0_[4]**2) # speed

        Fn = Uchar / np.sqrt(L * 9.81) # Froude number

        psi1 = (sd.Ta - sd.Tf) /L # tangent ot static trim angle

        # cx0 = self.get_cx0(sd=sd, ship_resistance=sd.ship_resistance, Uchar=Uchar)
        cx0 = np.interp(Uchar, sd.ship_resistance["m/s"] , sd.ship_resistance["ct_kry"])  # interpolierter Widerstand bei speed Uchar 
        
        # if sd.noh == 2:
        #     cx0 = RTx0 / (rho * Uchar**2 * sd.S)
        # else:
        #     cx0 = RTx0 / (0.5 * rho * Uchar**2 * sd.S)

        cx0 = np.clip(cx0, 0.0000000001, 0.075) # limit cx0 to avoid numerical issues, lower limit is arbitrary and can be adjusted based on expected range of cx0 values

        # print(f"Interpolated resistance RT at speed {Uchar:.2f} m/s, cx0: {cx0:.4f}")

        # calc psi2
        check = 0
        psi2p = self.psi2p
        for fnr in psi2p.values():                      # fnr = Fn range
            if fnr["fn"][0] <= Fn <= fnr["fn"][1]:
                # print(f"fnr: {fnr.keys()}")
                for xgr in fnr["xg"].values():
                    if xgr["r"][0] <= xtg <= xgr["r"][1]:
                        a1 = coeffs(xtg, *xgr["a1"])
                        b1 = coeffs(xtg, *xgr["b1"])
                        c1 = coeffs(xtg, *xgr["c1"])
                        check += 1

        if check == 1:
            psi2 = coeffs(Fn, a1, b1, c1)
        else:
            print(f"Fn: {Fn}, xtg: {xtg}, check: {check}")
            print("Fn or xg out of range for psi2 calculation")
            print("If check > 1 then multiple ranges are overlapping CODE INCORRECT")
            psi2 = 0.0

        # calc sigam, Asigma
        psi_res = psi1 + psi2
        if sd.shiptype == 1:
            sigma = 1.-(3./(20.-sd.fr_i))*(sd.askeg/(L*Tm))+(0.054/(TmL)) * psi_res
        elif sd.shiptype == 2:
            sigma = 0.975 + 0.054/TmL * psi_res
        elif sd.shiptype == 3:
            sigma = 0.962 + 0.054/ TmL * psi_res

        
        # lower limit
        if sigma <= 0.93:
            sigma = 0.93 
        
        # lateral area A_{L sigma}
        Asigma = L * Tm * sigma 



        # beta_eff, beta_eff_sign = self.eff_drift_angle(x0) # self.beta_eff
        # psi2 = self.calc_psi2(sd = sd, Fn = Fn, psi2p = self.psi2p)
        # for faster calc the last Fn should be saved and only updated if Fn changes significantly, same for beta_eff

        # sigma, Asigma = self.calc_sigma(sd = sd,psi1=psi1, psi2=psi2, Tm=sd.Tm, L=sd.L, askeg=sd.askeg, fr_i=sd.fr_i)

        # c2, c3 = self.calc_cs(L= sd.L,
        #             B= sd.B,
        #             LB = sd.LB,
        #             TmL= sd.TmL,
        #             c2p= self.c2p,
        #             x0_= x0_,
        #             cp= sd.cp,
        #             sigma= sigma)
        

        # calc c2 , c3
        for tml in self.c2p.values():
            if "tml" in tml.keys():
                if tml["tml"][0] <= TmL <= tml["tml"][1]:
                    c2_a3 = coeffs(TmL, *tml["a3"])
                    c2_b3 = coeffs(TmL, *tml["b3"])
        
        c2_a1 = 54.46*cp - 59.43
        c2_b1 = -31.44*cp + 46.8

        U = c2_a1 * sigma + c2_b1

        for U_ in self.c2p["U"].values():

            # print(f"u0: {U['r'][0]}, u1: {U['r'][1]}, x0_3: {x0_[3]}")

            if U_["r"][0] <= x0_[3] <= U_["r"][1]:
                c2_a2 = coeffs(x0_[3], *U_["a2"])
                c2_b2 = coeffs(x0_[3], *U_["b2"])
        
        Q = c2_a2 * (LB) + c2_b2
        c2 = np.clip(c2_a3 * Q + c2_b3, 0.3, 1.6)

        c3_a2 = coeffs(TmL, 2.269, -0.5805, 0.00183)
        c3_b2 = coeffs(TmL, -27.7, 6.428, -0.01749)

        if cp <= 0.72:
            c3_a1 = coeffs(cp, 24.65, -29.67, 7.547)
        elif cp > 0.72:
            c3_a1 = coeffs(cp, 0, 5.917, 5.3)

        if cp <= 0.68:
            c3_b1 = coeffs(cp, -60.44, 74.61, 9.255)
        elif cp > 0.68:
            c3_b1 = coeffs(cp, 0, 10.08, 20.34)

        U_1 = c3_a1 * LB + c3_b1
        # print(f"c3: {c3_a2, U_1, c3_b2}")
        c3 = np.clip(c3_a2 * U_1 + c3_b2, 0.0, 0.35)


        # calc_cy_beta


        # cy_beta = self.calc_cy_beta(LB= sd.LB, TmL= sd.TmL, cp= sd.cp, sigma= sigma, beta_eff= beta_eff, beta_eff_sign= beta_eff_sign, c2= c2, c3= c3)
        for lb in self.cy_betap.values():
            if lb["r"][0] <= LB <= lb["r"][1]:
                a1 = a1_fallback = None
                b1 = b1_fallback = None
                for sigmas in lb["sigma"].values():
                    if sigmas["r"][0] is None and sigmas["r"][1] is None:
                        a1_fallback = coeffs(LB, *sigmas["a1"])
                        b1_fallback = coeffs(LB, *sigmas["b1"])
                    elif sigmas["r"][0] <= sigma <= sigmas["r"][1]:
                        a1 = coeffs(LB, *sigmas["a1"])
                        b1 = coeffs(LB, *sigmas["b1"])
                if a1 is None:
                    a1, b1 = a1_fallback, b1_fallback

        a2 = coeffs(TmL, 16.67, -11.92, 0.06)
        b2 = coeffs(TmL, 261.1, 213.6, 2.468)

        a3 = coeffs(cp, 0.2392, -0.4009, 0.1815)
        b3 = coeffs(cp, 0.4033, -0.6965, 0.3263)

        U_2 = a1 * LB + b1
        Q = a2 * U_2 + b2

        # cy_beta_2 = np.clip(a3 * Q + b3, 0.0, 0.5)
        cy_beta_2 = np.clip(a3 * Q + b3, 0.0, 1)
        print(f"cy_beta_2: {cy_beta_2}")
        cy_beta = 0.5* cy_beta_2 * np.sin(2.* beta_eff)* np.cos(beta_eff) + (c2*(np.sin(beta_eff)**2)+ c3*(np.sin(2*beta_eff)**4))* beta_eff_sign
        
        term1 = cy_beta_2 * np.sin(2*beta_eff) * np.cos(beta_eff)
        term2_mag = c2 * np.sin(beta_eff)**2 + c3 * np.sin(2*beta_eff)**4
        term2 = term2_mag * beta_eff_sign

#         print(
#             f"beta_eff={beta_eff}, "
#             f"np.sign(beta_eff)={np.sign(beta_eff)}, "
#             f"beta_eff_sign={beta_eff_sign}, "
#             f"term1={term1}, term2_mag={term2_mag}, term2={term2}, "
#             f"cy_beta={term1 + term2}"
# )

        # print(f"cy_beta: {cy_beta}, c2: {c2}, c3: {c3}, beta_eff: {beta_eff}, beta_eff_sign: {beta_eff_sign}")
        # calc ms

        # ms = self.calc_ms(TmL= sd.TmL, sigma= sigma, LB= sd.LB, cp= sd.cp)
        ms = [0,0,0,0]

        # ----m1------
        a1 = coeffs(TmL, -0.1317, 0.05358, 0.000181)
        b1 = coeffs(TmL, -2.361, 0.8653, -0.000161)

        if cp <= 0.72:
            U0 = coeffs(sigma, -235, 474.2, 235.8)
            SCP = coeffs(cp, -74.67, 110.9, -39.64)
        elif cp > 0.72:
            U0 = coeffs(sigma, -210, 422.9, 207.2)
            SCP = coeffs(cp, 12, -8.8, -0.64)
        
        UUU = U0 + SCP

        if UUU >= 4:
            Su = -1.3 * UUU + 7.8
            Sv0 = coeffs(LB, 0.02333, -0.045, 1.187)
        else:
            Su = -1.3 * UUU + 2.6
            Sv0 = coeffs(LB, 0.02333, -0.045, 1.187) + 0.01 * UUU 

        S = Su + Sv0
        ms[0] = np.clip(a1 * S + b1,0.02, 0.08)

        # ----m2------
        ms[1] = np.maximum(-(np.log(1.023 * sigma))/ (11.6* sigma -9.29), -0.01)

        # ----m3------ NOTE!!!!: sigma is clipped therefore it changes the whole following code

        sigma_m34 = np.minimum(sigma, 1)

        a1 = 31.26 -9.0146 * np.exp(0.066947* LB)
        b1 = 8.6245 * np.exp(0.071419* LB) - 32.26

        a2 = (np.exp(8.20939* cp)* 0.7728*0.001-1.873)*0.001
        b2 = (np.exp(7.47893* cp)*0.4404 * 0.01+5.709)*0.01

        UUUU = (a1 * sigma_m34 + b1)/ (sigma_m34 -1.029)

        ms[2] = np.clip(a2 * UUUU + b2, 0.016, 0.054)
    
        # ----m4------
        if TmL <= 0.028:
            Sm4 = coeffs(TmL, -71.88, 4.238, -0.066)
        elif 0.028 < TmL <= 0.04:
            Sm4 = coeffs(TmL, -9.375, 0.8875, 0.0121)
        else:
            Sm4 = coeffs(TmL, -3.833, 0.415, -0.01117)
        
        if 0.55 <= cp <= 0.64:
            U0 = coeffs(cp, -140.62, 180.62, 53.35)
        elif 0.64 < cp <= 0.74:
            U0 = coeffs(cp, -56.67, 75.1, -20.2)
        else:
            U0 = coeffs(cp, -216.7, 312.8, 108.51)
        
        if sigma_m34 <= 0.96:
            Ss = coeffs(sigma_m34, 1900, -3696, 1796)
        elif sigma_m34 > 0.96:
            Ss = coeffs(sigma_m34, 391.7, -810.4, 415.8)
        
        UUUUU = U0 + Ss
        Su = 0.00827 * UUUUU - 0.017

        ms[3] = np.clip(Sm4 + Su, 0.03, 0.04)

        #calc cn_beta

        # cn_beta, cxb =  self.calc_cn_beta(cx0 = cx0, beta_eff= beta_eff, m= ms)
        cn_beta = ms[0]*np.sin(2*beta_eff)+ms[1]*np.sin(beta_eff)+ms[2]*(np.sin(2*beta_eff)**3) + ms[3] * (np.sin(2*beta_eff)**5)
        

    

        cxb = -self.kp["a1x"] * np.sin((np.pi-np.arcsin(cx0/self.kp["a1x"]))*(1-(abs(beta_eff)*180/np.pi/self.kp["psix"])))

        # print(f"cxb: {round(cxb, 5)}, beta_eff: {round(beta_eff, 5)}, cx0: {round(cx0, 5)}, uchar: {round(Uchar, 2)}, u: {round(x0_[3], 2)}, v: {round(x0_[4], 2)}, r: {round(x0_[5], 2)}")
# 
        # print(f"cx0: {cx0}, cxb: {cxb}")
        # calc_cn
        # cn = self.calc_cn(x0_= x0_, sd= sd, c2= c2, cn_beta= cn_beta, beta_eff= beta_eff, Tml= sd.TmL, sigma= sigma, LB= LB, Uchar= Uchar, eps= self.eps)

        cn0 = 0.059*c2
        cnw2= (0.739 +8.7 * TmL)*(1.611*(sigma_m34**2)-2.873*sigma_m34+1.33)

        a1 = 0.09-cnw2 - 0.0033*(LB -7)-20*((TmL-0.005)**2)+ 0.4*(sigma_m34-0.9)+ 0.05*(sd.CM-0.9)
        a2 = 0.008*LB + 0.9 *(TmL -0.05) + 0.45*(sigma_m34-0.955)

        cnw = cnw2 + a1 *abs(np.sin(beta_eff))+ a2 * (1-np.cos((2*np.pi-4*abs(beta_eff))*np.cos(beta_eff)+0.1*abs(np.sin(2*beta_eff))))

        # omega is rate of turn

        if Uchar > eps:
            omega_strich = x0_[5] * L/ Uchar
            omega_large = omega_strich /np.sqrt(1+omega_strich**2) 
        else:
            omega_large = 1.0 

        
        # according to S.Winkler no negative sign in the first term
        # cnom = cn0*abs(x0_[5])* x0_[5]*L**2 - cnw/ np.pi*(Uchar**2 +(x0_[5]**2)*L**2)* np.sin(np.pi*omega_large)
        cnom = -cn0*abs(x0_[5])* x0_[5]*L**2 - cnw/ np.pi*(Uchar**2 +(x0_[5]**2)*L**2)* np.sin(np.pi*omega_large)
        

        cn = cnom +cn_beta*Uchar**2 # called cn
        cns = [cn, cnom, cn_beta, Uchar, cn_beta*Uchar**2]
        # print(f"cn: {cn}, cnom: {cnom}, cn_beta: {cn_beta}, cnb*U: {cn_beta*Uchar**2}, Uchar: {Uchar}")
        # print(f"asigma: {Asigma}, sigma: {sigma}, s: {sd.S}")

        # if self.debug:
            # print(f"cn_beta: {cn_beta}, ms: {ms}, beta_eff: {beta_eff}")
            # print(f"cn: {cn:.5f}, cnom: {cnom:.5f}, cn_beta*U²: {cn_beta*Uchar**2:.5f}")
            # print(f"M_N_hull: {cn * rho*Asigma*L/2:.1f} Nm")
            # print(f"Asigma: {Asigma:.2f} m², sigma: {sigma:.3f}, LB: {sd.LB:.2f}, TmL: {sd.TmL:.4f}")

        Umnos_cn = rho*Asigma*L/2 
        Umnos_fo = rho*Asigma*(Uchar**2)/2
        # print(f" Umnos_cn: {Umnos_cn}, {cn}, {self.Corr_x}")

        # print(f"cn: {cn}, cnom: {cnom}, cn_beta: {cn_beta}")
        # print(f" x {cxb * Umnos_fo     * self.Corr_x},Y: {cy_beta * Umnos_fo * self.Corr_y}, M: {cn * Umnos_cn * self.Corr_n }")
        # correction factor are different to the given krylov code. Original code is is deplayed afterwards
        return (
            self.k_x * cxb * Umnos_fo     * self.Corr_y,      # corr_y
            self.k_y * cy_beta * Umnos_fo * self.Corr_n,      # corr_n
            self.k_n * cn * Umnos_cn * self.Corr_x,            # corr_X
            cns

        )
    
    def get_cx0(self, sd: ShipConfig = None, ship_resistance = None, Uchar = None, ):
        # interpolate zerodrift resistance from resistance curve
        # print(ship_resistance)
        RTx0 = np.interp(Uchar, ship_resistance["kn"] , ship_resistance["kN"])

        # NOTE: For catamaran the wetted area of demi hull is used!!! Whereas RT is for the whole ship!!

        # resolve devision by zero error for speeds very close to zero setting speed to eps 
        # not sure if suitable
        if Uchar < 0.1 and Uchar >= 0:
            Uchar = 0.1
        elif Uchar > -0.1 and Uchar < 0:
            Uchar = -0.1

        if sd.noh == 2:
            return RTx0 / (sd.rho * Uchar**2 * sd.S)
        return RTx0 / (0.5 * sd.rho * Uchar**2 * sd.S) 

    
    def eff_drift_angle(self, x0_, eps):

        uchar = np.sqrt(x0_[3]**2 + x0_[4]**2)
        if uchar <= eps:
            beta_eff = 0.0  # stehendes Schiff: kein Driftwinkel
        elif abs(x0_[3]) > eps:
            beta_eff = - np.arctan2(x0_[4], x0_[3])
        else:
            beta_eff = - np.pi/2 * np.where(x0_[4]>0, 1, -1)  # reine Drift (u≈0, v≠0)

        # print(f"beta_eff: {round(beta_eff*180/np.pi, 5)}")<Fuchar
        sign = np.where(beta_eff>0, 1, -1)

        # was könnte das sein?
        # if x0_[5] < 0.0:
        #     beta_eff = np.pi* sign

        if abs(beta_eff) < eps:
            beta_eff = 0.0
            sign = 1

        return beta_eff, sign
    
    def calc_psi2(self, sd,  Fn,  psi2p):
        def calc_single(fn_value):
            check = 0
            for fnr in psi2p.values():                      # fnr = Fn range
                if fnr["fn"][0] <= fn_value <= fnr["fn"][1]:
                    for xgr in fnr["xg"].values():
                        if xgr["r"][0] <= sd.xtg <= xgr["r"][1]:
                            a1 = coeffs(sd.xtg, *xgr["a1"])
                            b1 = coeffs(sd.xtg, *xgr["b1"])
                            c1 = coeffs(sd.xtg, *xgr["c1"])
                            check += 1
            if check == 1:
                return coeffs(fn_value, a1, b1, c1)

            print(f"Fn: {fn_value}, xtg: {sd.xtg}, check: {check}")
            print("Fn or xg out of range for psi2 calculation")
            print("If check > 1 then multiple ranges are overlapping CODE INCORRECT")
            return 0.0

        if np.isscalar(Fn):
            return calc_single(Fn)

        if isinstance(Fn, pd.Series):
            return Fn.apply(calc_single)

        Fn_array = np.asarray(Fn)
        return np.array([calc_single(fn_value) for fn_value in Fn_array])


    def calc_sigma(self, sd, psi1, psi2, Tm, L, askeg, fr_i):
        TmL = Tm/L
        psi_res = psi1 + psi2
        if sd.shiptype == 1:
            sigma = 1.-(3./(20.-fr_i))*(askeg/(L*Tm))+(0.054/(TmL)) * psi_res
        elif sd.shiptype == 2:
            sigma = 0.975 + 0.054/TmL * psi_res
        elif sd.shiptype == 3:
            sigma = 0.962 + 0.054/ TmL * psi_res
        
        # lower limit
        if sigma <= 0.93:
            sigma = 0.93 
        
        # lateral area A_{L sigma}
        Asigma = L * Tm * sigma 
        return sigma, Asigma

    # def calc_cs(self, L, B, TmL, c2p, x0_, cp, sigma, LB):    
    #     for tml in c2p.values():
    #         if "tml" in tml.keys():
    #             if tml["tml"][0] <= TmL <= tml["tml"][1]:
    #                 c2_a3 = coeffs(TmL, *tml["a3"])
    #                 c2_b3 = coeffs(TmL, *tml["b3"])
        
    #     c2_a1 = 54.46*cp - 59.43
    #     c2_b1 = -31.44*cp + 46.8

    #     U = c2_a1 * sigma + c2_b1

    #     for U in c2p["U"].values():
    #         if U["r"][0] <= x0_[3] <= U["r"][1]:
    #             c2_a2 = coeffs(x0_[3], *U["a2"])
    #             c2_b2 = coeffs(x0_[3], *U["b2"])
        
    #     Q = c2_a2 * (L/ B) + c2_b2
    #     c2 = np.clip(c2_a3 * Q + c2_b3, 0.3, 1.6)

    #     c3_a2 = coeffs(TmL, 2.269, -0.5805, 0.00183)
    #     c3_b2 = coeffs(TmL, -27.7, 6.428, -0.01749)

    #     if cp <= 0.72:
    #         c3_a1 = coeffs(cp, 24.65, -29.67, 7.547)
    #     elif cp > 0.72:
    #         c3_a1 = coeffs(cp, 0, 5.917, 5.3)

    #     if cp <= 0.68:
    #         c3_b1 = coeffs(cp, -60.44, 74.61, 9.255)
    #     elif cp > 0.68:
    #         c3_b1 = coeffs(cp, 0, 10.08, 20.34)

    #     U = c3_a1 * LB + c3_b1
    #     c3 = np.clip(c3_a2 * U + c3_b2, 0.0, 0.35)

    #     return c2, c3

    # def calc_cy_beta(self, LB, TmL, cp, sigma, beta_eff, beta_eff_sign, c2, c3):
    #     for lb in self.cy_betap.values():
    #         if lb["r"][0] <= LB <= lb["r"][1]:
    #             a1 = a1_fallback = None
    #             b1 = b1_fallback = None
    #             for sigmas in lb["sigma"].values():
    #                 if sigmas["r"][0] is None and sigmas["r"][1] is None:
    #                     a1_fallback = coeffs(LB, *sigmas["a1"])
    #                     b1_fallback = coeffs(LB, *sigmas["b1"])
    #                 elif sigmas["r"][0] <= sigma <= sigmas["r"][1]:
    #                     a1 = coeffs(LB, *sigmas["a1"])
    #                     b1 = coeffs(LB, *sigmas["b1"])
    #             if a1 is None:
    #                 a1, b1 = a1_fallback, b1_fallback

    #     a2 = coeffs(TmL, 16.67, -11.92, 0.06)
    #     b2 = coeffs(TmL, 261.1, 213.6, 2.468)

    #     a3 = coeffs(cp, 0.2392, -0.4009, 0.1815)
    #     b3 = coeffs(cp, 0.4033, -0.6965, 0.3263)

    #     U = a1 * LB + b1
    #     Q = a2 * U + b2

    #     cy_beta_2 = np.clip(a3 * Q + b3, 0.0, 0.5)
    #     cy_beta = 0.5* cy_beta_2 * np.sin(2.* beta_eff)* np.cos(beta_eff) + (c2*(np.sin(beta_eff)**2))+ c3*(np.sin(2*beta_eff)**4)* beta_eff_sign
    #     return cy_beta


    # def calc_ms(self, TmL, sigma, LB, cp):
    #     # ----m1------
    #     a1 = coeffs(TmL, -0.1317, 0.05358, 0.000181)
    #     b1 = coeffs(TmL, -2.361, 0.8653, -0.000161)

    #     if cp <= 0.72:
    #         U0 = coeffs(sigma, -235, 474.2, 235.8)
    #         SCP = coeffs(cp, -74.67, 110.9, -39.64)
    #     elif cp > 0.72:
    #         U0 = coeffs(sigma, -210, 422.9, 207.2)
    #         SCP = coeffs(cp, 12, -8.8, -0.64)
        
    #     UUU = U0 + SCP

    #     if UUU >= 4:
    #         Su = -1.3 * UUU + 7.8
    #         Sv0 = coeffs(LB, 0.02333, -0.045, 1.187)
    #     else:
    #         Su = -1.3 * UUU + 2.6
    #         Sv0 = coeffs(LB, 0.02333, -0.045, 1.187) + 0.01 * UUU 

    #     S = Su + Sv0
    #     m1 = np.clip(a1 * S + b1,0.02, 0.08)

    #     # ----m2------
    #     m2 = np.maximum(-(np.log(1.023 * sigma))/ (11.6* sigma -9.29), -0.01)

    #     # ----m3------ 
    #     sigma = np.maximum(sigma, 1)

    #     a1 = 31.26 -9.0146 * np.exp(0.066947* LB)
    #     b1 = 8.6245 * np.exp(0.071419* LB) - 32.26

    #     a2 = (np.exp(8.20939* cp)* 0.7728*0.001-1.873)*0.001
    #     b2 = (np.exp(7.47893* cp)*0.4404 * 0.01+5.709)*0.01

    #     UUUU = (a1 * sigma + b1)/ (sigma -1.029)

    #     m3 = np.clip(a2 * UUUU + b2, 0.016, 0.054)
    
    #     # ----m4------
    #     if TmL <= 0.028:
    #         Sm4 = coeffs(TmL, -71.88, 4.238, -0.066)
    #     elif 0.028 < TmL <= 0.04:
    #         Sm4 = coeffs(TmL, -9.375, 0.8875, 0.0121)
    #     else:
    #         Sm4 = coeffs(TmL, -3.833, 0.415, -0.01117)
        
    #     if 0.55 <= cp <= 0.64:
    #         U0 = coeffs(cp, -140.62, 180.62, 53.35)
    #     elif 0.64 < cp <= 0.74:
    #         U0 = coeffs(cp, -56.67, 75.1, -20.2)
    #     else:
    #         U0 = coeffs(cp, -216.7, 312.8, 108.51)
        
    #     if sigma <= 0.96:
    #         Ss = coeffs(sigma, 1900, -3696, 1796)
    #     elif sigma > 0.96:
    #         Ss = coeffs(sigma, 391.7, -810.4, 415.8)
        
    #     UUUUU = U0 + Ss
    #     Su = 0.00827 * UUUUU - 0.017

    #     m4 = np.clip(Sm4 + Su, 0.03, 0.04)

    #     return [m1, m2, m3, m4]

    # def calc_cn_beta(self, cx0, beta_eff, m):
    #     beta = beta_eff
    #     cn_beta = m[0]*np.sin(2*beta)+m[1]*np.sin(beta)+m[2]*(np.sin(2*beta)**3) + m[3] * (np.sin(2*beta)**5)
    #     # if cx0/self.kp['a1x'] > 1 or cx0/self.kp['a1x'] < -1:
    #         # print(f"Invalid arcsin input: {cx0/self.kp['a1x']}, {cx0}, {self.kp['a1x']}")

    #     cxb = -self.kp["a1x"] * np.sin((np.pi-np.arcsin(cx0/self.kp["a1x"]))*(1-(abs(beta)*180/np.pi/self.kp["psix"])))
    #     return cn_beta, cxb
    
    # def calc_cn(self, x0_, sd, c2, cn_beta, beta_eff, Tml, sigma, LB, Uchar, eps):
    #     cn0 = 0.059*c2
    #     cnw2= (0.739 +8.7 * Tml)*(1.611*(sigma**2)-2.873*sigma+1.33)

    #     a1 = 0.09-cnw2 - 0.0033*(LB -7)-20*((Tml-0.005)**2)+ 0.4*(sigma-0.9)+ 0.05*(sd.CM-0.9)
    #     a2 = 0.008*LB + 0.9 *(Tml -0.05) + 0.45*(sigma-0.955)

    #     cnw = cnw2 + a1 *abs(np.sin(beta_eff))+ a2 * (1-np.cos( (2*np.pi-4*abs(beta_eff))*np.cos(beta_eff)+0.1*abs(np.sin(2*beta_eff))))

    #     # omega is rate of turn

    #     if Uchar > eps:
    #         omega_strich = x0_[5] * sd.L/ Uchar
    #         omega_large = omega_strich /np.sqrt(1+omega_strich**2)
    #     else:
    #         omega_large = 1.0 

    #     cnom = -cn0*abs(x0_[5])* x0_[5]*sd.L**2 - cnw/ np.pi*(Uchar**2 +(x0_[5]**2)*sd.L**2)* np.sin(np.pi*omega_large)

    #     return cnom +cn_beta*Uchar**2 # called cn

    # End Krylov Force
    #------------------------------------------------------

    # Propeller Forces
    def calc_propeller_Thrust(self, N: list,  Thr: list):
        pass
        # in init



    def calc_pod_forces(self, input: np.ndarray, input_columns: list, sd: ShipConfig, urx: float):
        # changed from original kryov code: each pod gets its own delta_r and therefore the forces and moments differ 
        # NOTE!!: pod thrust is first assmued to be in the rotating center of the pod. Later the effects of another lever arm can be added 
        # the pod forces are acting through the pod rotation center on the ship therefore no leverarm is assumed for either azipull or push thrusters
        # this code only works with even numbers of pods which are symetrically arranged at the back of the ship

        '''
        input: pd.DataFrame with columns for each pod's thrust and deflection angle, e.g. "N0", "delta_r0", "N1", "delta_r1", etc.
        input_columns: list of column names in the input DataFrame that correspond to the pod
        Thr: empty list with length equal to number of pods
        dbh: distance between the centerlines of the two hulls (for catamarans)
        lop: list of "stb" or "ps" indicating the side of each pod (for catamarans)
        lcg: longitudinal center of gravity of the ship
        urx: surge velocity of the ship
        w: wake fraction
        D_p: propeller diameter
        prop_openwater: DataFrame with columns "J", "KT", and "KQ" for propeller open water characteristics

        '''
        pod = sd.pod
        dbh = sd.dbh
        lop = sd.lop
        lcg = sd.lcg
        w = sd.w
        D_p = sd.D_p
        prop_openwater = sd.prop_openwater

        if isinstance(input, pd.DataFrame):
            input_values = input.loc[:, input_columns].iloc[0].to_numpy(dtype=float)
        elif isinstance(input, pd.Series):
            input_values = input.loc[input_columns].to_numpy(dtype=float)
        else:
            input_values = np.asarray(input, dtype=float)

        input_values = np.ravel(input_values)
        if input_values.size < 4:
            raise ValueError(
                f"calc_pod_forces expected at least 4 input values, got {input_values.size}"
            )

        # print(f" input numpy: {input_values}")
        N = input_values[:2] / 60
        # print(f" N: {N}")
        # N = input.loc[:, input_columns[:2]].values[0]  /60
        Thr = [0.0] * len(N)
        # ------
        if len(Thr) == 1:
            #Advance Ratio J
            J = urx * (1- w)/ (N[0]* D_p)
            print(f"NOT implemented yet!")
            # to be coninued
        elif len(Thr) == 2:

            # von claude eingefügt sinnvolles clamping um negative interpolationen zu vermeiden
        
            J_min = prop_openwater["J"].iloc[0]
            J_max = prop_openwater["J"].iloc[-1]
            for i, n in enumerate(N):
                if abs(n) < self.eps:
                    Thr[i] = 0.0
                    continue
                J = (urx * (1 - w)) / (n * D_p)
                J = np.clip(J, J_min, J_max)
                Kt = np.interp(J, prop_openwater["J"], prop_openwater["KT"])
                Kt = max(Kt, 0.0)  # kein negativer Schub durch J-Extrapolation
                # print(f"pod {i}, J: {J}, Kt: {Kt}")

                Thr[i] = Kt * sd.rho * (N[i]**2)*(D_p**4)* np.where(N[i]>= 0, 1.0, -1.0)
                if self.debug:
                    print(f"  pod {i}: N={N[i]*60:.1f} RPM, J={J:.3f} (clip=[{J_min:.2f},{J_max:.2f}]), KT={Kt:.4f}, T={Thr[i]:.1f} N")
        
        if pod == 1:
            # cyvondr, cnvondr  = 0., 0. # only needed in fortran code
            X_pod, Y_pod, N_pod = 0, 0, 0
            for i, T in enumerate(Thr):
                X_pod = X_pod +T * math.cos(input_values[2+i])
                # Y_pod = Y_pod - T * math.sin(input[2+i])
                Y_pod = Y_pod + T * math.sin(input_values[2+i])
                
                if len(Thr) == 2:
                    # moment arm separation according to prop location added by Jelle 
                    dbh2 = {
                        "stb": dbh/2, 
                        "ps": -dbh/2  
                        }.get(lop[i], 0)
                    if lcg == 0: 
                        print("lcg = zero, please check ship_data.yml")
                    
                    h = np.sqrt((lcg)**2 + dbh2**2)*math.sin(input_values[2+i] + math.atan(dbh2/lcg))
                    N_pod -= T * h # positive rudder angle results in negative moment (turning to port)
                # print(f"pod {i}, T: {T}, delta: {input_values[2+i]}, h: {h}, X_pod: {T*math.cos(input_values[2+i])}, Y_pod: {T*math.sin(input_values[2+i])}, N_pod: {T * h}")
        else:
            print("Rudder forces only implemented for podthrusters, not for shaftline propellers")
            X_pod, Y_pod, N_pod = 0, 0, 0
        # print(f"delta_r: {input_values[2:]}, Thr: {Thr}")
        if self.debug:
            print(f"  pod total: X={X_pod:.1f} N, Y={Y_pod:.1f} N, N={N_pod:.1f} Nm, urx={urx:.2f} m/s")
        return [X_pod, Y_pod, N_pod]

    def calc_pod_rud_force(self, input: np.ndarray, input_columns: list, sd: ShipConfig, urx: float):
        """Pod-Kraefte inkl. Ruderkraft-Abschaetzung.

        Wie calc_pod_forces (Schubvektor T*cos/sin(delta)), zusaetzlich wirkt das
        Pod-Gehaeuse + Flosse als Tragfluegel:

            F_N = 0.5 * rho * A_R * u_eff^2 * f_alpha * sin(delta)

        mit f_alpha = 6.13*Lambda/(Lambda+2.25) (Auftriebsgradient nach Soeding,
        Lambda = H_R^2/A_R effektive Streckung) und u_eff der Anstroemung am Pod:
        Nachstrom-reduzierte Fahrt, durch den Propellerstrahl beschleunigt
        (Impulstheorie: u_eff^2 = (u*(1-w))^2 + 8*T/(rho*pi*D_p^2)).

        Gleiche Signatur und Rueckgabe wie calc_pod_forces, damit sie an den
        Aufrufstellen 1:1 austauschbar ist.
        """
        pod = sd.pod
        dbh = sd.dbh
        lop = sd.lop
        lcg = sd.lcg
        w = sd.w
        D_p = sd.D_p
        prop_openwater = sd.prop_openwater

        if isinstance(input, pd.DataFrame):
            input_values = input.loc[:, input_columns].iloc[0].to_numpy(dtype=float)
        elif isinstance(input, pd.Series):
            input_values = input.loc[input_columns].to_numpy(dtype=float)
        else:
            input_values = np.asarray(input, dtype=float)

        input_values = np.ravel(input_values)
        if input_values.size < 4:
            raise ValueError(
                f"calc_pod_rud_force expected at least 4 input values, got {input_values.size}"
            )

        N = input_values[:2] / 60
        Thr = [0.0] * len(N)

        if len(Thr) == 1:
            print(f"NOT implemented yet!")
        elif len(Thr) == 2:
            J_min = prop_openwater["J"].iloc[0]
            J_max = prop_openwater["J"].iloc[-1]
            for i, n in enumerate(N):
                if abs(n) < self.eps:
                    Thr[i] = 0.0
                    continue
                J = (urx * (1 - w)) / (n * D_p)
                J = np.clip(J, J_min, J_max)
                Kt = np.interp(J, prop_openwater["J"], prop_openwater["KT"])
                Kt = max(Kt, 0.0)  # kein negativer Schub durch J-Extrapolation
                Thr[i] = Kt * sd.rho * (N[i]**2)*(D_p**4)* np.where(N[i]>= 0, 1.0, -1.0)
                if self.debug:
                    print(f"  pod {i}: N={N[i]*60:.1f} RPM, J={J:.3f} (clip=[{J_min:.2f},{J_max:.2f}]), KT={Kt:.4f}, T={Thr[i]:.1f} N")

        if pod == 1:
            X_pod, Y_pod, N_pod = 0, 0, 0
            for i, T in enumerate(Thr):
                delta_i = input_values[2+i]

                # Anstroemung am Pod: Nachstrom + Propellerstrahl (Impulstheorie)
                u_p = urx * (1 - w)
                u_eff2 = u_p**2
                if T > 0:
                    u_eff2 += 8.0 * T / (sd.rho * np.pi * D_p**2)

                X_lift, Y_lift = 0.0, 0.0
                if sd.A_R and sd.H_R:
                    asp = sd.H_R**2 / sd.A_R                # effektive Streckung
                    f_alpha = 6.13 * asp / (asp + 2.25)     # Auftriebsgradient nach Soeding
                    F_N = 0.5 * sd.rho * sd.A_R * u_eff2 * f_alpha * math.sin(delta_i)
                    X_lift = -F_N * math.sin(delta_i)       # induzierter Widerstand
                    Y_lift = F_N * math.cos(delta_i)        # gleiche Vorzeichenkonvention wie Schubvektor


                print(f"  pod {i}: T={T:.1f} N, delta={math.degrees(delta_i):.1f} deg, u_eff={np.sqrt(u_eff2):.2f} m/s, Y_thrust={T*math.sin(delta_i):.1f} N, Y_lift={Y_lift:.1f} N")
                
                X_i = T * math.cos(delta_i) + X_lift
                Y_i = T * math.sin(delta_i) + Y_lift
                X_pod += X_i
                Y_pod += Y_i

                if len(Thr) == 2:
                    dbh2 = {
                        "stb": dbh/2,
                        "ps": -dbh/2
                        }.get(lop[i], 0)
                    if lcg == 0:
                        print("lcg = zero, please check ship_data.yml")

                    # N = x*Fy - y*Fx mit Pod bei (x=-lcg, y=dbh2); fuer reinen Schub
                    # identisch zu h = sqrt(lcg^2+dbh2^2)*sin(delta+atan(dbh2/lcg))
                    N_pod -= Y_i * lcg + X_i * dbh2  # positive rudder angle results in negative moment (turning to port)
                if self.debug:
                    print(f"  pod {i}: T={T:.1f} N, delta={math.degrees(delta_i):.1f} deg, u_eff={np.sqrt(u_eff2):.2f} m/s, Y_thrust={T*math.sin(delta_i):.1f} N, Y_lift={Y_lift:.1f} N")
        else:
            print("Rudder forces only implemented for podthrusters, not for shaftline propellers")
            X_pod, Y_pod, N_pod = 0, 0, 0
        
        if self.debug:
            print(f"  pod total: X={X_pod:.1f} N, Y={Y_pod:.1f} N, N={N_pod:.1f} Nm, urx={urx:.2f} m/s")
        return [X_pod, Y_pod, N_pod]

    def calc_windforces(self, uwind, wind_dir, rho_air, Ax, Ay, L, x0_, cxw = 0, cyw = 0, cnw = 0):
        # need to add boundary effekts for wind forces 
               
        # relative windspeeds
        urx = - x0_[3] + uwind* math.cos(x0_[2]-wind_dir)
        ury = - x0_[4] + uwind* math.sin(x0_[2]-wind_dir)

        f_wind_x = cxw * 0.5 * rho_air * urx * abs(urx) * Ax
        f_wind_y = -cyw * 0.5 * rho_air * ury * abs(ury) * Ay
        f_wind_n = cnw * 0.5 * rho_air * urx * abs(urx) * L

        return f_wind_x, f_wind_y, f_wind_n

#--------------------------------------------------
# test written by copilot to check the rudder force calculation in isolation from the rest of the code.

# RESULT: new code shows same results for all tested forces and moments. 
def calc_rudder_forces_direct(sp: dict, Thr, deltas, lop):
    """Standalone test implementation of pod/pod-thruster forces.

    Args:
        sp: ship parameters dict, requires keys `lcg`, `dbh`, and `noh` (optional)
        Thr: iterable of thrust magnitudes
        deltas: iterable of same length with deflection angles (radians)
        lop: list or single value indicating side per propeller ('stb' or 'ps')

    Returns:
        (X_pod, Y_pod, N_pod)
    """
    

    X_pod = 0.0
    Y_pod = 0.0
    N_pod = 0.0

    for i, T in enumerate(Thr):
        delta = deltas[i]
        X_pod += T * math.cos(delta)
        Y_pod -= T * math.sin(delta)

        if sp.get("noh", 1) == 2:
            lop_i = lop[i] if isinstance(lop, (list, tuple)) else lop
            dbh2 = {"stb": sp["dbh"]/2, "ps": -sp["dbh"]/2}.get(lop_i, 0)
            # clearer linear form for h (equivalent to the rotated-vector form):
            h = np.sqrt((sp["lcg"])**2 + dbh2**2)*math.sin(deltas[i] + math.atan(dbh2/sp["lcg"]))
            N_pod += T * h 

    return X_pod, Y_pod, N_pod

def krylov_rudder_forces(sp: dict, Thr, deltas, lop):
    X_pod = 0.0
    Y_pod = 0.0
    N_pod = 0.0

    T_gesamt = sum(Thr)
    delta = deltas[0]
    X_pod += T_gesamt * math.cos(delta)
    Y_pod -= T_gesamt * math.sin(delta)

    if sp.get("noh", 1) == 2:
        dbh2 = sp["dbh"]/2
        # clearer linear form for h (equivalent to the rotated-vector form):
        h = np.sqrt((sp["lcg"])**2 + dbh2**2)*math.sin(deltas[0] - math.atan(dbh2/sp["lcg"]))
        N_pod = T_gesamt * h + (T_gesamt/2)* dbh2*2 * math.cos(delta)

    return X_pod, Y_pod, N_pod

def test_calc_rudder_forces_direct():
    """Simple test harness that passes all inputs directly and prints result."""
    import math

    sp = {"lcg": 10.0, "dbh": 4.0, "noh": 2}
    Thr = [100.0, 100.0]
    deltas = [math.radians(100), math.radians(100)]
    lop = ["ps", "stb"]

    X, Y, N = calc_rudder_forces_direct(sp, Thr, deltas, lop)
    Xk, Yk, Nk = krylov_rudder_forces(sp, Thr, deltas, lop)

    x_err, y_err, n_err = Xk - X, Yk - Y, Nk - N
    # print(f"Direct calc: X={X:.2f}, Y={Y:.2f}, N={N:.2f}")
    # print(f"Krylov calc: X={Xk:.2f}, Y={Yk:.2f}, N={Nk:.2f}")   
    # print(f"Errors: ΔX={x_err:.2f}, ΔY={y_err:.2f}, ΔN={n_err:.2f}")
    # print("test_calc_rudder_forces_direct ->", X, Y, N)
    return 
        
        

        
     


if __name__ == "__main__":
    with open("conf/base/parameters/krylov.yml", "r") as f:
        kry_p = yaml.safe_load(f)

    with open("data/01_raw/wlfa/ship_data.yml", "r") as f:
        ship_parameters = yaml.safe_load(f)

    with open("data/01_raw/wlfa/resistance_curve_HM.csv", "r") as f:
        ship_resistance = pd.read_csv(f)
    
    with open("data/01_raw/wlfa/freif.inp", "r") as f:
        prop_openwater = pd.read_csv(f, sep='\s+', header=None, names=['J', 'KT', 'KQ'])

    with open("data/01_raw/wlfa/test_data.csv", "r") as f:
        input_data = pd.read_csv(f, index_col=0)

    # test_calc_rudder_forces_direct()


    # pd.options.plotting.backend = "plotly"
    # fig = ship_resistance.plot(x="kn", y="kN", kind="line", title="Resistance Curve", labels={"kn": "Speed (knots)", "kN": "Resistance (kN)"})
    # fig = prop_openwater.plot(x="J", y="KT", kind="line", title="Resistance Curve", labels={"J": "J", "KT": "KT"})
    # fig.show()



    # kpc = Krylov_pre_calc(ship_parameters, krylov_parameters)
    # kpc.hydro_mass()


    data = pd.DataFrame({
        "x0": [0.0, 0.0, 0.0, 2.0, 1.0, 0.0],
        "y0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "psi": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "u": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        "v": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "r": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "N0": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "delta_r0": [math.radians(10), math.radians(10), math.radians(10), math.radians(10), math.radians(10), math.radians(10)],
        "N1": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "delta_r1": [math.radians(10), math.radians(10), math.radians(10), math.radians(10), math.radians(10), math.radians(10)],
    })

    kf = Krylov_forces(ship_parameters=ship_parameters, krylov_parameters=kry_p, ship_resistance=ship_resistance, prop_openwater=prop_openwater, data=input_data)
    kf.hydro_mass()
    # kf.sd.Fn = 0.50
    # kf.xtg = -0.03
    x0_ = [0,0,0,3.6,0,0]

    # x, y, n = kf.forces(x0 = x0_, eps = kf.eps, input = input_data[["N0", "N1", "delta_r0", "delta_r1"]], input_columns = ["N0", "N1", "delta_r0", "delta_r1"], sd = kf.sd)
    # a, b, c , d, e, f =kf.equations()
    # print(f"{a}\n{b}\n{c}\n{d}\n{e}\n{f}")
    df = kf.simulate(x0_)

    df.reset_index(inplace=True)
    df_plot = TelemetryPlotter(df, t_unit="s", timecolumn="index", relative_time=True, sensor=False)


    # df_plot.plot_track(start= 0, end = 1e90, lat_col= "y0", lon_col= "x0", figsize= (12, 8), invert_y= True)
    # def __init__(self, df: pd.DataFrame, t_unit: str = "ns", timecolumn: str = "timestamp_ns", relative_time: bool = False, sensor: bool = True)

    ##########################################################
    # als end einfach den höchsten index nehmen!!!,
    ##########################################################


    print(df.index[-1])

    df_plot.plot_azimuth(start= 0, end = df.index[-1], columns= ["psi"], heading_col= "psi", figsize= (12, 8), invert_y= False)



    fig_2 = df.plot(x="y0", y="x0", kind="line", title="trajectory", labels={"x0": "x", "y0": "y"})
    fig_3 = df.plot(x=df.index, y="u", kind="line", title="u", labels={"u": "u"})
    fig_4 = df.plot(x=df.index, y="psi", kind="line", title="psi", labels={"psi": "psi"})
    fig_2.update_yaxes(
            scaleanchor="x",
            scaleratio=1
            )   
    fig_2.update_xaxes(constrain="domain")
    fig_2.update_layout(
        width=600,
        height=600,
        title={
            'text': "Trajectory",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    fig_2.show()
    # fig_3.show()
    # fig_4.show()
    print(df.head(-50))
