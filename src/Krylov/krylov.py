import numpy as np
import math
import pandas as pd
import sympy as sp
import yaml 

def calc_helper():
    pass

def coeffs(value, a,b,c):
    return a*value**2 + b*value + c





class Krylov_pre_calc:    
    def __init__(self, ship_parameters: dict, krylov_parameters: dict):
        self.sp = ship_parameters
        self.kp = krylov_parameters
        self.eps = self.kp["eps"]
        self.psi2p = self.kp["psi_2"] # psi_2 parameters
        self.c2p = self.kp["c_2"] # c_2 parameters
        self.cy_betap = self.kp["cy_beta"] # cy_beta parameters

        self.cp = self.sp["CB"]/self.sp["CM"] # prismatic coefficient

        self.rho = self.sp["rho"]
        self.L = self.scale_dim1D(self.sp["L"])
        self.B = self.scale_dim1D(self.sp["B"])
        self.Tm = self.scale_dim1D(self.sp["Tm"])
        # self.dbh = self.scale_dim1D(self.kp["dbh"]) # currently unknown 
        self.xtg = self.scale_dim1D(self.sp['x_G'])/self.L # distance from centerline to bilge keel, scaled with ship scale factor
        self.volume = self.L * self.B * self.Tm * self.sp["CB"]
        self.m = self.volume * self.rho    #
        self.xtg = self.sp['x_G']/self.L
        self.askeg = self.scale_dim2D(self.sp["askeg"])
        self.LB = self.L / self.B
        self.TmL = self.Tm / self.L
        if self.LB > 11:
            print("Krylov code only implemented for ships with L/B ratio less than 11")
            return
        self.hydro_mass()

    def scale_dim1D(self, value):
        return value * self.sp["scale_factor"]
    def scale_dim2D(self, value):
        return value * self.sp["scale_factor"]**2
    def scale_dim3D(self, value):
        return value * self.sp["scale_factor"]**3

    def hydro_mass(self):
        # calculation of hydrodynamic mass using the Krylov code
        R1_munk=self.kp["R1_munk"]      
        R2_munk=self.kp["R2_munk"]        
        R3_munk=self.kp["R3_munk"]        
        C=self.kp["C"]	           #Lewis coefficient

        self.hydro_mass = {}
        
        m11 = np.pi * self.rho* self.Tm**2 * C *R1_munk * self.L/2.
        m22 = np.pi * self.rho* self.Tm**2 * self.L/2. * C *R2_munk* 1/2
        m66 = np.pi * self.rho* self.Tm**2 * self.L**2. * C *R3_munk / 24  * self.L
        

        if self.sp["noh"] > 2:
            print("Krylov code only implemented for monohulls and catamarans, not for trimarans or higher")
        
        elif self.sp["noh"] == 2:
            qqq = -1.0 + self.sp["dbh"] / self.B
            Akxx = 2. + np.exp(-qqq)
            Akyy = 2. -0.8*np.exp(-2.*qqq)
            self.Corr_x=2.	
            self.Corr_y = 2.-0.5*np.exp(-2.*qqq)
            self.Corr_n=2.-0.65*np.exp(-2.*qqq)
            m = self.sp["CB"]*self.rho*self.L*self.sp["B"]*self.Tm*2
            volume = self.L*self.sp["B"]*self.Tm*self.sp["CB"]
            self.hydro_mass["m11"] = m11 * Akxx
            self.hydro_mass["m22"] = m22 * Akyy
            self.hydro_mass["m66"] = Akyy*(m66+((self.sp["dbh"]/2.0)**2)*m22)+Akxx*((self.sp["dbh"]/2.0)**2)*m11 
            self.hydro_mass["izz"] = 11115 # fixed value for catamaran no idea about dimension 
        else: 
            volume = self.sp["CB"] * self.L * self.sp["B"] * self.Tm
            m = volume * self.rho
            C = 1.0

            izz = m66 / 1.5
            self.Corr_x = 1.0
            self.Corr_y = 1.0
            self.Corr_n = 1.0

            self.hydro_mass["m11"] = m11 * self.Corr_x
            self.hydro_mass["m22"] = m22 * self.Corr_y
            self.hydro_mass["m66"] = m66 * self.Corr_n
            self.hydro_mass["izz"] = izz


class Krylov_forces(Krylov_pre_calc):
    def __init__(self, 
                 ship_parameters: dict, krylov_parameters: dict,
                 ship_resistance: pd.DataFrame = None,
                 prop_openwater: pd.DataFrame = None,
                 data: pd.DataFrame = None, 
                #  x0: np.array = None, 
                 state_columns: list =["x0", "y0", "psi", "u", "v", "r"],
                 input_colums: list  = ["N0", "N1", "delta_r0", "delta_r1"]
                 ):
        
        super().__init__(ship_parameters, krylov_parameters)
        self.states = data[state_columns]
        self.input = data[input_colums]
        self.input_colums = input_colums
        self.state_columns = state_columns
        self.ship_resistance = ship_resistance
        self.prop_openwater = prop_openwater

    def forces(self,
                x0 = None,
                input = None,
                input_colums = None
                ):
        # adding force equations to be simpified and lambdified with sympy
        # If None take value from object:
        if x0 is None:
            x0 = self.data.iloc[0][self.state_columns].values()

        
        # adding wave induced velocities to 
        # need to be added

        
        self.eff_drift_angle(x0) # self.beta_eff
        self.wave_induced_velocities(x0)
        self.krylov_force(x0)
        self.calc_propeller_forces([input.iloc[0][input_colums][:1].values()], x0[3])
        self.calc_pod_forces()


    
    #------------------------------------------------------
    # wave induced velocities
    def wave_induced_velocities(self, x0):
        # need to be added
        ucx = 0
        ucy = 0
        x0[3] = x0[3] + ucx 
        x0[4] = x0[4] + ucy 
        return x0

    #------------------------------------------------------
    # Krylov Force

    def krylov_force(self, x0= None, ta= None, tf= None, L= None, Tm= None):
        '''
        ta = t aft
        tf = t fore

        '''    
        # not sure if needed, i suppose it needs to be only symbolic (sympy) for the lambdification
        if x0 is None:
            x0 = self.x0
        # If no initial values for ta and tf, take mean draft as inital value for both
        if ta is None or tf is None:
            ta, tf = Tm, Tm 
            # später kann es auch aus dem mittleren Tiefgang und dem Trim aus dem IMU berechnet werden zu Fahrtantritt


        Uchar = np.sqrt(x0[3]**2 + x0[4]**2) # speed
        Fn = Uchar / np.sqrt(L * 9.81) # Froude number

        self.xtg = self.sp['x_G']/L
        self.psi1 = (ta - tf) /L # tangent ot static trim angle



        cx0 = self.get_cx0()




        beta_eff, beta_eff_sign = self.eff_drift_angle(x0) # self.beta_eff
        self.calc_psi2()
        self.calc_sigma()
        self.calc_c2()
        self.calc_c3()
        self.calc_cy_beta()
        self.calc_ms()
        self.calc_cn_beta()
        self.calc_cn(x0)

        Umnos_fo = self.rho*self.Asigma*self.L/2
        Umnos_cn = self.rho*self.Asigma*(self.Uchar**2)/2


        # correction factor are different to the given krylov code. Original code is is deplayed afterwards
        return (
            self.cxb * Umnos_fo     * self.Corr_x,      # corr_n
            self.cy_beta * Umnos_fo * self.Corr_y,      # corr_y
            self.cn * Umnos_cn * self.Corr_n            # corr_X

        )
        
    def get_cx0(self, ship_resistance = None, Uchar = None):
        # interpolate zerodrift resistance from resistance curve
        
        RTx0 = np.interp(Uchar, ship_resistance["kn"]*0.5144 , ship_resistance["kN"])

        # NOTE: For catamaran the wetted area of demi hull is used!!! Whereas RT is for the whole ship!!
        return RTx0 / (0.5 * self.rho * Uchar**2 * self.sp["S"]) 

    def eff_drift_angle(self, x0):
        if x0[4] >= self.eps:
            beta_eff = np.arctan(x0[4]/x0[3])
        else:
            beta_eff = np.pi/2 * np.where(x0[4]>0, 1, -1)

        if x0[5] < 0.0:
            beta_eff = np.pi* np.where(beta_eff>0, 1, -1)

            beta_eff_sign = np.where(beta_eff>0, 1, -1)
        return beta_eff, beta_eff_sign



    def calc_psi2(self, psi2p):
        check = 0
        for fnr in psi2p.values():                      # fnr = Fn range
            if fnr["fn"][0] <= self.Fn <= fnr["fn"][1]:
                print(f"fnr: {fnr.keys()}")
                for xgr in fnr["xg"].values():
                    if xgr["r"][0] <= self.xtg <= xgr["r"][1]:
                        a1 = coeffs(self.xtg, *xgr["a1"])
                        b1 = coeffs(self.xtg, *xgr["b1"])
                        c1 = coeffs(self.xtg, *xgr["c1"])
                        check += 1
        if check == 1:
            psi2 = coeffs(self.Fn, a1, b1, c1)
            return psi2
        else:
            print(f"Fn: {self.Fn}, xtg: {self.xtg}, check: {check}")
            print("Fn or xg out of range for psi2 calculation")
            print("If check > 1 then multiple ranges are overlapping CODE INCORRECT")
            return 0.0


    def calc_sigma(self,psi1, psi2, Tm, L, askeg):
        TmL = Tm/L
        psi_res = psi1 + psi2
        if self.sp["shiptype"] == 1:
            sigma = 1.-(3./(20.-self.sp["fr_i"]))*(askeg/(L*Tm))+(0.054/(TmL)) * psi_res
        elif self.sp["shiptype"] == 2:
            sigma = 0.975 + 0.054/self.TmL * psi_res
        elif self.sp["shiptype"] == 3:
            sigma = 0.962 + 0.054/ self.TmL * psi_res
        
        # lower limit
        if sigma <= 0.93:
            sigma = 0.93 
        
        # lateral area A_{L sigma}
        Asigma = self.L * self.Tm * self.sigma 
        return sigma, Asigma

    def calc_c2(self, L, B, TmL, c2p, x0):    
        for tml in c2p.values():
            if "tml" in tml.keys():
                if tml["tml"][0] <= TmL <= tml["tml"][1]:
                    a3 = coeffs(TmL, *tml["a3"])
                    b3 = coeffs(TmL, *tml["b3"])
        
        a1 = 54.46*self.cp - 59.43
        b1 = -31.44*self.cp + 46.8

        U = a1 * self.sigma + b1

        for U in c2p["U"].values():
            if U["r"][0] <= x0[3] <= U["r"][1]:
                a2 = coeffs(x0[3], *U["a2"])
                b2 = coeffs(x0[3], *U["b2"])
        
        Q = a2 * (L/ B) + b2
        c2 = np.clip(a3 * Q + b3, 0.3, 1.6)

        return c2

    def calc_c3(self, LB, TmL, cp, coeffs):
        a2 = coeffs(TmL, 2.269, -0.5805, 0.00183)
        b2 = coeffs(TmL, -27.7, 6.428, -0.01749)

        if cp <= 0.72:
            a1 = coeffs(cp, 24.65, -29.67, 7.547)
        elif cp > 0.72:
            a1 = coeffs(cp, 0, 5.917, 5.3)

        if cp <= 0.68:
            b1 = coeffs(cp, -60.44, 74.61, 9.255)
        elif cp > 0.68:
            b1 = coeffs(cp, 0, 10.08, 20.34)

        U = a1 * LB + b1
        c3 = np.clip(a2 * U + b2, 0.0, 0.35)

        return c3

    def calc_cy_beta(self, LB, TmL, cp, beta_eff, beta_eff_sign):
        for lb in self.cy_betap.values():
            if lb["r"][0] <= LB <= lb["r"][1]:
                for sigma in lb["sigma"].values():
                    if sigma["r"][0] is None and sigma["r"][1] is None:
                        print()
                        a1 = coeffs(LB, *sigma["a1"])
                        b1 = coeffs(LB, *sigma["b1"])
                    elif sigma["r"][0] <= self.sigma <= sigma["r"][1]:
                        a1 = coeffs(LB, *sigma["a1"])
                        b1 = coeffs(LB, *sigma["b1"])

        a2 = coeffs(TmL, 16.67, -11.92, 0.06)
        b2 = coeffs(TmL, 261.1, 213.6, 2.468)

        a3 = coeffs(cp, 0.2392, -0.4009, 0.1815)
        b3 = coeffs(cp, 0.4033, -0.6965, 0.3263)

        U = a1 * LB + b1
        Q = a2 * U + b2

        cy_beta_2 = np.clip(a3 * Q + b3, 0.0, 0.5)
        cy_beta = 0.5* cy_beta_2 * np.sin(2.* beta_eff)* np.cos(beta_eff) + (self.c2*(np.sin(beta_eff)**2))+ self.c3*(np.sin(2*beta_eff)**4)* beta_eff_sign
        return cy_beta


    def calc_ms(self, TmL, sigma, LB, cp):
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
        m1 = np.clip(a1 * S + b1,0.02, 0.08)

        # ----m2------
        m2 = np.maximum(-(np.log(1.023 * sigma))/ (11.6* sigma -9.29), -0.01)

        # ----m3------ 
        self.sigma = np.maximum(self.sigma, 1)

        a1 = 31.26 -9.0146 * np.exp(0.066947* LB)
        b1 = 8.6245 * np.exp(0.071419* LB) - 32.26

        a2 = (np.exp(8.20939* cp)* 0.7728*0.001-1.873)*0.001
        b2 = (np.exp(7.47893* cp)*0.4404 * 0.01+5.709)*0.01

        UUUU = (a1 * self.sigma + b1)/ (self.sigma -1.029)

        m3 = np.clip(a2 * UUUU + b2, 0.016, 0.054)
    
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
        
        if sigma <= 0.96:
            Ss = coeffs(sigma, 1900, -3696, 1796)
        elif sigma > 0.96:
            Ss = coeffs(sigma, 391.7, -810.4, 415.8)
        
        UUUUU = U0 + Ss
        Su = 0.00827 * UUUUU - 0.017

        m4 = np.clip(Sm4 + Su, 0.03, 0.04)

        return [m1, m2, m3, m4]

    def calc_cn_beta(self, cx0, beta_eff, m):
        beta = beta_eff
        cn_beta = m[0]*np.sin(2*beta)+m[1]*np.sin(beta)+m[2]*(np.sin(2*beta)**3) + m[3] * (np.sin(2*beta)**5)

        cxb = -self.kp["a1x"] * np.sin((np.pi-np.arcsin(cx0/self.kp["a1x"]))*(1-(abs(beta)*180/np.pi/self.kp["psix"])))
        return cn_beta, cxb
    
    def calc_cn(self, x0, c2, cn_beta, beta_eff, Tml, sigma, LB):
        cn0 = 0.059*c2
        cnw2= (0.739 +8.7 * Tml)*(1.611*(sigma**2)-2.873*sigma+1.33)

        a1 = 0.09-cnw2 - 0.0033*(LB -7)-20*((Tml-0.005)**2)+ 0.4*(sigma-0.9)+ 0.05*(self.sp["CM"]-0.9)
        a2 = 0.008*LB + 0.9 *(Tml -0.05) + 0.45*(sigma-0.955)

        cnw = cnw2 + a1 *abs(np.sin(beta_eff))+ a2 * (1-np.cos( (2*np.pi-4*abs(beta_eff))*np.cos(beta_eff)+0.1*abs(np.sin(2*beta_eff))))

        # omega is rate of turn

        if self.Uchar > self.eps:
            omega_strich = x0[5] * self.L/ self.Uchar
            omega_large = omega_strich /np.sqrt(1+omega_strich**2)
        else:
            omega_large = 1.0 

        cnom = -cn0*abs(x0[5])* x0[5]*self.L**2 - cnw/ np.pi*(self.Uchar**2 +(x0[5]**2)*self.L**2)* np.sin(np.pi*omega_large)

        self.cn = cnom +cn_beta*self.Uchar**2 # called cn_full

    # End Krylov Force
    #------------------------------------------------------

    # Propeller Forces
    def calc_propeller_Thrust(self, N: list,  Thr: list):
        pass
        # in init



    def calc_pod_forces(self, input: pd.DataFrame, input_colums,  Thr: list, dbh, lop, lcg, urx, w, D_p, prop_openwater: pd.DataFrame):
        # changed from original kryov code: each pod gets its own delta_r and therefore the forces and moments differ 
        # NOTE!!: pod thrust is first assmued to be in the rotating center of the pod. Later the effects of another lever arm can be added 
        # the pod forces are acting through the pod rotation center on the ship therefore no leverarm is assumed for either azipull or push thrusters
        # this code only works with even numbers of pods which are symetrically arranged at the back of the ship
        
        N = [input.iloc[0][input_colums[:1]].values()]

        T = [len(self.sp["nop"])]
        # ------
        if self.sp["noh"] == 1:
            #Advance Ratio J
            J = urx * (1- w)/ (N[0]* D_p)
            print(f"NOT implemented yet!")
            # to be coninued
        elif self.sp["noh"] == 2:
            for i, n in enumerate(N):
                J = urx * (1- w)/ (n* D_p)
                Kt = np.interp(J, prop_openwater["J"], prop_openwater["KT"])
                T[i] = Kt * self.rho * (N[i]**2)*(D_p**4)* np.where(N[i]>= 0, 1.0, -1.0)
        





        if self.sp["Pod"] == 1:
            # cyvondr, cnvondr  = 0., 0. # only needed in fortran code
            X_pod, Y_pod, N_pod = 0, 0, 0
            for i, T in enumerate(Thr):
                X_pod = X_pod +T * math.cos(input[f"delta_r{i}"])
                Y_pod = Y_pod - T * math.sin(input[f"delta_r{i}"])
                if self.sp["noh"] == 2:
                    # moment arm separation according to prop location added by Jelle 
                    dbh2 = {
                        "stb": dbh/2, 
                        "ps": -dbh/2
                        }.get(lop[i], 0)
                    
                    h = np.sqrt((lcg)**2 + dbh2**2)*math.sin(input[f"delta_r{i}"] + math.atan(dbh2/lcg))
                    N_pod = N_pod + T * h
            else:
                print("Rudder forces only implemented for podthrusters, not for shaftline propellers")
                X_pod, Y_pod, N_pod = 0, 0, 0

            return [X_pod, Y_pod, N_pod]

    def calc_windforces(self):
        # to be implemented
        pass
        

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
    print(f"Direct calc: X={X:.2f}, Y={Y:.2f}, N={N:.2f}")
    print(f"Krylov calc: X={Xk:.2f}, Y={Yk:.2f}, N={Nk:.2f}")   
    print(f"Errors: ΔX={x_err:.2f}, ΔY={y_err:.2f}, ΔN={n_err:.2f}")
    # print("test_calc_rudder_forces_direct ->", X, Y, N)
    return 
        
        

        
     


if __name__ == "__main__":
    with open("conf/base/parameters/krylov.yml", "r") as f:
        krylov_parameters = yaml.safe_load(f)

    with open("data/01_raw/wlfa/ship_data.yml", "r") as f:
        ship_parameters = yaml.safe_load(f)

    with open("data/01_raw/wlfa/resistance_curve_HM.csv", "r") as f:
        ship_resistance = pd.read_csv(f)
    
    with open("data/01_raw/wlfa/freif.inp", "r") as f:
        prop_openwater = pd.read_csv(f, sep='\s+', header=None, names=['J', 'KT', 'KQ'])


    test_calc_rudder_forces_direct()


    # kpc = Krylov_pre_calc(ship_parameters, krylov_parameters)
    # kf = Krylov_forces(ship_parameters, krylov_parameters, ship_resistance=ship_resistance, prop_openwater = prop_openwater)

    # kf.Fn = 0.50
    # kf.xtg = -0.03
    # kf.forces(x0 = [0,0,0,2,1,0])
    # kf.krylov_force(ta=0.45, tf=0.40)
    # kf.calc_sigma()
    # print(f"result: {kf.sigma}")


