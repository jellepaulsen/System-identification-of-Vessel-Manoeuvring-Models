import numpy as np
import pandas as pd
import sympy as sp
import yaml 



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
        if self.LB > 11:
            print("Krylov code only implemented for ships with L/B ratio less than 11")
            return

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
        
        m11 = np.pi() * self.rho* self.T**2 * C *R1_munk * self.L/2.
        m22 = np.pi() * self.rho* self.T**2 * self.L/2. * C *R2_munk* 1/2
        m66 = np.pi() * self.rho* self.T**2 * self.L**2. * C *R3_munk / 24  * self.L
        

        if self.sp["noh"] > 2:
            print("Krylov code only implemented for monohulls and catamarans, not for trimarans or higher")
        
        elif self.sp["noh"] == 2:
            qqq = -1.0 + self.dbh / self.B
            Akxx = 2. + np.exp(-qqq)
            Akyy = 2. -0.8*np.exp(-2.*qqq)
            Corr_x=2.	
            Corr_y = 2.-0.5*np.exp(-2.*qqq)
            Corr_n=2.-0.65*np.exp(-2.*qqq)
            m = self.sp["CB"]*self.rho*self.L*self.sp["B"]*self.T*2
            volume = self.L*self.sp["B"]*self.T*self.sp["CB"]
            self.hydro_mass["m11"] = m11 * Akxx
            self.hydro_mass["m22"] = m22 * Akyy
            self.hydro_mass["m66"] = Akyy*(m66+((self.dbh/2.0)**2)*m22)+Akxx*((self.dbh/2.0)**2)*m11 
            self.hydro_mass["izz"] = 11115 # fixed value for catamaran no idea about dimension 
        else: 
            volume = self.sp["CB"] * self.L * self.sp["B"] * self.T
            m = volume * self.rho
            C = 1.0

            izz = m66 / 1.5
            Corr_x = 1.0
            Corr_y = 1.0
            Corr_n = 1.0

            self.hydro_mass["m11"] = m11 * Corr_x
            self.hydro_mass["m22"] = m22 * Corr_y
            self.hydro_mass["m66"] = m66 * Corr_n
            self.hydro_mass["izz"] = izz


class Krylov_forces:
    def __init__(self, 
                 kpc: Krylov_pre_calc, 
                 data: pd.DataFrame = None, 
                #  x0: np.array = None, 
                 state_columns=["x0", "y0", "psi", "u", "v", "r"]
                 ):
        self.kpc = kpc
        self.data = data
        self.state_columns = state_columns

    @staticmethod
    def coeffs(value, a,b,c):
            return a*value**2 + b*value + c

    def forces(self, x0: None):
        # adding force equations to be simpified and lambdified with sympy
        
        # If None take value from object:
        if x0 is None:
            if hasattr(self, "x0"):
                self.x0_ = self.x0
            else:
                self.x0_ = self.data.iloc[0][self.state_columns].values

        
        # adding wave induced velocities to 
        # need to be added

        
        self.eff_drift_angle(x0) # self.beta_eff

    def krylov_force(self, x0: None, ta: None, tf: None):
        '''
        ta = t aft
        tf = t fore

        '''    
        # not sure if needed, i suppose it needs to be only symbolic (sympy) for the lambdification
        if x0 is None:
            x0 = self.x0_
        # If no initial values for ta and tf, take mean draft as inital value for both
        if ta or tf is None:
            ta, tf = self.kpc.Tm, self.kpc.Tm 
            # später kann es auch aus dem mittleren Tiefgang und dem Trim aus dem IMU berechnet werden zu Fahrtantritt


        Uchar = np.sqrt(x0[3]**2 + x0[4]**2) # speed
        self.Fn = Uchar / np.sqrt(self.kpc.L * 9.81) # Froude number

        self.xtg = self.kpc.sp['x_G']/self.kpc.L
        self.psi1 = (ta - tf) / self.kpc.L # tangent ot static trim angle


        


    def eff_drift_angle(self, x0):
        if x0[4] >= self.eps:
            self.beta_eff = np.arctan(x0[4]/x0[3])
        else:
            self.beta_eff = np.pi()/2 * np.sign(x0[4])

        if x0[5] < 0.0:
            self.beta_eff = np.pi()* np.sign(self.beta_eff)

        self.beta_eff_sign = np.sign(self.beta_eff)

    def calc_psi2(self):
        psi2p = self.kpc.psi2p
        for fnr in psi2p.values():                      # fnr = Fn range
            if fnr["fn"][0] <= self.Fn <= fnr["fn"][1]:
                print(f"fnr: {fnr.keys()}")
                for xgr in fnr["xg"].values():
                    if xgr["r"][0] <= self.kpc.xtg <= xgr["r"][1]:
                        a1 = self.coeffs(self.kpc.xtg, *xgr["a1"])
                        b1 = self.coeffs(self.kpc.xtg, *xgr["b1"])
                        c1 = self.coeffs(self.kpc.xtg, *xgr["c1"])
                        self.psi2 = self.coeffs(self.Fn, a1, b1, c1)


    def calc_sigma(self):
        self.TmL = self.kpc.Tm / self.kpc.L
        askeg = self.kpc.askeg
        psi_res = self.psi1 + self.psi2
        if self.kpc.sp["shiptype"] == 1:
            self.sigma = 1.-(3./(20.-self.kpc.sp["fr_i"]))*(askeg/(self.kpc.L*self.kpc.Tm))+(0.054/(self.TmL)) * psi_res
        elif self.kpc.sp["shiptype"] == 2:
            self.sigma = 0.975 + 0.054/self.TmL * psi_res
        elif self.kpc.sp["shiptype"] == 3:
            self.sigma = 0.962 + 0.054/ self.TmL * psi_res
        
        # lower limit
        if self.sigma <= 0.93:
            self.sigma = 0.93 
        
        # lateral area A_{L sigma}
        self.Asigma = self.kpc.L * self.kpc.Tm * self.sigma 

    def calc_c2(self):
        for tml in self.kpc.c2p.values():
            if tml["tml"][0] <= self.Tml <= tml["tml"][1]:
                a3 = self.coeffs(self.TmL, *tml["a3"])
                b3 = self.coeffs(self.TmL, *tml["b3"])
        
        a1 = 54.46*self.kpc.cp - 59.43
        b1 = -31.44*self.kpc.cp + 46.8

        U = a1 * self.sigma + b1

        for U in self.kpc.c2p["U"].values():
            if U["r"][0] <= self.x0_[3] <= U["r"][1]:
                a2 = self.coeffs(self.x0_[3], *U["a2"])
                b2 = self.coeffs(self.x0_[3], *U["b2"])
        
        Q = a2 * (self.kpc.L/ self.kpc.B) + b2
        self.c2 = np.clip(a3 * Q + b3, 0.3, 1.6)

    def calc_cy_beta(self):
        for lb in self.kpc.cy_betap.values():
            if lb["r"][0] <= self.kpc.LB <= lb["r"][1]:
                for sigma in lb["sigma"].values():
                    if sigma["r"][0] <= self.sigma <= sigma["r"][1]:
                        a1 = self.coeffs(self.kpc.LB, *lb["a1"])
                        b1 = self.coeffs(self.kpc.LB, *lb["b1"])

        a2 = self.coeffs(self.TmL, 16.67, -11.92, 0.06)
        b2 = self.coeffs(self.TmL, 261.1, 213.6, 2.468)

        a3 = self.coeffs(self.kpc.cp, 0.2392, -0.4009, 0.1815)
        b3 = self.coeffs(self.kpc.cp, 0.4033, -0.6965, 0.3263)

        U = a1 * self.LB + b1
        Q = a2 * U + b2

        self.cy_beta = np.clip(a3 * Q + b3, 0.0, 0.5)

    def calc_c3(self):
        a2 = self.coeffs(self.TmL, 2.269, -0.5805, 0.00183)
        b2 = self.coeffs(self.TmL, -27.7, 6.428, -0.01749)

        if self.kpc.cp <= 0.72:
            a1 = self.coeffs(self.kpc.cp, 24.65, -29.67, 7.547)
        elif self.kpc.cp > 0.72:
            a1 = self.coeffs(self.kpc.cp, 0, 5.917, 5.3)

        if self.kpc.cp <= 0.68:
            b1 = self.coeffs(self.kpc.cp, -60.44, 74.61, 9.255)
        elif self.kpc.cp > 0.68:
            b1 = self.coeffs(self.kpc.cp, 0, 10.08, 20.34)

        U = a1 * self.LB + b1
        self.c3 = np.clip(a2 * U + b2, 0.0, 0.35)

    def calc_m1(self):
        a1 = self.coeffs(self.kpc.TmL, -0.1317, 0.05358, 0.000181)
        b1 = self.coeffs(self.kpc.TmL, -2.361, 0.8653, -0.000161)

        if self.kpc.cp <= 0.72:
            U0 = self.coeffs(self.sigma, -235, 474.2, 235.8)
            SCP = self.coeffs(self.kpc.cp, -74.67, 110.9, -39.64)
        elif self.kpc.cp > 0.72:
            U0 = self.coeffs(self.sigma, -210, 422.9, 207.2)
            SCP = self.coeffs(self.kpc.cp, 12, -8.8, -0.64)
        
        UUU = U0 + SCP

        if UUU >= 4:
            Su = -1.3 * UUU + 7.8
            Sv0 = self.coeffs(self.kpc.lb, 0.02333, -0.045, 1.187)
        else:
            Su = -1.3 * UUU + 2.6
            Sv0 = self.coeffs(self.kpc.lb, 0.02333, -0.045, 1.187) + 0.01 * UUU 

        S = Su + Sv0
        self.m1 = np.clip(a1 * S + b1,0.02, 0.08)
    
    def calc_m2(self):
        self.m2 = np.maximum(-(np.log(1.023 * self.sigma))/ (11.6* self.sigma -9.29), -0.01)
         
    def calc_m3(self):
        self.sigma = np.maximum(self.sigma, 1)

        a1 = 31.26 -9.0146 * np.exp(0.066947* self.kpc.LB)
        b1 = 8.6245 * np.exp(0.071419* self.kpc.LB) - 32.26

        a2 = (np.exp(8.20939* self.kpc.cp)* 0.7728*0.001-1.873)*0.001
        b2 = (np.exp(7.47893* self.kpc.cp)*0.4404 * 0.01+5.709)*0.01

        UUUU = (a1 * self.sigma + b1)/ (self.sigma -1.029)

        self.m3 = np.clip(a2 * UUUU + b2, 0.016, 0.054)


if __name__ == "__main__":
    with open("conf/base/parameters/krylov.yml", "r") as f:
        krylov_parameters = yaml.safe_load(f)

    with open("data/01_raw/wlfa/ship_data.yml", "r") as f:
        ship_parameters = yaml.safe_load(f)

    kpc = Krylov_pre_calc(ship_parameters, krylov_parameters)
    kf = Krylov_forces(kpc)

    kf.Fn = 0.50
    kf.kpc.xtg = -0.03
    kf.calc_psi2()
    kf.krylov_force([0,0,0, 2,1,0], ta=0.45, tf=0.40)
    kf.calc_sigma()
    print(f"result: {kf.sigma}")

