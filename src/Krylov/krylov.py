import numpy as np
import pandas as pd
import sympy as sp
import yaml 



class Krylov_pre_calc:    
    def __init__(self, ship_parameters: dict, krylov_parameters: dict):
        self.sparams = ship_parameters
        self.kparams = krylov_parameters
        self.eps = self.kparams["eps"]
        self.psi2p = self.kparams["psi2"] # psi2 parameters

        self.rho = self.sparams["rho"]
        self.L = self.scale_dim(self.sparams["L"])
        self.B = self.scale_dim(self.sparams["B"])
        self.T = self.scale_dim(self.sparams["T"])
        # self.dbh = self.scale_dim(self.kparams["dbh"]) # currently unknown 
        self.xtg = self.scale_dim(self.sparams['x_G'])/self.L # distance from centerline to bilge keel, scaled with ship scale factor
        self.volume = self.L * self.B * self.T * self.sparams["CB"]
        self.m = self.volume * self.rho    #
        self.xtg = self.sparams['x_G']/self.L
        if self.L/self.B > 11:
            print("Krylov code only implemented for ships with L/B ratio less than 11")
            return

    def scale_dim(self, value):
        return value * self.sparams["scale_factor"]

    def hydro_mass(self):
        # calculation of hydrodynamic mass using the Krylov code
        R1_munk=self.kparams["R1_munk"]      
        R2_munk=self.kparams["R2_munk"]        
        R3_munk=self.kparams["R3_munk"]        
        C=self.kparams["C"]	           #Lewis coefficient

        self.hydro_mass = {}
        
        m11 = np.pi() * self.rho* self.T**2 * C *R1_munk * self.L/2.
        m22 = np.pi() * self.rho* self.T**2 * self.L/2. * C *R2_munk* 1/2
        m66 = np.pi() * self.rho* self.T**2 * self.L**2. * C *R3_munk / 24  * self.L
        

        if self.sparams["noh"] > 2:
            print("Krylov code only implemented for monohulls and catamarans, not for trimarans or higher")
        
        elif self.sparams["noh"] == 2:
            qqq = -1.0 + self.dbh / self.B
            Akxx = 2. + np.exp(-qqq)
            Akyy = 2. -0.8*np.exp(-2.*qqq)
            Corr_x=2.	
            Corr_y = 2.-0.5*np.exp(-2.*qqq)
            Corr_n=2.-0.65*np.exp(-2.*qqq)
            m = self.sparams["CB"]*self.rho*self.L*self.sparams["B"]*self.T*2
            volume = self.L*self.sparams["B"]*self.T*self.sparams["CB"]
            self.hydro_mass["m11"] = m11 * Akxx
            self.hydro_mass["m22"] = m22 * Akyy
            self.hydro_mass["m66"] = Akyy*(m66+((self.dbh/2.0)**2)*m22)+Akxx*((self.dbh/2.0)**2)*m11 
            self.hydro_mass["izz"] = 11115 # fixed value for catamaran no idea about dimension 
        else: 
            volume = self.sparams["CB"] * self.L * self.sparams["B"] * self.T
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
            ta, tf = self.kpc.T, self.kpc.T 
            # später kann es auch aus dem mittleren Tiefgang und dem Trim aus dem IMU berechnet werden zu Fahrtantritt


        Uchar = np.sqrt(self.x0_[3]**2 + self.x0_[4]**2) # speed
        self.Fn = Uchar / np.sqrt(self.kpc.L * 9.81) # Froude number

        xtg = self.kpc.sparams['x_G']/self.kpc.L
        psi1 = (ta- tf) / self.kpc.L # tangent ot static trim angle


        


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
    print(f"result: {kf.psi2}")

