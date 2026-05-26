import numpy as np
import sympy as sp
import yaml 



class Krylov:    
    def __init__(self, ship_parameters: dict, krylov_parameters: dict):
        self.sparams = ship_parameters
        self.kparams = krylov_parameters
        self.rho = self.sparams["rho"]
        self.L = self.sparams["L"]* self.sparams["scale_factor"]
        self.B = self.sparams["B"]* self.sparams["scale_factor"]
        self.T = self.sparams["T"]* self.sparams["scale_factor"]
        self.dbh = self.kparams["dbh"]* self.sparams["scale_factor"] # distance from centerline to bilge keel, scaled with ship scale factor
        self.volume = self.L * self.B * self.T * self.sparams["CB"]
        self.m = self.volume * self.rho    

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

    def kyrylov_force
