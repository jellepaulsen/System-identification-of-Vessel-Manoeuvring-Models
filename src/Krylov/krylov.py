import numpy as np
import sympy as sp
import yaml 



class Krylov:    
    def __init__(self, ship_parameters: dict, krylov_parameters: dict):
        self.sparams = ship_parameters
        self.kparams = krylov_parameters
        

    

    def hydro_mass(self):
        # calculation of hydrodynamic mass using the Krylov code
        R1_munk=self.kparams["R1_munk"]      
        R2_munk=self.kparams["R2_munk"]        
        R3_munk=self.kparams["R3_munk"]        
        C=self.kparams["C"]	           #Lewis coefficient


        return