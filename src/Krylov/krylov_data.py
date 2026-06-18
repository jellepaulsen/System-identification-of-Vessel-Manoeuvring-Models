from dataclasses import dataclass, field
from typing import Optional, List
import pandas as pd
import numpy as np

from anyio import value


@dataclass
class ShipConfig:
    # general ship data
    shiptype: Optional[int] = None
    L: Optional[float] = None
    B: Optional[float] = None

    # Type depending
    noh: Optional[int] = None
    twin: Optional[int] = None
    pod: Optional[int] = None
    dbh: Optional[float] = None


    scale_factor: Optional[float] = None
    # static clearwater floating position 
    # should be implemented in the trial data 
    Ta: Optional[float] = None
    Tm: Optional[float] = None
    Tf: Optional[float] = None


    # shape an volumetrics

    S: Optional[float] = None
    lcg: Optional[float] = None

    volume: Optional[float] = None
    m: Optional[float] = None
    
    askeg: Optional[float] = None
    fr_i: Optional[int] = None

    I_z: Optional[float] = None

    CB: Optional[float] = None
    CM: Optional[float] = None



    rho: Optional[float] = None
    A_R: Optional[float] = None

    nop: Optional[int] = None
    lop: Optional[List[str]] = None

    x_p: Optional[float] = None
    x_r: Optional[float] = None
    D_p: Optional[float] = None

    w: Optional[float] = None
    tdf: Optional[float] = None

    w_p0: Optional[float] = None
    C_1: Optional[float] = None
    C_2_beta_p_pos: Optional[float] = None
    C_2_beta_p_neg: Optional[float] = None

    H_R: Optional[float] = None
    C_R: Optional[float] = None

    ship_resistance: Optional[pd.DataFrame] = None
    prop_openwater: Optional[pd.DataFrame] = None


    def __post_init__(self):
        self.cp = self.CB/self.CM
        self.L = self.scale_dim(self.L, 1)
        self.B = self.scale_dim(self.B, 1)
        self.Tm = self.scale_dim(self.Tm, 1)
        self.S = self.scale_dim(self.S, 2)
        self.lcg = self.scale_dim(self.lcg, 1)
        self.volume = self.scale_dim(self.volume, 3)
        self.I_z = self.scale_dim(self.I_z, 4) # need to be checked
        self.askeg = self.scale_dim(self.askeg, 2)

        # if self.noh == 2:
        #     self.LB = self.L/(self.B+self.dbh)
        # else:
        self.LB = self.L / self.B


        self.m = self.volume*self.rho
        self.TmL = self.Tm / self.L 
        self.xtg = self.lcg/self.L # lcg in per from ap

        # if self.LB > 11:
        #      raise ValueError(f"LB must be <= 11 {self.LB} for ship {self.shiptype}")

    def scale_dim(self, value, exp):
        return value * self.scale_factor**exp


    def validate(self) -> None:
        missing = []

        required_fields = [
            "L", "B", "Tm", "m", "rho", "CB"
        ]

        for f in required_fields:
            if getattr(self, f) is None:
                missing.append(f)

        if missing:
            raise ValueError(f"Missing required config fields: {missing}")

        if self.L <= 0:
            raise ValueError("L must be > 0")

        if self.noh is not None and self.noh < 1:
            raise ValueError("noh must be >= 1")
        


