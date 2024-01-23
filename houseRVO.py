import numpy as np
from numpy.linalg import inv
from dataclasses import dataclass

'''
The full implementation of this class contains more methods which are relevant 
for generating the baseline profiles. These will be added later. Now this class functions as a data class
'''


@dataclass
class House:
    def __init__(self, capacities: dict, resistances: dict, window_area: float):
        # Create capacity matrix and its inverse
        self.C = np.diag(np.array([capacities['C_in'], capacities['C_out']]))
        self.C_inv = inv(self.C)

        # Create heat conductance matrices and their inverse
        k_exch = 1.0 / resistances['R_exch']
        k_floor = 1.0 / resistances['R_floor']
        k_vent = 1.0 / resistances['R_vent']
        k_cond = 1.0 / resistances['R_cond']

        # Estimate total conductance with parallel circuit of floor, transm, and ventilation
        k_transm = 1.0/(1.0/k_exch + 1.0/k_cond)  # in series
        self.k_total = k_floor + k_transm + k_vent

        self.K = np.array([[k_vent + k_exch + k_floor, -k_exch], [-k_exch, k_cond + k_exch]])
        self.K_amb = np.array([[k_vent, k_floor], [k_cond, 0]])

        # Note that both K and K_amb are diagonally dominant and thus invertible
        self.K_inv = inv(self.K)
        self.K_amb_inv = inv(self.K_amb)

        # precomputed matrices
        self.A = np.matmul(self.C_inv, self.K)
        self.A_amb = np.matmul(self.C_inv, self.K_amb)
        self.A_inv = inv(self.A)

        self.exponential_matrix = None

        self.window_area = window_area
        self.shgc = 0.7  # solar heat gain coefficient