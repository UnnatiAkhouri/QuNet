import src.setup as setup

sp = setup.sp

from math import comb
import src.density_matrix as DM
from src.ket import energy_basis, canonical_basis

import numpy as np

from scipy.stats import unitary_group
import scipy.sparse as sp
SPARSE_TYPE = setup.SPARSE_TYPE

from scipy.stats import rv_continuous

class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)


# Samples of theta should be drawn from between 0 and pi
sin_sampler = sin_prob_dist(a=0, b=np.pi)

def hadamard():
    return DM.DensityMatrix(np.array([[1, 1], [1, -1]]) / np.sqrt(2),canonical_basis(1))

def pauli_x():
    return DM.DensityMatrix(np.array([[0, 1], [1, 0]]),canonical_basis(1))

def pauli_y():
    return DM.DensityMatrix(np.array([[0, -1j], [1j, 0]]),canonical_basis(1))

def pauli_z():
    return DM.DensityMatrix(np.array([[1, 0], [0, -1]]),canonical_basis(1))

def s_gate():
    return DM.DensityMatrix(np.array([[1, 0],[0, 1j]]),canonical_basis(1))

def t_gate():
    return DM.DensityMatrix(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),canonical_basis(1))

def r_x_rotation(p):
    return DM.DensityMatrix(np.array([[np.cos(p / 2), -1j * np.sin(p / 2)], [-1j * np.sin(p / 2), np.cos(p / 2)]]),canonical_basis(1))

def r_y_rotation(p):
    return DM.DensityMatrix(np.array([[np.cos(p/2), -np.sin(p/2)],[np.sin(p/2), np.cos(p/2)]]),canonical_basis(1))

def r_z_rotation(p):
    return DM.DensityMatrix(np.array([[np.exp(-1j * p / 2), 0],[0, np.exp(1j * p / 2)]]),canonical_basis(1))

