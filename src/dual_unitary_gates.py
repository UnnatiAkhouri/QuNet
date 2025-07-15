import src.setup as setup

sp = setup.sp

from math import comb
import src.density_matrix as DM
from src.ket import energy_basis, canonical_basis

import numpy as np

import scipy

from scipy.stats import unitary_group
import scipy.sparse as sp
from scipy import linalg
SPARSE_TYPE = setup.SPARSE_TYPE

from scipy.stats import rv_continuous

def create_SU2_matrix(alpha, beta, theta):
    """
    Generates SU2 matrix
    ----

    Args:
        alpha: real parameter
        beta: complex parameter
        theta: angle

    Returns:
        SU2 matrix

    """
    a = alpha + beta*1j
    b = np.exp(1j*theta)*np.sqrt(1-a*a.conjugate())
    return np.array([
        [a, -1*b.conjugate() ],
        [b, a.conjugate()]], dtype=complex)


def create_dual_unitary_gate(su2gen, theta, J):
    """
    Creates dual unitary gate from SU2 matrix and two angles
    Args:
        su2gen: A list of 4 lists, each containing the unique parameters for the su2 generation
        theta: A phase
        J: a phase in the XXZ hamiltonian

    Returns:
        A 4x4 dual unitary gate
    """
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    preV = np.pi*(np.kron(X,X)+np.kron(Y,Y))+J*np.kron(Z,Z)

    V = scipy.linalg.expm(-1j*preV)

    return np.exp(1j*theta)*(np.kron(create_SU2_matrix(su2gen[0][0],su2gen[0][1],su2gen[0][2]),create_SU2_matrix(su2gen[1][0],su2gen[1][1],su2gen[1][2])))@ V @(np.kron(create_SU2_matrix(su2gen[2][0],su2gen[2][1],su2gen[2][2]),create_SU2_matrix(su2gen[3][0],su2gen[3][1],su2gen[3][2])))


