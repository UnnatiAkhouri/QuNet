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

from src.gates import hadamard,s_gate

import random

def create_clifford_gate(numclass,index):
    """
    Creates a U1 2 qubit Clifford Unitary
    --------------

    Args:
        numclass: Class of cliford gate you want to generate, {1,2,3,4}
        index: Index of clifford gate you want to generate,
            For Classes 1,4: (1-576)
            For Classes 2,3: (1-5184)

    Returns:
        4x4 2 qubit u1 clifford unitary
    """
    # Defining Relevant Gates For Clifford Gate Generation
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    CNOT1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]], dtype=complex)

    CNOT2 = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]], dtype=complex)

    S = np.array([
        [1, 0],
        [0, 1j]],
        dtype=complex)

    H = np.array([
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [1 / np.sqrt(2), -1 / np.sqrt(2)]],
        dtype=complex)

    W = H @ S

    V = W @ W

    h = [I, H]
    v = [I, V, W]
    p = [I, X, Y, Z]

    if numclass > 4 or numclass < 1: return ValueError

    if numclass == 1:
        if index > 576 or index <1:
            return ValueError
        hvp1_val, p2_rem = divmod(index,4)
        hv_val, p1_rem = divmod(hvp1_val,4)
        hv1_val,v2_rem = divmod(hv_val,3)
        h_val, v1_rem = divmod(hv1_val,3)
        h1_val, h2_rem = divmod(h_val,2)
        h1_rem= h1_val % 2
        return np.kron(h[h1_rem],h[h2_rem]) @ np.kron(v[v1_rem],v[v2_rem]) @ np.kron(p[p1_rem],p[p2_rem])

    if numclass == 4:
        if index > 576 or index <1:
            return ValueError
        hvp1_val, p2_rem = divmod(index, 4)
        hv_val, p1_rem = divmod(hvp1_val, 4)
        hv1_val, v2_rem = divmod(hv_val, 3)
        h_val, v1_rem = divmod(hv1_val, 3)
        h1_val, h2_rem = divmod(h_val, 2)
        h1_rem = h1_val % 2
        return np.kron(h[h1_rem], h[h2_rem]) @ np.kron(v[v1_rem], v[v2_rem]) @ CNOT1 @ CNOT2 @ CNOT1 @ np.kron(p[p1_rem], p[p2_rem])

    if numclass == 2:
        if index > 5184 or index <1:
            return ValueError
        hvp1_val, p2_rem = divmod(index,4)
        hv_val,p1_rem = divmod(hvp1_val,4)
        hv123_val, v4_rem = divmod(hv_val,3)
        hv12_val, v3_rem = divmod(hv123_val,3)
        hv1_val, v2_rem = divmod(hv12_val,3)
        h_val, v1_rem = divmod(hv1_val,3)
        h1_val, h2_rem = divmod(h_val, 2)
        h1_rem = h1_val % 2
        return np.kron(h[h1_rem], h[h2_rem]) @ np.kron(v[v1_rem],v[v2_rem]) @ CNOT1 @ np.kron(v[v3_rem], v[v4_rem]) @ np.kron(p[p1_rem], p[p2_rem])

    if numclass == 3:
        if index > 5184 or index < 1:
            return ValueError
        hvp1_val, p2_rem = divmod(index, 4)
        hv_val, p1_rem = divmod(hvp1_val, 4)
        hv123_val, v4_rem = divmod(hv_val, 3)
        hv12_val, v3_rem = divmod(hv123_val, 3)
        hv1_val, v2_rem = divmod(hv12_val, 3)
        h_val, v1_rem = divmod(hv1_val, 3)
        h1_val, h2_rem = divmod(h_val, 2)
        h1_rem = h1_val % 2
        return np.kron(h[h1_rem], h[h2_rem]) @ np.kron(v[v1_rem], v[v2_rem]) @ CNOT1 @ CNOT2 @ np.kron(v[v3_rem],v[v4_rem]) @ np.kron(p[p1_rem], p[p2_rem])


def create_u1_clifford_gate(numclass,index):
    """
    Creates a U1 2 qubit Clifford Unitary
    --------------

    Args:
        numclass: Class of u1 cliford gate you want to generate, {1,2,3,4}
        index: index of u1 cliford gate you want to generate, (1-16)

    Returns:
        4x4 2 qubit u1 clifford unitary

    """
    # Defining Relevant Gates For Clifford Gate Generation
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    CNOT1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]], dtype=complex)

    CNOT2 = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]], dtype=complex)

    S = np.array([
        [1, 0],
        [0, 1j]],
        dtype=complex)

    H = np.array([
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [1 / np.sqrt(2), -1 / np.sqrt(2)]],
        dtype=complex)

    W = H @ S

    V = W @ W

    ha = [I, S]
    p = [I, Z]
    hb = [V, H]

    if numclass > 4 or numclass < 1: return ValueError

    if numclass == 1:
        if index > 16 or index <1:
            return ValueError
        hap1_val, p2_rem = divmod(index,2)
        ha_val, p1_rem = divmod(hap1_val,2)
        ha1_val, ha2_rem = divmod(ha_val,2)
        ha1_rem = ha1_val % 2
        return np.kron(ha[ha1_rem],ha[ha2_rem]) @ np.kron(p[p1_rem], p[p2_rem])

    if numclass == 4:
        if index > 16 or index <1:
            return ValueError
        hap1_val, p2_rem = divmod(index, 2)
        ha_val, p1_rem = divmod(hap1_val, 2)
        ha1_val, ha2_rem = divmod(ha_val, 2)
        ha1_rem = ha1_val % 2
        return np.kron(ha[ha1_rem],ha[ha2_rem]) @ CNOT1 @ CNOT2 @ CNOT1 @ np.kron(p[p1_rem], p[p2_rem])

    if numclass == 2:
        if index > 16 or index <1:
            return ValueError
        hahbp1_val, p2_rem = divmod(index,2)
        hahb_val, p1_rem = divmod(hahbp1_val,2)
        ha_val, hb_rem = divmod(hahb_val,2)
        ha_rem = ha_val % 2
        return np.kron(ha[ha_rem],hb[hb_rem]) @ CNOT1 @ np.kron(I,W) @ np.kron(p[p1_rem],p[p2_rem])

    if numclass == 3:
        if index > 16 or index < 1:
            return ValueError
        hahbp1_val, p2_rem = divmod(index, 2)
        hahb_val, p1_rem = divmod(hahbp1_val, 2)
        ha_val, hb_rem = divmod(hahb_val, 2)
        ha_rem = ha_val % 2
        return np.kron(hb[ha_rem],ha[hb_rem]) @ CNOT1 @ np.kron(I,W) @ np.kron(p[p1_rem],p[p2_rem])


def create_random_clifford_gate(numclass=0):
    """
    Creates a Random 2 qubit Clifford Gate
    ------------
    Args:
        numclass: Number of Class of clifford gate you want to randomly generate in {1,2,3,4}
         -if blank it is random

    Returns:
        A random 2 qubit Clifford gate
    """
    if numclass == 0:
        classnum = random.randint(1,4)
    elif numclass > 4 or numclass < 0:
        return ValueError
    else:
        classnum = numclass


    if classnum == 1 or classnum == 4:
        index = random.randint(1,576)
    elif classnum == 2 or classnum == 3:
        index = random.randint(1,5184)
    return create_clifford_gate(classnum,index)

def create_random_u1_clifford_gate(numclass=0):
    """
        Creates a Random U1 2 qubit Clifford Gate
        ------------
        Args:
            numclass: Number of Class of clifford gate you want to randomly generate in {1,2,3,4}
             -if blank it is random

        Returns:
            A random U1 2 qubit Clifford gate
        """
    if numclass == 0:
        classnum = random.randint(1,4)
    elif numclass > 4 or numclass < 0:
        return ValueError
    else:
        classnum = numclass

    index = random.randint(1,16)
    return create_u1_clifford_gate(classnum,index)



def list_all_u1_clifford_gates():
    for i in range(1,5):
        for j in range(1,17):
            print("np.array(", np.array2string(create_u1_clifford_gate(i,j), separator=', ', prefix='np.array('), ",dtype=complex),")

def list_all_clifford_gates():
    for i in range(1,577):
        print("np.array(", np.array2string(create_clifford_gate(1,i), separator=', ', prefix='np.array('), ",dtype=complex),")
        print("np.array(", np.array2string(create_clifford_gate(4, i), separator=', ', prefix='np.array('),
              ",dtype=complex),")
    for i in range(1,5185):
        print("np.array(", np.array2string(create_clifford_gate(2, i), separator=', ', prefix='np.array('),
              ",dtype=complex),")
        print("np.array(", np.array2string(create_clifford_gate(3, i), separator=', ', prefix='np.array('),
              ",dtype=complex),")



def grab_u1_clifford_gate(SEED="NONE"):
    if SEED != "NONE":
        rng = np.random.default_rng(SEED)
    gatelist = [
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, -1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, -1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, -1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, -1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]], dtype=complex),
        np.array([[0.70710678 + 0.70710678j, 0. + 0.j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, -0.70710678 - 0.70710678j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   0.70710678 + 0.70710678j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   0. + 0.j, 0.70710678 + 0.70710678j]], dtype=complex),
        np.array([[0.70710678 + 0.70710678j, 0. + 0.j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0.70710678 + 0.70710678j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   -0.70710678 - 0.70710678j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   0. + 0.j, 0.70710678 + 0.70710678j]], dtype=complex),
        np.array([[0.70710678 + 0.70710678j, 0. + 0.j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, -0.70710678 - 0.70710678j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   -0.70710678 - 0.70710678j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   0. + 0.j, -0.70710678 - 0.70710678j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j]], dtype=complex),
        np.array([[0.70710678 + 0.70710678j, 0. + 0.j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0.70710678 + 0.70710678j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   -0.70710678 + 0.70710678j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   0. + 0.j, 0.70710678 - 0.70710678j]], dtype=complex),
        np.array([[0.70710678 + 0.70710678j, 0. + 0.j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, -0.70710678 - 0.70710678j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   -0.70710678 + 0.70710678j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   0. + 0.j, -0.70710678 + 0.70710678j]], dtype=complex),
        np.array([[0.70710678 + 0.70710678j, 0. + 0.j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0.70710678 + 0.70710678j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   0.70710678 - 0.70710678j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   0. + 0.j, -0.70710678 + 0.70710678j]], dtype=complex),
        np.array([[0.70710678 + 0.70710678j, 0. + 0.j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, -0.70710678 - 0.70710678j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   0.70710678 - 0.70710678j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   0. + 0.j, 0.70710678 - 0.70710678j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]], dtype=complex),
        np.array([[0.70710678 + 0.70710678j, 0. + 0.j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0.70710678 + 0.70710678j,
                   0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   0.70710678 + 0.70710678j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j,
                   0. + 0.j, -0.70710678 - 0.70710678j]], dtype=complex),
        np.array([[0.35355339 + 0.35355339j, 0.35355339 - 0.35355339j,
                   0.35355339 + 0.35355339j, -0.35355339 + 0.35355339j],
                  [0.35355339 + 0.35355339j, -0.35355339 + 0.35355339j,
                   0.35355339 + 0.35355339j, 0.35355339 - 0.35355339j],
                  [0.35355339 - 0.35355339j, -0.35355339 - 0.35355339j,
                   -0.35355339 + 0.35355339j, -0.35355339 - 0.35355339j],
                  [0.35355339 - 0.35355339j, 0.35355339 + 0.35355339j,
                   -0.35355339 + 0.35355339j, 0.35355339 + 0.35355339j]], dtype=complex),
        np.array([[0.35355339 + 0.35355339j, -0.35355339 + 0.35355339j,
                   -0.35355339 - 0.35355339j, -0.35355339 + 0.35355339j],
                  [0.35355339 + 0.35355339j, 0.35355339 - 0.35355339j,
                   -0.35355339 - 0.35355339j, 0.35355339 - 0.35355339j],
                  [0.35355339 - 0.35355339j, 0.35355339 + 0.35355339j,
                   0.35355339 - 0.35355339j, -0.35355339 - 0.35355339j],
                  [0.35355339 - 0.35355339j, -0.35355339 - 0.35355339j,
                   0.35355339 - 0.35355339j, 0.35355339 + 0.35355339j]], dtype=complex),
        np.array([[0.35355339 + 0.35355339j, 0.35355339 - 0.35355339j,
                   -0.35355339 - 0.35355339j, 0.35355339 - 0.35355339j],
                  [0.35355339 + 0.35355339j, -0.35355339 + 0.35355339j,
                   -0.35355339 - 0.35355339j, -0.35355339 + 0.35355339j],
                  [0.35355339 - 0.35355339j, -0.35355339 - 0.35355339j,
                   0.35355339 - 0.35355339j, 0.35355339 + 0.35355339j],
                  [0.35355339 - 0.35355339j, 0.35355339 + 0.35355339j,
                   0.35355339 - 0.35355339j, -0.35355339 - 0.35355339j]], dtype=complex),
        np.array([[0.35355339 + 0.35355339j, -0.35355339 + 0.35355339j,
                   0.35355339 + 0.35355339j, 0.35355339 - 0.35355339j],
                  [-0.35355339 + 0.35355339j, 0.35355339 + 0.35355339j,
                   -0.35355339 + 0.35355339j, -0.35355339 - 0.35355339j],
                  [0.35355339 - 0.35355339j, 0.35355339 + 0.35355339j,
                   -0.35355339 + 0.35355339j, 0.35355339 + 0.35355339j],
                  [0.35355339 + 0.35355339j, 0.35355339 - 0.35355339j,
                   -0.35355339 - 0.35355339j, 0.35355339 - 0.35355339j]], dtype=complex),
        np.array([[0.35355339 + 0.35355339j, 0.35355339 - 0.35355339j,
                   0.35355339 + 0.35355339j, -0.35355339 + 0.35355339j],
                  [-0.35355339 + 0.35355339j, -0.35355339 - 0.35355339j,
                   -0.35355339 + 0.35355339j, 0.35355339 + 0.35355339j],
                  [0.35355339 - 0.35355339j, -0.35355339 - 0.35355339j,
                   -0.35355339 + 0.35355339j, -0.35355339 - 0.35355339j],
                  [0.35355339 + 0.35355339j, -0.35355339 + 0.35355339j,
                   -0.35355339 - 0.35355339j, -0.35355339 + 0.35355339j]], dtype=complex),
        np.array([[0.35355339 + 0.35355339j, -0.35355339 + 0.35355339j,
                   -0.35355339 - 0.35355339j, -0.35355339 + 0.35355339j],
                  [-0.35355339 + 0.35355339j, 0.35355339 + 0.35355339j,
                   0.35355339 - 0.35355339j, 0.35355339 + 0.35355339j],
                  [0.35355339 - 0.35355339j, 0.35355339 + 0.35355339j,
                   0.35355339 - 0.35355339j, -0.35355339 - 0.35355339j],
                  [0.35355339 + 0.35355339j, 0.35355339 - 0.35355339j,
                   0.35355339 + 0.35355339j, -0.35355339 + 0.35355339j]], dtype=complex),
        np.array([[0.35355339 + 0.35355339j, 0.35355339 - 0.35355339j,
                   -0.35355339 - 0.35355339j, 0.35355339 - 0.35355339j],
                  [-0.35355339 + 0.35355339j, -0.35355339 - 0.35355339j,
                   0.35355339 - 0.35355339j, -0.35355339 - 0.35355339j],
                  [0.35355339 - 0.35355339j, -0.35355339 - 0.35355339j,
                   0.35355339 - 0.35355339j, 0.35355339 + 0.35355339j],
                  [0.35355339 + 0.35355339j, -0.35355339 + 0.35355339j,
                   0.35355339 + 0.35355339j, 0.35355339 - 0.35355339j]], dtype=complex),
        np.array([[0.5 + 0.j, 0. + 0.5j, 0.5 + 0.j, 0. - 0.5j],
                  [0.5 + 0.j, 0. - 0.5j, 0.5 + 0.j, 0. + 0.5j],
                  [0.5 + 0.j, 0. + 0.5j, -0.5 + 0.j, 0. + 0.5j],
                  [0.5 + 0.j, 0. - 0.5j, -0.5 + 0.j, 0. - 0.5j]], dtype=complex),
        np.array([[0.5 + 0.j, 0. - 0.5j, 0.5 + 0.j, 0. + 0.5j],
                  [0.5 + 0.j, 0. + 0.5j, 0.5 + 0.j, 0. - 0.5j],
                  [0.5 + 0.j, 0. - 0.5j, -0.5 + 0.j, 0. - 0.5j],
                  [0.5 + 0.j, 0. + 0.5j, -0.5 + 0.j, 0. + 0.5j]], dtype=complex),
        np.array([[0.5 + 0.j, 0. + 0.5j, -0.5 + 0.j, 0. + 0.5j],
                  [0.5 + 0.j, 0. - 0.5j, -0.5 + 0.j, 0. - 0.5j],
                  [0.5 + 0.j, 0. + 0.5j, 0.5 + 0.j, 0. - 0.5j],
                  [0.5 + 0.j, 0. - 0.5j, 0.5 + 0.j, 0. + 0.5j]], dtype=complex),
        np.array([[0.5 + 0.j, 0. - 0.5j, -0.5 + 0.j, 0. - 0.5j],
                  [0.5 + 0.j, 0. + 0.5j, -0.5 + 0.j, 0. + 0.5j],
                  [0.5 + 0.j, 0. - 0.5j, 0.5 + 0.j, 0. + 0.5j],
                  [0.5 + 0.j, 0. + 0.5j, 0.5 + 0.j, 0. - 0.5j]], dtype=complex),
        np.array([[0.5 + 0.j, 0. + 0.5j, 0.5 + 0.j, 0. - 0.5j],
                  [0. + 0.5j, 0.5 + 0.j, 0. + 0.5j, -0.5 + 0.j],
                  [0.5 + 0.j, 0. + 0.5j, -0.5 + 0.j, 0. + 0.5j],
                  [0. + 0.5j, 0.5 + 0.j, 0. - 0.5j, 0.5 + 0.j]], dtype=complex),
        np.array([[0.5 + 0.j, 0. - 0.5j, 0.5 + 0.j, 0. + 0.5j],
                  [0. + 0.5j, -0.5 + 0.j, 0. + 0.5j, 0.5 + 0.j],
                  [0.5 + 0.j, 0. - 0.5j, -0.5 + 0.j, 0. - 0.5j],
                  [0. + 0.5j, -0.5 + 0.j, 0. - 0.5j, -0.5 + 0.j]], dtype=complex),
        np.array([[0.5 + 0.j, 0. + 0.5j, -0.5 + 0.j, 0. + 0.5j],
                  [0. + 0.5j, 0.5 + 0.j, 0. - 0.5j, 0.5 + 0.j],
                  [0.5 + 0.j, 0. + 0.5j, 0.5 + 0.j, 0. - 0.5j],
                  [0. + 0.5j, 0.5 + 0.j, 0. + 0.5j, -0.5 + 0.j]], dtype=complex),
        np.array([[0.5 + 0.j, 0. - 0.5j, -0.5 + 0.j, 0. - 0.5j],
                  [0. + 0.5j, -0.5 + 0.j, 0. - 0.5j, -0.5 + 0.j],
                  [0.5 + 0.j, 0. - 0.5j, 0.5 + 0.j, 0. + 0.5j],
                  [0. + 0.5j, -0.5 + 0.j, 0. + 0.5j, 0.5 + 0.j]], dtype=complex),
        np.array([[0.35355339 + 0.35355339j, -0.35355339 + 0.35355339j,
                   0.35355339 + 0.35355339j, 0.35355339 - 0.35355339j],
                  [0.35355339 + 0.35355339j, 0.35355339 - 0.35355339j,
                   0.35355339 + 0.35355339j, -0.35355339 + 0.35355339j],
                  [0.35355339 - 0.35355339j, 0.35355339 + 0.35355339j,
                   -0.35355339 + 0.35355339j, 0.35355339 + 0.35355339j],
                  [0.35355339 - 0.35355339j, -0.35355339 - 0.35355339j,
                   -0.35355339 + 0.35355339j, -0.35355339 - 0.35355339j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, -1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j],
                  [0. + 0.j, -1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, -1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, -1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j]], dtype=complex),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]], dtype=complex)]
    return DM.DensityMatrix(DM.SPARSE_TYPE(rng.choice(gatelist),dtype=np.complex64), energy_basis(2))

