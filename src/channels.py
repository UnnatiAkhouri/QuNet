import src.setup as setup

sp = setup.sp

from math import comb
import src.density_matrix as DM
from src.ket import energy_basis, canonical_basis

import numpy as np

from scipy.stats import unitary_group
import scipy.sparse as sp
SPARSE_TYPE = setup.SPARSE_TYPE

from itertools import product
from scipy.stats import rv_continuous

class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


# Samples of theta should be drawn from between 0 and pi
sin_sampler = sin_prob_dist(a=0, b=np.pi)

def check_kraus_completeness(self, kraus_ops: list, tol: float = 1e-10) -> bool:
    """
    Check if a set of Kraus operators is complete (trace-preserving).
    Returns True if sum K_i^\dagger K_i == I (within tolerance).
    """
    d = kraus_ops[0].shape[0]
    completeness = sum([K.conj().T @ K for K in kraus_ops])
    return np.allclose(completeness, np.eye(d), atol=tol)

# def independent_weak_measurement_kraus(alpha1, alpha2, observable1='Z', observable2='Z'):
#     """
#     Create Kraus operators for independent weak measurements on two qubits.
#
#     Parameters:
#     alpha1, alpha2: measurement strengths (0 = no measurement, 1 = projective)
#     observable1, observable2: which Pauli observable to measure ('X', 'Y', 'Z')
#
#     Returns:
#     List of 4x4 Kraus operators
#     """
#     obs1 = PAULI[observable1]
#     obs2 = PAULI[observable2]
#
#     def single_qubit_kraus(alpha, observable):
#         """Single qubit weak measurement Kraus operators"""
#         eigvals, eigvecs = np.linalg.eigh(observable)
#         P0 = np.outer(eigvecs[:, 0], eigvecs[:, 0].conj())  # Projector onto first eigenspace
#         P1 = np.outer(eigvecs[:, 1], eigvecs[:, 1].conj())  # Projector onto second eigenspace
#
#         M0 = np.sqrt(1 - alpha) * I + np.sqrt(alpha) * P0
#         M1 = np.sqrt(alpha) * P1
#
#         return [M0, M1]
#
#     kraus1 = single_qubit_kraus(alpha1, obs1)
#     kraus2 = single_qubit_kraus(alpha2, obs2)
#
#     # Tensor product for all combinations
#     kraus_operators = []
#     for M1, M2 in product(kraus1, kraus2):
#         kraus_operators.append(np.kron(M1, M2))
#
#     return kraus_operators

# def correlated_weak_measurement_kraus(alpha, correlation_type='ZZ'):
#     """
#     Create Kraus operators for correlated two-qubit weak measurement.
#
#     Parameters:
#     alpha: measurement strength (0 to 1)
#     correlation_type: type of correlation ('ZZ', 'XX', 'YY', 'XZ', etc.)
#
#     Returns:
#     List of 4x4 Kraus operators
#     """
#     obs1 = PAULI[correlation_type[0]]
#     obs2 = PAULI[correlation_type[1]]
#     two_qubit_obs = np.kron(obs1, obs2)
#
#     # Diagonalize the two-qubit observable
#     eigvals, eigvecs = np.linalg.eigh(two_qubit_obs)
#     unique_eigvals = np.unique(np.round(eigvals, 10))
#
#     kraus_operators = []
#     for eigval in unique_eigvals:
#         # Find eigenvectors with this eigenvalue
#         mask = np.abs(eigvals - eigval) < 1e-10
#         indices = np.where(mask)[0]
#
#         # Construct projector onto eigenspace
#         P = np.zeros((4, 4), dtype=complex)
#         for idx in indices:
#             v = eigvecs[:, idx]
#             P += np.outer(v, v.conj())
#
#         # Weak measurement Kraus operator
#         if np.abs(eigval - 1) < 1e-10:  # +1 eigenvalue
#             M = np.sqrt((1 + alpha) / 2) * np.eye(4) + np.sqrt(alpha * (1 - alpha) / 2) * P
#         elif np.abs(eigval + 1) < 1e-10:  # -1 eigenvalue
#             M = np.sqrt((1 + alpha) / 2) * np.eye(4) - np.sqrt(alpha * (1 - alpha) / 2) * P
#         else:
#             # General case
#             M = np.sqrt(1 - alpha) * np.eye(4) + np.sqrt(alpha) * P
#
#         kraus_operators.append(M)
#
#     return kraus_operators


def phase_covariant_channel_affine(l1: int, l3:int, tau3:int, seed=None):
    """
    Args:
        l1: isotropic stretching in x and y direction
        l3: anisotropic stretching in z direction
        tau3: translation in z direction
        seed (optional): the seed for the random number generator.
    Returns: a phase covariant channel with the given map parameters

    """
    mat = sp.lil_matrix((4, 4), dtype=np.complex64)
    mat[0, 0] = 1
    mat[1, 1] = l1
    mat[2, 2] = l1
    mat[3, 0] = tau3
    mat[3, 3] = l3
    mat= mat.tocsr()
    return DM.DensityMatrix(DM.SPARSE_TYPE(mat, dtype=np.complex64), canonical_basis(2))


def phase_covariant_kraus_operators(l3, t3, l1):
    """
    Create the 4 Kraus operators from your LaTeX specification

    Parameters:
    -----------
    delta_t : float
        δ(t) parameter
    kappa_t : float
        κ(t) parameter
    gamma_t : float
        Γ(t) parameter
    phi_t : float
        φ(t) parameter

    Returns:
    --------
    K1, K2, K3, K4 : numpy arrays
        The four 2x2 Kraus operators
    """

    # Compute lambda_plus and lambda_minus
    sqrt_term = np.sqrt(t3 ** 2 + 4 * l1*l1)
    lambda_plus = (1 + l3 + sqrt_term) / 2
    lambda_minus = (1 + l3 - sqrt_term) / 2

    # Compute eta from cot[eta] definition
    cot_eta = (t3 + sqrt_term) / (2 * l1)
    #eta = np.arccot(cot_eta)  # eta = arccot(cot_eta)

    # Alternatively, since cot(eta) = cos(eta)/sin(eta), we can compute directly:
    # tan(eta) = 1/cot(eta)
    tan_eta = 1 / cot_eta
    cos_eta = 1 / np.sqrt(1 + tan_eta ** 2)
    sin_eta = tan_eta / np.sqrt(1 + tan_eta ** 2)

    # K1 coefficient
    coeff_K1 = np.sqrt((1 - l3*t3) / 2)
    K1 = coeff_K1 * np.array([[0, 1],
                              [0, 0]], dtype=complex)

    # K2 coefficient
    coeff_K2 = np.sqrt((1 - (l3/t3)) / 2)
    K2 = coeff_K2 * np.array([[0, 0],
                              [1, 0]], dtype=complex)

    # K3
    coeff_K3 = np.sqrt(lambda_plus)
    K3 = coeff_K3 * np.array([[cos_eta, 0],
                              [0, sin_eta]], dtype=complex)

    # K4
    coeff_K4 = np.sqrt(lambda_minus)
    K4 = coeff_K4 * np.array([[-sin_eta, 0],
                              [0, cos_eta]], dtype=complex)

    return K1, K2, K3, K4
    #return DM.DensityMatrix(K1,canonical_basis(1)), DM.DensityMatrix(K2,canonical_basis(1)), DM.DensityMatrix(K3,canonical_basis(1)), DM.DensityMatrix(K4,canonical_basis(1))


# Example usage:
def example_usage():
    # Example parameters
    l1=0.1
    l3=0.01
    t3=0.7

    K1, K2, K3, K4 = phase_covariant_kraus_operators(l3,t3,l1)

    print("K1 =")
    print(K1)
    print("\nK2 =")
    print(K2)
    print("\nK3 =")
    print(K3)
    print("\nK4 =")
    print(K4)

    # Verify completeness relation: ∑ᵢ Kᵢ† Kᵢ ≤ I
    #completeness = K1.data.toarray().conj().T @ K1.data.toarray()+ K2.data.toarray().conj().T @ K2.data.toarray() + K3.data.toarray().conj().T @ K3.data.toarray() + K4.data.toarray().conj().T @ K4.data.toarray()
    completeness=K1.conj().T@ K1 + K2.conj().T @ K2 + K3.conj().T @ K3 + K4.conj().T @ K4
    print("\nCompleteness check (should be ≤ I):")
    print(completeness)
    print("\nTrace of completeness:", np.trace(completeness))

    return K1, K2, K3, K4


def create_uncorrelated_2qubit_kraus(single_qubit_kraus):
    """
    Create L⊗L channel from single-qubit Kraus operators

    Parameters:
    -----------
    single_qubit_kraus : list of 2x2 arrays
        [K1, K2, K3, K4] - your single-qubit Kraus operators

    Returns:
    --------
    uncorr_kraus : list of 4x4 arrays
        All combinations Kᵢ⊗Kⱼ
    """
    uncorr_kraus = []

    for Ki in single_qubit_kraus:
        for Kj in single_qubit_kraus:
            # Each combination gives one 2-qubit Kraus operator
            Kij = np.kron(Ki, Kj)
            uncorr_kraus.append(Kij)

    return uncorr_kraus


def example_uncorrelated_kraus():
    l1=0.1
    l3=0.01
    t3=0.7
    K1, K2, K3, K4 = phase_covariant_kraus_operators(l3,t3,l1)  # Your 4 operators
    single_kraus = [K1,K2,K3,K4]
    uncorr_kraus = create_uncorrelated_2qubit_kraus(single_kraus)
    for k in uncorr_kraus:
        print(k)
    print("single kraus operators:")
    print(single_kraus)
    print(f"Uncorrelated channel has {len(uncorr_kraus)} Kraus operators (should be 16)")

def create_perfectly_correlated_2qubit_kraus(single_qubit_kraus):
    """
    Create perfectly correlated 2-qubit channel
    Both qubits experience the same random Kraus operator

    Λ_corr(ρ) = Σᵢ (Kᵢ ⊗ Kᵢ) ρ (Kᵢ ⊗ Kᵢ)†
    """
    corr_kraus = []

    for Ki in single_qubit_kraus:
        # Same index for both qubits
        Kii = np.kron(Ki, Ki)
        corr_kraus.append(Kii)

    return corr_kraus

def example_perfectly_correlated_kraus():
    l1=0.1
    l3=0.01
    t3=0.7
    K1, K2, K3, K4 = phase_covariant_kraus_operators(l3,t3,l1)  # Your 4 operators
    single_kraus = [K1,K2,K3,K4]
    # Usage
    corr_kraus = create_perfectly_correlated_2qubit_kraus(single_kraus)
    for k in corr_kraus:
        print(k)
    print(f"Correlated channel has {len(corr_kraus)} Kraus operators (should be 4)")

#New
def independent_depolarizing(params: list[float]) -> list[np.ndarray]:
    """Independent depolarizing noise
    params: [p1, p2] - depolarizing probabilities for each qubit
    """
    p1, p2 = params[0] , params[1]

    # Kraus operators for single-qubit depolarizing
    K_1 = [np.sqrt(1 - 3 * p1 / 4) * I,
           np.sqrt(p1 / 4) * X,
           np.sqrt(p1 / 4) * Y,
           np.sqrt(p1 / 4) * Z]

    K_2 = [np.sqrt(1 - 3 * p2 / 4) * I,
           np.sqrt(p2 / 4) * X,
           np.sqrt(p2 / 4) * Y,
           np.sqrt(p2 / 4) * Z]

    # Combine for two qubits
    kraus_ops = []
    for k1 in K_1:
        for k2 in K_2:
            kraus_ops.append(np.kron(k1, k2))

    return kraus_ops

#new
def correlated_depolarizing(params: list[float]) -> list[np.ndarray]:
        """Correlated depolarizing noise
        params: [p] - probability of correlated error
        """
        p = params[0]

        # Apply same Pauli error to both qubits
        K0 = np.sqrt(1 - 3 * p) * np.kron(I, I)
        K1 = np.sqrt(p) * np.kron(X, X)
        K2 = np.sqrt(p) * np.kron(Y, Y)
        K3 = np.sqrt(p) * np.kron(Z, Z)

        return [K0, K1, K2, K3]

#new
def independent_dephasing(params: list[float]) -> list[np.ndarray]:
        """Independent dephasing noise
        params: [gamma1, gamma2] - dephasing rates for each qubit
        """
        gamma1, gamma2 = params[0], params[1]

        # Kraus operators
        p1 = gamma1
        p2 = gamma2
        p0 = 1 - p1
        p4 = 1 - p2

        K0 = np.sqrt(p0)*np.sqrt(p4) * np.kron(I, I)
        K1 = np.sqrt(p1)*np.sqrt(p4) * np.kron(Z, I)
        K2 = np.sqrt(p2) * np.sqrt(p0)* np.kron(I, Z)
        K3 = np.sqrt(p1)*np.sqrt(p2) * np.kron(Z, Z)

        return [K0, K1, K2, K3]

#new
def correlated_dephasing(params: list[float]) -> list[np.ndarray]:
    """Correlated dephasing noise
    params: [gamma1, gamma2, gamma_c] - independent and correlated dephasing rates
    """
    gamma1 = params[0]

    # Probabilities for Kraus operators
    p1 = gamma1

    # Kraus operators
    K0 = np.sqrt(1-p1) * np.kron(I, I)  # No error
    K1 = np.sqrt(p1) * np.kron(Z, Z)  # Dephasing on qubit 1 # Correlated dephasing
    return [K0, K1]

def independent_bitflip_kraus(p1, p2):
    """
    Independent bit-flip on both qubits

    Parameters:
    -----------
    p1, p2 : float
        Bit-flip probabilities for qubits 1 and 2

    Returns:
    --------
    kraus_ops : list of 4x4 arrays
        4 Kraus operators
    """
    kraus_ops = [
        np.sqrt((1 - p1) * (1 - p2)) * np.kron(I, I),
        np.sqrt(p1 * (1 - p2)) * np.kron(X, I),
        np.sqrt((1 - p1) * p2) * np.kron(I, X),
        np.sqrt(p1 * p2) * np.kron(X, X)
    ]
    return kraus_ops

#New
def independent_damping(params:list[float]) -> list[np.ndarray]:
    """Independent amplitude damping
    params: [gamma1, gamma2] - damping rates for each qubit
    """
    gamma1, gamma2 = params[0], params[1]

    # Kraus operators for independent damping
    # Qubit 1
    K0_1 = np.array([[1, 0], [0, np.sqrt(1 - gamma1)]])
    K1_1 = np.array([[0, np.sqrt(gamma1)], [0, 0]])
    K0_2 = np.array([[1, 0], [0, np.sqrt(1 - gamma2)]])
    K1_2 = np.array([[0, np.sqrt(gamma2)], [0, 0]])

    # Combined Kraus operators
    K00 = np.kron(K0_1, K0_2)
    K01 = np.kron(K0_1, K1_2)
    K10 = np.kron(K1_1, K0_2)
    K11 = np.kron(K1_1, K1_2)

    return [K00, K01, K10, K11]

#new
def correlated_damping(params: list[float]) -> list[np.ndarray]:
        """Correlated amplitude damping
        params: [gamma1, gamma2, gamma_c] - independent and correlated damping rates
        """
        gamma1, gamma2, gamma_c = params[0], params[1], params[2]

        # Probabilities
        p1 = gamma1
        p2 = gamma2
        pc = gamma_c
        p0 = 1 - p1 - p2 - pc  # Ensure normalization

        # Kraus operators
        I = np.eye(2)
        zero = np.array([[1, 0], [0, 0]])
        one = np.array([[0, 0], [0, 1]])
        sigma_minus = np.array([[0, 0], [1, 0]])

        # No jump
        K0 = np.sqrt(p0) * np.kron(I, I)
        # Damping on qubit 1
        K1 = np.sqrt(p1) * np.kron(sigma_minus, I)
        # Damping on qubit 2
        K2 = np.sqrt(p2) * np.kron(I, sigma_minus)
        # Correlated damping (both qubits)
        K3 = np.sqrt(pc) * np.kron(sigma_minus, sigma_minus)

        return [K0, K1, K2, K3]

def pauli_twirling_kraus():
    """
    Uniform Pauli twirling channel

    Returns:
    --------
    kraus_ops : list of 4x4 arrays
        4 Kraus operators
    """
    kraus_ops = [
        0.5 * np.kron(I, I),
        0.5 * np.kron(X, X),
        0.5 * np.kron(Y, Y),
        0.5 * np.kron(Z, Z)
    ]
    return kraus_ops

def CNOT_kraus():

    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])
    return [CNOT]

def verify_completeness(kraus_ops):
    """
    Verify that Kraus operators satisfy completeness relation
    ∑ᵢ Kᵢ†Kᵢ = I
    """
    total = np.zeros((4, 4), dtype=complex)
    for K in kraus_ops:
        total += np.conj(K).T @ K

    identity = np.eye(4)
    is_complete = np.allclose(total, identity)

    print(f"Completeness check: {'PASSED' if is_complete else 'FAILED'}")
    if not is_complete:
        print(f"Max deviation: {np.max(np.abs(total - identity))}")

    return is_complete


def cz_gate():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]])

def cr_gate(theta):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, np.exp(1j)]])

def pswap_gate(theta):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), 1j * np.sin(theta), 0],
        [0, 1j * np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def embed_edge_channel_full(total_qubits,sub_channel, edge_position):
    """
    Embed edge channel in full space - returns list of full-space Kraus operators

    This is the channel equivalent of your tensor product for unitaries
    """
    identity_single = np.eye(2, dtype=complex)
    composite_kraus = []

    for K_2q in sub_channel:
        K_2q = K_2q  # Convert to numpy array if needed
        # Embed this 2Q Kraus operator in full space
        if edge_position == "start":
            # K_2q ⊗ I ⊗ I ⊗ ... ⊗ I
            K_full = K_2q
            for i in range(2, total_qubits):
                K_full = np.kron(K_full, identity_single)

        elif edge_position == "end":
            # I ⊗ I ⊗ ... ⊗ I ⊗ K_2q
            K_full = identity_single
            for i in range(1, total_qubits - 2):
                K_full = np.kron(K_full, identity_single)
            K_full = np.kron(K_full, K_2q)

        #composite_kraus.append(DM.DensityMatrix(K_full,canonical_basis(total_qubits)))
        composite_kraus.append(K_full)

    return composite_kraus
def example_embed_edge_channel():
    l1=0.1
    l3=0.01
    t3=0.7
    K1, K2, K3, K4 = phase_covariant_kraus_operators(l3,t3,l1)  # Your 4 operators
    single_kraus = [K1,K2,K3,K4]
    uncorr_kraus = create_uncorrelated_2qubit_kraus(single_kraus)
    embedchan=embed_edge_channel_full(6,uncorr_kraus,"start")
    return embedchan

print(len(example_embed_edge_channel()))


def apply_composite_edge_channel(rho, composite_channel, total_qubits):
    # Convert rho to a NumPy array if it's a DensityMatrix object
    if hasattr(rho, "data"):
        arr = rho.data
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        rho = arr
    # Ensure rho is a 2D array
    rho = np.asarray(rho)
    if rho.ndim != 2:
        raise ValueError("rho must be a 2D array (matrix)")
    rho_out = np.zeros(rho.shape, dtype=complex)
    # Apply each Kraus operator in the composite channel
    for K_full in composite_channel:
        rho_out += K_full @ rho @ K_full.conj().T
    return DM.DensityMatrix(rho_out, canonical_basis(total_qubits))


def pauli_channel(p_x=0, p_y=0, p_z=0):
    """General Pauli channel with bit flip (p_x), bit-phase flip (p_y), phase flip (p_z) probabilities"""
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1 - 2*p_y - 2*p_z, 0, 0],
        [0, 0, 1 - 2*p_x - 2*p_z, 0],
        [0, 0, 0, 1 - 2*p_x - 2*p_y]
    ])
    return DM.DensityMatrix(matrix,canonical_basis(2))

def bit_flip_channel(p):
    """Bit flip channel with probability p"""
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1 - 2*p, 0, 0],
        [0, 0, 1 - 2*p, 0],
        [0, 0, 0, 1]
    ])
    return DM.DensityMatrix(matrix,canonical_basis(2))

def phase_flip_channel(p):
    """Phase flip channel with probability p"""
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1 - 2*p, 0, 0],
        [0, 0, 1 - 2*p, 0],
        [0, 0, 0, 1]
    ])
    return DM.DensityMatrix(matrix,canonical_basis(2))

def depolarizing_channel(p):
    """Depolarizing channel with parameter p"""
    factor = 1 - 4*p/3
    matrix = np.array([
        [1, 0, 0, 0],
        [0, factor, 0, 0],
        [0, 0, factor, 0],
        [0, 0, 0, factor]
    ])
    return DM.DensityMatrix(matrix,canonical_basis(2))

def amplitude_damping_channel(gamma):
    """Amplitude damping channel with parameter gamma"""
    sqrt_factor = np.sqrt(1 - gamma)
    matrix = np.array([
        [1, 0, 0, 0],
        [0, sqrt_factor, 0, 0],
        [0, 0, sqrt_factor, 0],
        [gamma, 0, 0, 1 - gamma]
    ])
    return DM.DensityMatrix(matrix,canonical_basis(2))

def phase_damping_channel(lam):
    """Phase damping channel with parameter lambda"""
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1 - lam, 0, 0],
        [0, 0, 1 - lam, 0],
        [0, 0, 0, 1]
    ])
    return DM.DensityMatrix(matrix,canonical_basis(2))

def generalized_amplitude_damping_channel(gamma, p):
    """Generalized amplitude damping with damping gamma and thermal parameter p"""
    sqrt_factor = np.sqrt(1 - gamma)
    matrix = np.array([
        [1, 0, 0, 0],
        [0, sqrt_factor, 0, 0],
        [0, 0, sqrt_factor, 0],
        [gamma * (2*p - 1), 0, 0, 1 - gamma]
    ])
    return DM.DensityMatrix(matrix,canonical_basis(2))

def x_rotation_error_channel(delta):
    """X-rotation error by angle delta"""
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.cos(delta), -np.sin(delta)],
        [0, 0, np.sin(delta), np.cos(delta)]
    ])
    return DM.DensityMatrix(matrix,canonical_basis(2))

def y_rotation_error_channel(delta):
    """Y-rotation error by angle delta"""
    matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(delta), 0, np.sin(delta)],
        [0, 0, 1, 0],
        [0, -np.sin(delta), 0, np.cos(delta)]
    ])
    return DM.DensityMatrix(matrix,canonical_basis(2))

def z_rotation_error_channel(delta):
    """Z-rotation error by angle delta"""
    matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(delta), -np.sin(delta), 0],
        [0, np.sin(delta), np.cos(delta), 0],
        [0, 0, 0, 1]
    ])
    return DM.DensityMatrix(matrix,canonical_basis(2))

# Utility function to apply channel to augmented Bloch vector [1, x, y, z]
def apply_channel(r_aug, channel_matrix):
    """Apply channel matrix to augmented Bloch vector [1, x, y, z]"""
    return channel_matrix @ r_aug

# Utility function to compose channels
def compose_channels(channel1, channel2):
    """Compose two channel matrices: first channel1, then channel2"""
    return channel2 @ channel1

# Example usage:
channel = depolarizing_channel(0.1)
channel.plot()
# r_augmented = np.array([1, 0, 0, 1])  # [1, x, y, z]
# r_new = apply_channel(r_augmented, channel)
verify_completeness(independent_dephasing([0.1,0.2]))