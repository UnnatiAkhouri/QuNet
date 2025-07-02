import src.setup as setup

sp = setup.sp

from math import comb
import src.density_matrix as DM
from src.ket import energy_basis, canonical_basis

import numpy as np

from scipy.stats import unitary_group

SPARSE_TYPE = setup.SPARSE_TYPE

from scipy.stats import rv_continuous


# code from pennylane
# https://pennylane.ai/qml/demos/tutorial_haar_measure/
class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)


# Samples of theta should be drawn from between 0 and pi
sin_sampler = sin_prob_dist(a=0, b=np.pi)


def random_hamiltonian(num_qbits: int, seed=None):
    """
    Args:
        num_qbits: number of qubits
        seed (optional): the seed for the random number generator.
    Returns: a random hamiltonian with the given number of qubits in the energy basis that preserves energy
    """
    rng = np.random.default_rng(seed)

    blocks = [np.array([[0]])] + [rng.random((comb(num_qbits, e), comb(num_qbits, e))) for e in range(1, num_qbits)] + [np.array([[0]])]
    m = sp.linalg.block_diag(*blocks)
    np.fill_diagonal(m, 0)

    m = m + m.conj().T
    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), energy_basis(num_qbits))


def random_hamiltonian_in_subspace(num_qbits: int, energy_subspace: int, seed=None):
    """
    Args:
        num_qbits: number of qubits
        energy_subspace: the energy subspace to generate the hamiltonian in
        seed (optional): the seed for the random number generator.
    Returns: a random hamiltonian with the given number of qubits in the energy basis that preserves energy within the given subspace
    """
    rng = np.random.default_rng(seed)

    blocks = [np.array([[0]])] + [
        np.zeros((comb(num_qbits, e), comb(num_qbits, e))) if e is not energy_subspace else rng.random((comb(num_qbits, e), comb(num_qbits, e))) * 1j + rng.random(
            (comb(num_qbits, e), comb(num_qbits, e)))
        for e in range(1, num_qbits)] + [
                 np.array([[0]])]

    m = sp.linalg.block_diag(*blocks)
    np.fill_diagonal(m, 0)
    m = (m + m.conj().T)
    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), energy_basis(num_qbits))


def random_hamiltonian_in_subspace_coppying_mathematica(num_qbits: int, energy_subspace: int, seed=None):
    """
    Args:
        num_qbits: number of qubits
        energy_subspace: the energy subspace to generate the hamiltonian in
        seed (optional): the seed for the random number generator.
    Returns: a random hamiltonian with the given number of qubits in the energy basis that preserves energy within the given subspace
    """

    rng = np.random.default_rng(seed)

    def hamiltonian(i1, i2, n):
        base = np.zeros((2 ** n, 2 ** n)) * 1j
        base[i1, i2] = 1j
        base[i2, i1] = -1j
        return base

    # get the indices of the energy subspace (i.e. all numbers with energy_subspace number of 1s in their binary representation)\
    indices = [i for i in range(2 ** num_qbits) if bin(i).count("1") == energy_subspace]

    # sum over all the hamiltonians for each pair of indices
    m = sum([rng.random() * hamiltonian(indices[i1], indices[i2], num_qbits) for i1 in range(len(indices)) for i2 in range(i1)])

    result = DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), canonical_basis(num_qbits))

    result.change_to_energy_basis()

    return result


def random_unitary_in_subspace(num_qbits: int, energy_subspace: int, seed=None):
    rng = np.random.default_rng(seed)

    blocks = [np.array([[1]])] + [np.eye((comb(num_qbits, e))) if e is not energy_subspace else unitary_group.rvs(comb(num_qbits, e), random_state=rng) for e in range(1, num_qbits)] + [
        np.array([[1]])]
    m = sp.linalg.block_diag(*blocks)
    # m = sp.linalg.fractional_matrix_power(m, dt)
    # m[m < 10 ** -5] = 0
    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), energy_basis(num_qbits))


def random_energy_preserving_unitary(num_qbits: int, seed=None) -> DM.DensityMatrix:
    """
    Args:
        num_qbits: the number of qbits in the system.
        seed (optional): the seed for the random number generator.

    Returns: A density matrix object of a random energy preserving unitary on n qbits.

    This function uses the ability to generate completely random unitaries from scipy.stats.unitary_group to generate complete random unitaries in energy preserving subspaces of the full unitary.

    """

    rng = np.random.default_rng(seed)
    blocks = [np.array([[1. + 0j]])] + [unitary_group.rvs(comb(num_qbits, i), random_state=rng) for i in range(1, num_qbits)] + [np.array([[1. + 0j]])]
    m = sp.linalg.block_diag(*blocks)
    # m = sp.linalg.fractional_matrix_power(m, dt)
    # m[m < 10 ** -5] = 0
    return DM.DensityMatrix(SPARSE_TYPE(m), energy_basis(num_qbits))

def haar_random_unitary(theta_divisor=1,phi_divisor=1,omega_divisor=1, seed=None) -> DM.DensityMatrix:
    """
    based off of pennylanes tutorial on haar random unitaries, this only generates a random unitary in the 2 qubit subspace
    """

    rng = np.random.default_rng(seed)
    phi=0
    omega=0
    #phi, omega = 2 * np.pi * rng.uniform(size=2)  # Sample phi and omega as normal
    #theta = sin_sampler.rvs(size=1, random_state=rng)[0] / theta_divisor  # Sample theta from our new distribution

    c = np.cos(np.pi / 15)
    s = np.sin(np.pi / 15)
    data = np.array([[np.exp(-1j * (phi + omega) / 2) * c, -np.exp(1j * (phi - omega) / 2) * s],
                     [np.exp(-1j * (phi - omega) / 2) * s, np.exp(1j * (phi + omega) / 2) * c]])

    m = sp.linalg.block_diag(np.array([[1]]), data, np.array([[1]]))
    return DM.DensityMatrix(DM.SPARSE_TYPE(m, dtype=np.complex64), energy_basis(2))


import numpy as np
import scipy.sparse as scp
import scipy.linalg
from scipy.stats import unitary_group
from math import comb
from typing import Optional, Union


def random_energy_preserving_unitary_optimized(
        num_qbits: int,
        seed: Optional[int] = None,
        dtype: np.dtype = np.complex128,
        return_sparse: bool = True
) -> Union[np.ndarray, scp.csr_matrix]:
    """
    Generate a random energy-preserving unitary matrix with optimizations.

    Args:
        num_qbits: Number of qubits in the system
        seed: Random seed for reproducibility
        dtype: Data type for the matrix (complex128 for double precision)
        return_sparse: Whether to return sparse matrix format

    Returns:
        Random unitary matrix that preserves energy (excitation number)

    Optimizations:
    - Uses double precision by default
    - Validates inputs
    - More efficient sparse construction
    - Optional dense return for small systems
    """

    # Input validation
    if num_qbits <= 0:
        raise ValueError("num_qbits must be positive")
    if num_qbits > 25:  # Practical limit warning
        import warnings
        warnings.warn(f"Large system ({num_qbits} qubits) may cause memory issues")

    rng = np.random.default_rng(seed)

    # Calculate block dimensions
    block_sizes = [comb(num_qbits, i) for i in range(num_qbits + 1)]
    total_dim = sum(block_sizes)

    if not return_sparse and total_dim > 10000:
        import warnings
        warnings.warn("Large dense matrix requested - consider using sparse format")

    # Generate blocks more efficiently
    blocks = []

    # First block (0 excitations): always 1x1 identity
    blocks.append(np.array([[1.0 + 0j]], dtype=dtype))

    # Intermediate blocks: truly random unitaries using Haar measure
    for i in range(1, num_qbits):
        dim = block_sizes[i]
        if dim == 1:
            # Single state - trivial unitary
            blocks.append(np.array([[1.0 + 0j]], dtype=dtype))
        else:
            # Generate Haar-random unitary for this subspace
            U = unitary_group.rvs(dim, random_state=rng).astype(dtype)
            blocks.append(U)

    # Last block (all excitations): always 1x1 identity
    blocks.append(np.array([[1.0 + 0j]], dtype=dtype))

    # Construct block diagonal matrix
    if return_sparse:
        # More memory-efficient sparse construction
        U_full = scp.block_diag(blocks, format='csr', dtype=dtype)
    else:
        U_full = scipy.linalg.block_diag(*blocks).astype(dtype)

    return U_full


def random_energy_preserving_unitary_structured(
        num_qbits: int,
        coupling_strength: float = 1.0,
        seed: Optional[int] = None,
        dtype: np.dtype = np.complex128
) -> scp.csr_matrix:
    """
    Generate a more structured random energy-preserving unitary.

    This version generates unitaries that are closer to physical quantum gates
    by using random rotations with controlled coupling strengths.

    Args:
        num_qbits: Number of qubits
        coupling_strength: Controls how "mixed" the unitary is (0=identity, 1=fully random)
        seed: Random seed
        dtype: Data type
    """

    if not 0 <= coupling_strength <= 1:
        raise ValueError("coupling_strength must be between 0 and 1")

    rng = np.random.default_rng(seed)
    block_sizes = [comb(num_qbits, i) for i in range(num_qbits + 1)]

    blocks = []

    for i, dim in enumerate(block_sizes):
        if dim == 1:
            # Trivial case
            blocks.append(np.array([[1.0 + 0j]], dtype=dtype))
        else:
            if coupling_strength == 0:
                # Identity
                blocks.append(np.eye(dim, dtype=dtype))
            elif coupling_strength == 1:
                # Fully random Haar unitary
                U = unitary_group.rvs(dim, random_state=rng).astype(dtype)
                blocks.append(U)
            else:
                # Interpolate between identity and random unitary
                U_random = unitary_group.rvs(dim, random_state=rng).astype(dtype)
                U_identity = np.eye(dim, dtype=dtype)

                # Use matrix geometric mean for proper interpolation on unitary manifold
                # Simplified approach: exponentiate interpolated generators
                H_random = 1j * scipy.linalg.logm(U_random)
                U_interpolated = scipy.linalg.expm(coupling_strength * H_random)
                blocks.append(U_interpolated.astype(dtype))

    return scp.block_diag(blocks, format='csr', dtype=dtype)


def verify_unitarity(U: Union[np.ndarray, scp.spmatrix], tol: float = 1e-12) -> bool:
    """
    Verify that a matrix is unitary within tolerance.

    Args:
        U: Matrix to check
        tol: Numerical tolerance

    Returns:
        True if unitary, False otherwise
    """
    if scp.issparse(U):
        U = U.toarray()

    # Check U @ U.H = I
    should_be_identity = U @ U.conj().T
    identity = np.eye(U.shape[0])

    return np.allclose(should_be_identity, identity, atol=tol)


# Example usage and testing
if __name__ == "__main__":
    # Test basic functionality
    num_qbits = 12

    print(f"Testing with {num_qbits} qubits")
    print(f"Total Hilbert space dimension: {2 ** num_qbits}")

    # Generate optimized unitary
    U = random_energy_preserving_unitary_optimized(num_qbits, seed=42)
    DM.DensityMatrix(U,energy_basis(num_qbits)).plot()
    print(f"Generated matrix shape: {U.shape}")
    print(f"Matrix type: {type(U)}")
    print(f"Is unitary: {verify_unitarity(U)}")

    # Test structured version
    U_structured = random_energy_preserving_unitary_structured(
        num_qbits, coupling_strength=0.5, seed=42
    )
    DM.DensityMatrix(U_structured, energy_basis(num_qbits)).plot()
    print(f"Structured unitary is unitary: {verify_unitarity(U_structured)}")