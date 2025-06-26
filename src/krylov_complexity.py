# For spin systems, operators can be represented as:
# - Dense matrices (small systems)
# - Sparse matrices (medium systems)
# - Matrix Product Operators (large systems)
import numpy as np
from scipy.linalg import expm

import numpy as np
from itertools import product
from typing import List, Tuple, Dict
import time
import matplotlib.pyplot as plt

class KrylovComplexity:
    def __init__(self, hamiltonian, dt, O0):
        # Set Hamiltonian
        self.H = hamiltonian

        # Set time step interval
        self.dt = dt

        # Set initial operator
        self.O0 = O0

        # Generate unitary evolution operator
        self.U = expm(-1j * hamiltonian * dt)

    def apply_unitary(self, operator):
        """Apply unitary: U†OU"""
        return self.U.conj().T @ operator @ self.U
    
    def inner_product(self, op1, op2):
        """Hilbert-Schmidt inner product"""
        return np.trace(op1.conj().T @ op2) / op1.shape[0]
    
    def norm(self, operator):
        """Operator norm"""
        return np.sqrt(np.real(self.inner_product(operator, operator)))


class QuantumOperatorAnalyzer:
    """
    A class for analyzing operator spreading in quantum circuits.
    """

    def __init__(self, n_qubits: int, symmetry: str = None):
        """
        Initialize the analyzer for n qubits.

        Args:
            n_qubits: Number of qubits in the system
            symmetry: Symmetry type ('Z2', 'U1', or None)
        """
        self.n_qubits = n_qubits
        self.symmetry = symmetry
        self.pauli_basis = self._generate_pauli_basis()
        self.pauli_matrices = self._get_pauli_matrices()

        # Only pre-compute for small systems
        if len(self.pauli_basis) < 5000:  # Only for manageable sizes
            self._precompute_optimization_data()
        else:
            print(f"Skipping pre-computation for large basis ({len(self.pauli_basis):,} states)")
            self.pauli_tensor = None

    def _get_pauli_matrices(self) -> Dict[str, np.ndarray]:
        """Generate the 2x2 Pauli matrices."""
        return {
            'I': np.array([[1, 0], [0, 1]], dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex)
        }

    def _precompute_optimization_data(self):
        """Pre-compute data structures for optimization."""
        print(f"Pre-computing optimization data for {len(self.pauli_basis):,} Pauli strings...")
        start_time = time.time()

        # Pre-compute all Pauli matrices as a 3D tensor
        dim = 2 ** self.n_qubits
        self.pauli_tensor = np.zeros((len(self.pauli_basis), dim, dim), dtype=complex)

        for i, pauli_string in enumerate(self.pauli_basis):
            self.pauli_tensor[i] = self.pauli_string_to_matrix(pauli_string)

        print(f"Pre-computation completed in {time.time() - start_time:.2f}s")

    def compute_overlap_vectorized(self, operator: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Vectorized overlap computation using Einstein summation.
        Falls back to batched computation for large systems.
        """
        if self.pauli_tensor is None:
            if verbose:
                print("Pre-computed tensor not available, using batched computation...")
            return self.compute_overlap_batched(operator, verbose=verbose)

        if verbose:
            print(f"Computing {len(self.pauli_basis):,} overlaps (vectorized)...")
            start_time = time.time()

        # Vectorized trace computation: trace(O @ P_i) for all i
        overlaps = np.einsum('ij,kji->k', operator, self.pauli_tensor)

        if verbose:
            total_time = time.time() - start_time
            print(f"Vectorized computation completed in {total_time:.3f}s")

        return overlaps

    def compute_overlap_batched(self, operator: np.ndarray, batch_size: int = 1000,
                                verbose: bool = False) -> np.ndarray:
        """
        Batched computation to balance speed and memory.
        This is the recommended method for large systems.
        """
        if verbose:
            print(f"Computing {len(self.pauli_basis):,} overlaps (batched, size={batch_size})...")
            start_time = time.time()

        overlaps = np.zeros(len(self.pauli_basis), dtype=complex)
        n_batches = (len(self.pauli_basis) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.pauli_basis))

            if verbose and batch_idx % 5 == 0:
                progress = 100 * start_idx / len(self.pauli_basis)
                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx + 1}/{n_batches}: {progress:.1f}% ({elapsed:.1f}s)")

            # Compute overlaps for this batch (no pre-computation)
            for i in range(start_idx, end_idx):
                pauli_matrix = self.pauli_string_to_matrix(self.pauli_basis[i])
                overlaps[i] = (1/len(operator))*(np.trace(operator @ pauli_matrix))

        if verbose:
            total_time = time.time() - start_time
            print(f"Batched computation completed in {total_time:.3f}s")

        return overlaps

    def _generate_pauli_basis(self) -> List[str]:
        """
        Generate Pauli strings respecting symmetries efficiently.

        Returns:
            List of Pauli strings respecting the specified symmetry
        """
        if self.symmetry is None:
            return self._generate_full_basis()
        elif self.symmetry == 'Z2':
            return self._generate_z2_basis_direct()
        elif self.symmetry == 'U1':
            return self._generate_u1_basis_direct()
        else:
            raise ValueError(f"Unknown symmetry: {self.symmetry}")

    def _generate_full_basis(self) -> List[str]:
        """Generate all possible Pauli strings for N qubits."""
        pauli_chars = ['I', 'X', 'Y', 'Z']
        return [''.join(p) for p in product(pauli_chars, repeat=self.n_qubits)]

    def _generate_z2_basis_direct(self) -> List[str]:
        """
        Generate only Z2-symmetric Pauli strings (even X+Y count) directly.

        Returns:
            List of Z2-symmetric Pauli strings
        """
        pauli_chars = ['I', 'X', 'Y', 'Z']
        z2_strings = []

        # Generate only strings with even X+Y count
        for pauli_tuple in product(pauli_chars, repeat=self.n_qubits):
            xy_count = pauli_tuple.count('X') + pauli_tuple.count('Y')
            if xy_count % 2 == 0:
                z2_strings.append(''.join(pauli_tuple))

        return z2_strings

    def _generate_u1_basis_direct(self) -> List[str]:
        """
        Generate only U1-symmetric Pauli strings (equal X and Y count) directly.

        Uses combinatorial enumeration for efficiency.

        Returns:
            List of U1-symmetric Pauli strings
        """
        from itertools import combinations

        u1_strings = []

        # For each possible number of X's (and equal Y's)
        max_xy_pairs = self.n_qubits // 2

        for num_x in range(max_xy_pairs + 1):
            num_y = num_x  # U1 symmetry: equal X and Y
            num_iz = self.n_qubits - num_x - num_y

            if num_iz < 0:
                continue

            # Choose positions for X's
            for x_positions in combinations(range(self.n_qubits), num_x):
                remaining_positions = [i for i in range(self.n_qubits) if i not in x_positions]

                # Choose positions for Y's from remaining positions
                for y_positions in combinations(remaining_positions, num_y):
                    iz_positions = [i for i in remaining_positions if i not in y_positions]

                    # For each way to assign I and Z to remaining positions
                    for num_i in range(len(iz_positions) + 1):
                        num_z = len(iz_positions) - num_i

                        # Choose positions for I's from I/Z positions
                        for i_positions in combinations(iz_positions, num_i):
                            z_positions = [i for i in iz_positions if i not in i_positions]

                            # Construct the Pauli string
                            pauli_string = ['I'] * self.n_qubits

                            for pos in x_positions:
                                pauli_string[pos] = 'X'
                            for pos in y_positions:
                                pauli_string[pos] = 'Y'
                            for pos in z_positions:
                                pauli_string[pos] = 'Z'
                            # I positions already set

                            u1_strings.append(''.join(pauli_string))

        return u1_strings

    def _filter_z2_symmetric(self, pauli_strings: List[str]) -> List[str]:
        """
        Filter Pauli strings with even number of X+Y operators (Z2 symmetry).
        DEPRECATED: Use _generate_z2_basis_direct() instead.
        """
        filtered = []
        for s in pauli_strings:
            xy_count = s.count('X') + s.count('Y')
            if xy_count % 2 == 0:
                filtered.append(s)
        return filtered

    def _filter_u1_symmetric(self, pauli_strings: List[str]) -> List[str]:
        """
        Filter Pauli strings with equal number of X and Y operators (U1 symmetry).
        DEPRECATED: Use _generate_u1_basis_direct() instead.
        """
        filtered = []
        for s in pauli_strings:
            x_count = s.count('X')
            y_count = s.count('Y')
            if x_count == y_count:
                filtered.append(s)
        return filtered

    def pauli_string_to_matrix(self, pauli_string: str) -> np.ndarray:
        """
        Convert a Pauli string to its matrix representation.

        Args:
            pauli_string: String of Pauli operators (e.g., 'XYZ')

        Returns:
            Matrix representation of the Pauli string
        """
        if len(pauli_string) != self.n_qubits:
            raise ValueError(f"Pauli string length must be {self.n_qubits}")

        result = self.pauli_matrices[pauli_string[0]]
        for pauli_char in pauli_string[1:]:
            result = np.kron(result, self.pauli_matrices[pauli_char])
        return result

    def apply_unitary_to_operator(self, operator: np.ndarray, unitary: np.ndarray) -> np.ndarray:
        """
        Apply unitary evolution to an operator: U † O U.

        Args:
            operator: The operator to evolve
            unitary: The unitary operator

        Returns:
            Evolved operator
        """
        return unitary.conj().T @ operator @ unitary

    def compute_overlap_with_pauli_basis(self, operator: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Compute overlaps Tr[O P_i] for all Pauli strings P_i.

        Args:
            operator: The operator to analyze
            verbose: Print progress for debugging

        Returns:
            Array of overlaps with each Pauli string
        """
        overlaps = np.zeros(len(self.pauli_basis), dtype=complex)
        total_basis = len(self.pauli_basis)

        if verbose:
            print(f"Computing {total_basis:,} overlaps...")

        for i, pauli_string in enumerate(self.pauli_basis):
            if verbose and i % 5000 == 0:
                print(f"  Progress: {i:,}/{total_basis:,} ({100 * i / total_basis:.1f}%)")

            pauli_matrix = self.pauli_string_to_matrix(pauli_string)
            overlaps[i] = 1/(total_basis)*(np.trace(operator @ pauli_matrix))

        if verbose:
            print(f"  Completed: {total_basis:,}/{total_basis:,} (100.0%)")

        return overlaps

    def compute_weight_distribution(self, overlaps: np.ndarray) -> np.ndarray:
        """
        Compute the weight distribution from overlaps.

        Args:
            overlaps: Complex overlaps with Pauli basis

        Returns:
            Real-valued weight distribution
        """
        return np.abs(overlaps) ** 2

    def evolve_operator(self, initial_pauli_string: str, unitaries: List[np.ndarray],
                        time_steps: int, verbose: bool = False, method: str = 'vectorized') -> Tuple[
        List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Time evolve an initial Pauli operator and track its spreading.

        Args:
            initial_pauli_string: Initial Pauli string
            unitaries: List of unitaries to apply cyclically
            time_steps: Number of time steps to evolve
            verbose: Print progress for debugging
            method: Computation method ('vectorized', 'batched', or 'original')

        Returns:
            Tuple of (evolved_operators, overlaps_per_time, weights_per_time)
        """
        if verbose:
            print(f"Starting evolution using {method} method")
            print(f"System: {self.n_qubits} qubits, {len(self.pauli_basis):,} basis states")

        # Choose overlap computation method
        if method == 'vectorized':
            overlap_func = self.compute_overlap_vectorized
        elif method == 'batched':
            overlap_func = self.compute_overlap_batched
        elif method == 'original':
            overlap_func = self.compute_overlap_with_pauli_basis
        else:
            raise ValueError(f"Unknown method: {method}")

        # Initialize with the initial Pauli operator
        current_operator = self.pauli_string_to_matrix(initial_pauli_string)

        if verbose:
            print(f"Operator matrix size: {current_operator.shape}")

        # Storage for results
        evolved_operators = [current_operator.copy()]
        overlaps_per_time = []
        weights_per_time = []

        # Compute initial overlaps and weights
        if verbose:
            print("Computing initial overlaps...")
        initial_overlaps = overlap_func(current_operator, verbose=verbose)
        overlaps_per_time.append(initial_overlaps)
        weights_per_time.append(self.compute_weight_distribution(initial_overlaps))

        # Time evolution
        for t in range(time_steps):
            if verbose:
                print(f"\n--- Time step {t + 1}/{time_steps} ---")

            # Apply unitary cyclically
            unitary = unitaries[t % len(unitaries)]
            if verbose:
                print(f"Applying unitary {(t % len(unitaries)) + 1}/{len(unitaries)}")

            current_operator = self.apply_unitary_to_operator(current_operator, unitary)

            # Store evolved operator
            evolved_operators.append(current_operator.copy())

            # Compute overlaps and weights
            if verbose:
                print(f"Computing overlaps for step {t + 1}...")
            overlaps = overlap_func(current_operator, verbose=verbose)
            weights = self.compute_weight_distribution(overlaps)

            overlaps_per_time.append(overlaps)
            weights_per_time.append(weights)

        if verbose:
            print(f"\n✓ Evolution completed using {method} method!")

        return evolved_operators, overlaps_per_time, weights_per_time


class GramSchmidtOrthogonalizer:
    """
    A class for Gram-Schmidt orthogonalization of evolved operators.
    """

    def __init__(self, analyzer):
        """
        Initialize with a quantum operator analyzer.

        Args:
            analyzer: QuantumOperatorAnalyzer instance
        """
        self.analyzer = analyzer

    def vectorize_operator(self, operator: np.ndarray) -> np.ndarray:
        """Vectorize an operator matrix for orthogonalization."""
        return operator.flatten()

    def matrix_from_vector(self, vector: np.ndarray) -> np.ndarray:
        """Reconstruct matrix from vectorized form."""
        dim = int(np.sqrt(len(vector)))
        return vector.reshape(dim, dim)

    def gram_schmidt_orthogonalization(self, operators: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """
        Perform Gram-Schmidt orthogonalization on a list of operators.

        Args:
            operators: List of operator matrices

        Returns:
            Tuple of (orthogonal_operators, transformation_matrix, coefficients_matrix)
        """
        # Vectorize all operators
        vectors = [self.vectorize_operator(op) for op in operators]
        n_ops = len(vectors)
        dim = len(vectors[0])

        # Initialize orthogonal vectors and coefficient matrices
        orthogonal_vectors = []
        transformation_matrix = np.zeros((n_ops, n_ops), dtype=complex)
        coefficients_matrix = np.zeros((n_ops, n_ops), dtype=complex)

        for i in range(n_ops):
            # Start with current vector
            current_vector = vectors[i].copy()
            coefficients = np.zeros(n_ops, dtype=complex)
            coefficients[i] = 1  # Initially, the vector is itself

            # Subtract projections onto previous orthogonal vectors
            for j in range(len(orthogonal_vectors)):
                projection = np.vdot(orthogonal_vectors[j], current_vector)
                current_vector = current_vector - projection * orthogonal_vectors[j]
                coefficients -= projection * coefficients_matrix[j]

            # Normalize
            norm = np.linalg.norm(current_vector)
            if norm > 1e-12:  # Avoid division by zero
                current_vector = current_vector / norm
                coefficients = coefficients / norm
                orthogonal_vectors.append(current_vector)

                # Store transformation coefficients
                for j in range(len(orthogonal_vectors)):
                    transformation_matrix[i, j] = np.vdot(orthogonal_vectors[j], vectors[i])

                # Save the linear combination coefficients
                coefficients_matrix[i] = coefficients

        # Convert back to operator matrices
        orthogonal_operators = [self.matrix_from_vector(vec) for vec in orthogonal_vectors]

        return orthogonal_operators, transformation_matrix, coefficients_matrix

    def express_in_orthogonal_basis(self, operators: List[np.ndarray],
                                    orthogonal_operators: List[np.ndarray]) -> List[np.ndarray]:
        """
        Express operators in the orthogonal basis.

        Args:
            operators: List of operators to express
            orthogonal_operators: Orthogonal basis operators

        Returns:
            List of coefficient arrays for each operator
        """
        orthogonal_vectors = [self.vectorize_operator(op) for op in orthogonal_operators]

        coefficients_list = []
        for operator in operators:
            op_vector = self.vectorize_operator(operator)
            coefficients = np.zeros(len(orthogonal_vectors), dtype=complex)

            for i, ortho_vec in enumerate(orthogonal_vectors):
                coefficients[i] = np.vdot(ortho_vec, op_vector)

            coefficients_list.append(coefficients)

        return coefficients_list


def analyze_operator_spreading(n_qubits: int, initial_pauli_string: str,
                               unitaries: List[np.ndarray], time_steps: int,
                               symmetry: str = None, verbose: bool = False, method: str = 'vectorized'):
    """
    Main function to analyze operator spreading.

    Args:
        n_qubits: Number of qubits
        initial_pauli_string: Initial Pauli operator
        unitaries: List of unitaries for time evolution
        time_steps: Number of evolution steps
        symmetry: Symmetry type ('Z2', 'U1', or None)
        verbose: Print progress for debugging
        method: Computation method ('vectorized', 'batched', or 'original')

    Returns:
        Dictionary containing all analysis results
    """
    # Initialize analyzer
    analyzer = QuantumOperatorAnalyzer(n_qubits, symmetry)

    if verbose:
        print(f"Initialized analyzer with {symmetry} symmetry")
        print(f"Basis size: {len(analyzer.pauli_basis):,}")

    # Evolve operator and get spreading data
    evolved_ops, overlaps, weights = analyzer.evolve_operator(
        initial_pauli_string, unitaries, time_steps, verbose=verbose, method=method
    )

    return {
        'evolved_operators': evolved_ops,
        'overlaps_per_time': overlaps,
        'weights_per_time': weights,
        'pauli_basis': analyzer.pauli_basis,
        'basis_size': len(analyzer.pauli_basis),
        'symmetry': symmetry,
        'method': method
    }


# Example usage functions with optimization options
def analyze_with_z2_symmetry(n_qubits: int, initial_pauli_string: str,
                             unitaries: List[np.ndarray], time_steps: int,
                             verbose: bool = False, method: str = 'vectorized'):
    """Analyze with Z2 symmetry (even X+Y count)."""
    return analyze_operator_spreading(n_qubits, initial_pauli_string,
                                      unitaries, time_steps, symmetry='Z2',
                                      verbose=verbose, method=method)


def analyze_with_u1_symmetry(n_qubits: int, initial_pauli_string: str,
                             unitaries: List[np.ndarray], time_steps: int,
                             verbose: bool = False, method: str = 'vectorized'):
    """Analyze with U1 symmetry (equal X and Y count)."""
    return analyze_operator_spreading(n_qubits, initial_pauli_string,
                                      unitaries, time_steps, symmetry='U1',
                                      verbose=verbose, method=method)


# Efficiency comparison function
def compare_basis_generation_efficiency():
    """
    Compare efficiency of direct generation vs filtering approaches.
    """
    print("=== Basis Generation Efficiency Comparison ===\n")

    import time

    for n_qubits in [6, 8, 10]:
        print(f"N = {n_qubits} qubits:")

        # Full basis size
        full_size = 4 ** n_qubits
        print(f"  Full basis size: {full_size:,}")

        # Z2 symmetry comparison
        print("  Z2 Symmetry:")

        # Direct generation timing
        start_time = time.time()
        analyzer_z2_direct = QuantumOperatorAnalyzer(n_qubits, symmetry='Z2')
        z2_direct_time = time.time() - start_time
        z2_size = len(analyzer_z2_direct.pauli_basis)

        print(f"    Direct generation: {z2_size:,} strings in {z2_direct_time:.4f}s")
        print(f"    Reduction factor: {full_size / z2_size:.1f}x")

        # U1 symmetry
        print("  U1 Symmetry:")
        start_time = time.time()
        analyzer_u1_direct = QuantumOperatorAnalyzer(n_qubits, symmetry='U1')
        u1_direct_time = time.time() - start_time
        u1_size = len(analyzer_u1_direct.pauli_basis)

        print(f"    Direct generation: {u1_size:,} strings in {u1_direct_time:.4f}s")
        print(f"    Reduction factor: {full_size / u1_size:.1f}x")
        print()


# Updated demonstration function
def demonstrate_symmetry_benefits():
    """
    Demonstrate the computational benefits of using symmetries with direct generation.
    """
    n_qubits = 8

    print(f"=== Symmetry Benefits for {n_qubits} Qubits ===\n")

    # Without symmetry
    analyzer_full = QuantumOperatorAnalyzer(n_qubits)
    print(f"Full basis size: {len(analyzer_full.pauli_basis):,} = 4^{n_qubits}")

    # With Z2 symmetry
    analyzer_z2 = QuantumOperatorAnalyzer(n_qubits, symmetry='Z2')
    print(f"Z2 basis size: {len(analyzer_z2.pauli_basis):,}")

    # With U1 symmetry
    analyzer_u1 = QuantumOperatorAnalyzer(n_qubits, symmetry='U1')
    print(f"U1 basis size: {len(analyzer_u1.pauli_basis):,}")

    print(f"\nMemory reduction factors:")
    print(f"Z2: {len(analyzer_full.pauli_basis) / len(analyzer_z2.pauli_basis):.1f}x")
    print(f"U1: {len(analyzer_full.pauli_basis) / len(analyzer_u1.pauli_basis):.1f}x")

    print(f"\nExample Z2 strings (first 10):")
    for i, s in enumerate(analyzer_z2.pauli_basis[:10]):
        xy_count = s.count('X') + s.count('Y')
        print(f"  {s} (X+Y count: {xy_count})")

    print(f"\nExample U1 strings (first 10):")
    for i, s in enumerate(analyzer_u1.pauli_basis[:10]):
        x_count = s.count('X')
        y_count = s.count('Y')
        print(f"  {s} (X: {x_count}, Y: {y_count})")


def orthogonalize_evolved_operators(evolved_operators: List[np.ndarray],
                                    analyzer: QuantumOperatorAnalyzer):
    """
    Perform Gram-Schmidt orthogonalization on evolved operators.

    Args:
        evolved_operators: List of evolved operator matrices
        analyzer: QuantumOperatorAnalyzer instance

    Returns:
        Dictionary containing orthogonalization results
    """
    # Initialize orthogonalizer
    orthogonalizer = GramSchmidtOrthogonalizer(analyzer)

    # Perform Gram-Schmidt orthogonalization (now returns 3 values)
    orthogonal_ops, transform_matrix, coefficients_matrix = orthogonalizer.gram_schmidt_orthogonalization(evolved_operators)

    # Express original operators in orthogonal basis
    coefficients = orthogonalizer.express_in_orthogonal_basis(
        evolved_operators, orthogonal_ops
    )

    return {
        'orthogonal_operators': orthogonal_ops,
        'transformation_matrix': transform_matrix,
        'coefficients_in_orthogonal_basis': coefficients,
        'gs_in_original_coefficients': coefficients_matrix  # key added to invert coeff
    }



def create_partial_swap_gate(theta: float) -> np.ndarray:
    """
    Create a partial SWAP gate (PSWAP) with parameter theta.
    Acts as identity on |00> and |11>, and as a rotation in the |01>, |10> subspace.

    Args:
        theta: Swap angle (radians)

    Returns:
        4x4 PSWAP(theta) gate matrix
    """
    pswap = np.array([
        [1, 0,           0,          0],
        [0, np.cos(theta), 1j*np.sin(theta), 0],
        [0, 1j*np.sin(theta), np.cos(theta), 0],
        [0, 0,           0,          1]
    ], dtype=complex)
    return pswap


def create_two_qubit_gate_on_full_system(gate_2q: np.ndarray, qubit_pair: tuple, n_qubits: int) -> np.ndarray:
    """
    Embed a 2-qubit gate into the full n-qubit system.

    Args:
        gate_2q: 4x4 two-qubit gate
        qubit_pair: Tuple (i, j) of qubits to apply gate to
        n_qubits: Total number of qubits

    Returns:
        2^n_qubits x 2^n_qubits unitary for full system
    """
    i, j = qubit_pair
    if i >= j or i < 0 or j >= n_qubits:
        raise ValueError(f"Invalid qubit pair {qubit_pair} for {n_qubits} qubits")

    # Identity matrices
    I = np.eye(2, dtype=complex)

    # Build the full system unitary
    full_gate = np.array([[1]], dtype=complex)

    for qubit in range(n_qubits):
        if qubit == i:
            # Start building the 2-qubit gate
            if j == i + 1:
                # Adjacent qubits - directly tensor the 2-qubit gate
                full_gate = np.kron(full_gate, gate_2q)
                qubit += 1  # Skip next qubit since we handled the pair
            else:
                # Non-adjacent case - more complex (not needed for brickwork)
                raise NotImplementedError("Non-adjacent qubits not implemented")
        elif qubit == j and j == i + 1:
            # Already handled in the i case for adjacent qubits
            continue
        else:
            # Apply identity to this qubit
            full_gate = np.kron(full_gate, I)

    return full_gate


def create_brickwork_unitaries(n_qubits: int,theta:float) -> List[np.ndarray]:
    """
    Create brickwork circuit unitaries for n qubits.

    Brickwork pattern:
    - Even layers: pairs (0,1), (2,3), (4,5), (6,7), ...
    - Odd layers: pairs (1,2), (3,4), (5,6), (7,8), ...

    Args:
        n_qubits: Number of qubits (should be even for full coverage)

    Returns:
        List of two unitaries [U_even, U_odd] for brickwork pattern
    """
    if n_qubits % 2 != 0:
        print(f"Warning: {n_qubits} is odd. Last qubit will not participate in all gates.")

    pswap = create_partial_swap_gate(theta)

    # Even layer: (0,1), (2,3), (4,5), (6,7)
    U_even = np.eye(2 ** n_qubits, dtype=complex)
    for i in range(0, n_qubits - 1, 2):
        # Apply PSWAP to qubits (i, i+1)
        gate_full = create_two_qubit_gate_on_full_system(pswap, (i, i + 1), n_qubits)
        U_even = gate_full @ U_even

    # Odd layer: (1,2), (3,4), (5,6), (7,8)
    U_odd = np.eye(2 ** n_qubits, dtype=complex)
    for i in range(1, n_qubits - 1, 2):
        # Apply PSWAP to qubits (i, i+1)
        gate_full = create_two_qubit_gate_on_full_system(pswap, (i, i + 1), n_qubits)
        U_odd = gate_full @ U_odd

    return [U_even, U_odd]


def run_8_qubit_brickwork_example(theta:float):
    """
    Run the 8-qubit brickwork circuit analysis with PSWAP gates.
    """
    print("=== 8-Qubit Brickwork Circuit with PSWAP Gates ===\n")

    # System parameters
    n_qubits = 8
    initial_operator = "ZIIIIIII"  # X on first and last qubits
    time_steps = 20

    print(f"System size: {n_qubits} qubits")
    print(f"Initial operator: {initial_operator}")
    print(f"Time steps: {time_steps}")

    # Create brickwork unitaries
    print("\nCreating brickwork unitaries...")
    unitaries = create_brickwork_unitaries(n_qubits,theta)
    print(f"Created {len(unitaries)} unitaries:")
    print(f"- U_even: applies PSWAP to pairs (0,1), (2,3), (4,5), (6,7)")
    print(f"- U_odd: applies PSWAP to pairs (1,2), (3,4), (5,6)")

    # Check unitary dimensions
    print(f"\nUnitary dimensions: {unitaries[0].shape}")
    print(f"Hilbert space dimension: 2^{n_qubits} = {2 ** n_qubits}")

    # Analyze without symmetry (will be large!)
    print(f"\n=== Analysis without symmetry ===")
    print(f"Full Pauli basis size: 4^{n_qubits} = {4 ** n_qubits:,}")
    print("This is computationally intensive for 8 qubits!")

    # Analyze with Z2 symmetry
    print(f"\n=== Analysis with Z2 symmetry ===")
    print("Using Z2 symmetry (even number of X+Y operators)")

    # Check basis size with Z2 symmetry (using direct generation)
    analyzer_z2 = QuantumOperatorAnalyzer(n_qubits, symmetry='Z2')
    print(f"Z2 basis size: {len(analyzer_z2.pauli_basis):,}")
    print(f"Reduction factor: {4 ** n_qubits / len(analyzer_z2.pauli_basis):.1f}x")

    # Run the Z2 analysis
    print(f"\nRunning Z2 symmetric analysis...")
    results_z2 = analyze_with_z2_symmetry(n_qubits, initial_operator, unitaries, time_steps)

    print(f"✓ Z2 Analysis completed!")
    print(f"Evolved {len(results_z2['evolved_operators'])} operators")
    print(f"Tracked {len(results_z2['overlaps_per_time'])} time steps")
    print(f"Final weight distribution shape: {results_z2['weights_per_time'][-1].shape}")

    # Analyze with U1 symmetry
    print(f"\n=== Analysis with U1 symmetry ===")
    print("Using U1 symmetry (equal number of X and Y operators)")

    analyzer_u1 = QuantumOperatorAnalyzer(n_qubits, symmetry='U1')
    print(f"U1 basis size: {len(analyzer_u1.pauli_basis):,}")
    print(f"Reduction factor: {4 ** n_qubits / len(analyzer_u1.pauli_basis):.1f}x")

    # Check if initial operator respects U1 symmetry
    x_count = initial_operator.count('X')
    y_count = initial_operator.count('Y')
    if x_count == y_count:
        print(f"Initial operator has equal X ({x_count}) and Y ({y_count}) → U1 analysis valid")

        results_u1 = analyze_with_u1_symmetry(n_qubits, initial_operator, unitaries, time_steps)
        print(f"✓ U1 analysis completed!")
        print(f"U1 basis size used: {results_u1['basis_size']:,}")
    else:
        print(f"Initial operator has X={x_count}, Y={y_count} → Not U1 symmetric")
        print("Skipping U1 analysis for this initial operator")

    return unitaries, results_z2


def demonstrate_pswap_properties(theta:float):
    """
    Demonstrate properties of the PSWAP gate.
    """
    print("\n=== PSWAP Gate Properties ===")

    pswap = create_partial_swap_gate(theta)
    print("PSWAP gate matrix:")
    print(pswap)

    # Check unitarity
    should_be_identity = pswap @ pswap.conj().T
    is_unitary = np.allclose(should_be_identity, np.eye(4))
    print(f"\nIs unitary: {is_unitary}")

    # Check if it's actually different from SWAP
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    is_same_as_swap = np.allclose(pswap, swap)
    print(f"Same as regular SWAP: {is_same_as_swap}")

    # Show the phase difference
    phase_factor = pswap[1, 2] / swap[1, 2]  # Should be e^(iφ)
    print(f"Phase factor: {phase_factor}")
    print(f"Phase angle: {np.angle(phase_factor):.3f} radians = {np.angle(phase_factor) * 180 / np.pi:.1f}°")


def demonstrate_efficiency_improvements():
    """
    Show the efficiency improvements from direct basis generation.
    """
    print("\n=== Efficiency Improvements ===")

    import time

    n_qubits = 8

    print(f"Comparing basis generation for {n_qubits} qubits:")

    # Z2 symmetry timing
    start_time = time.time()
    analyzer_z2 = QuantumOperatorAnalyzer(n_qubits, symmetry='Z2')
    z2_time = time.time() - start_time
    print(f"Z2 basis generation: {len(analyzer_z2.pauli_basis):,} strings in {z2_time:.4f}s")

    # U1 symmetry timing
    start_time = time.time()
    analyzer_u1 = QuantumOperatorAnalyzer(n_qubits, symmetry='U1')
    u1_time = time.time() - start_time
    print(f"U1 basis generation: {len(analyzer_u1.pauli_basis):,} strings in {u1_time:.4f}s")

    # Show some example strings
    print(f"\nFirst 5 Z2 strings: {analyzer_z2.pauli_basis[:5]}")
    print(f"First 5 U1 strings: {analyzer_u1.pauli_basis[:5]}")

def plot_final_weight_histogram(results, bins=50):
    """
    Plot a histogram of nonzero weights at the last time step.

    Args:
        results: Output from analyze_with_z2_symmetry() or similar
        bins: Number of histogram bins
    """
    final_weights = np.array(results['weights_per_time'][-1])
    nonzero_weights = final_weights[final_weights > 0]

    plt.figure(figsize=(7, 4))
    plt.hist(nonzero_weights, bins=bins, log=True, color='navy', alpha=0.7)
    plt.xlabel('Weight')
    plt.ylabel('Count (log scale)')
    plt.title('Distribution of Nonzero Weights at Final Time Step')
    plt.tight_layout()
    plt.show()

def plot_nonzero_operator_count(results):
    """
    Plot the number of nonzero-weight operators as a function of time.

    Args:
        results: Output from analyze_with_z2_symmetry() or similar
    """
    weights_per_time = results['weights_per_time']
    n_time_steps = len(weights_per_time)

    nonzero_counts = [np.sum(np.array(w) > 0) for w in weights_per_time]

    plt.figure(figsize=(7, 4))
    plt.plot(range(n_time_steps), nonzero_counts, marker='o', color='purple')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Nonzero Operators')
    plt.title('Number of Explored Operators Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def quick_plot(results):
    """
    Super simple single plot - just the weight of top operators over time.
    """
    weights_per_time = results['weights_per_time']
    pauli_basis = results['pauli_basis']
    n_time_steps = len(weights_per_time)

    plt.figure(figsize=(10, 6))

    # Find top 10 operators by maximum weight
    all_max_weights = []
    for i, pauli_string in enumerate(pauli_basis):
        max_weight = max(weights_per_time[t][i] for t in range(n_time_steps))
        all_max_weights.append((max_weight, i, pauli_string))

    all_max_weights.sort(reverse=True)

    # Plot top 10
    for rank, (_, idx, pauli_string) in enumerate(all_max_weights[:100]):
        weights_over_time = [weights_per_time[t][idx] for t in range(n_time_steps)]
        plt.plot(range(n_time_steps), weights_over_time, 'o-', linewidth=2,
                 label=f'{pauli_string}', markersize=4)

    plt.xlabel('Time Step')
    plt.ylabel('Weight')
    plt.title('Top 10 Operators: Weight Evolution')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def plot_pauli_coefficients_heatmap(coefficients_matrix, pauli_basis, time_steps):
    """
    Visualize the coefficients of Pauli strings as a heatmap.

    Args:
        coefficients_matrix: 2D array of coefficients (time_steps x n_pauli_strings)
        pauli_basis: List of Pauli strings
        time_steps: Number of time steps
    """
    # Compute the magnitude of coefficients
    coefficients_magnitude = np.abs(coefficients_matrix)

    plt.figure(figsize=(12, 8))
    plt.imshow(coefficients_magnitude.T, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='|Coefficient|')
    plt.xlabel('Time Step')
    plt.ylabel('Pauli String Index')
    plt.title('Heatmap of Pauli String Coefficients Over Time')
    plt.xticks(range(0, time_steps, max(1, time_steps // 10)))
    plt.yticks(range(len(pauli_basis)), pauli_basis, fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_gs_basis_evolution(results, gs_results, figsize=(12, 8)):
    """
    Plot n*|a_n(t)|^2 vs time (top left), leave top right empty,
    and keep the bottom two plots as before.
    """
    coefficients_per_time = gs_results['coefficients_in_orthogonal_basis']
    n_time_steps = len(coefficients_per_time)
    n_gs_operators = len(coefficients_per_time[0]) if coefficients_per_time else 0

    # |a_n(t)|^2
    weights_gs_basis = np.abs(np.array(coefficients_per_time)) ** 2  # shape: (n_time, n_gs_ops)
    time_steps = range(n_time_steps)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Operator Evolution in Gram-Schmidt Basis', fontsize=16)

    # Top left: sum_n n*|a_n(t)|^2
    ax1 = axes[0, 0]
    n_indices = np.arange(n_gs_operators)
    weighted_sum = np.sum(n_indices * weights_gs_basis, axis=1)
    ax1.plot(time_steps, weighted_sum, 'o-', linewidth=2, color='darkorange')
    ax1.set_xlabel('Time Step t')
    ax1.set_ylabel(r'$\sum_n n\,|a_n(t)|^2$')
    ax1.set_title(r'Weighted GS Index: $\sum_n n\,|a_n(t)|^2$')
    ax1.grid(True, alpha=0.3)

    # Top right: empty
    axes[0, 1].axis('off')

    # Bottom left: heatmap
    ax3 = axes[1, 0]
    im = ax3.imshow(weights_gs_basis.T, aspect='auto', cmap='hot', origin='lower')
    ax3.set_xlabel('Time Step t')
    ax3.set_ylabel('GS Operator Index n')
    ax3.set_title('Full GS Evolution Heatmap\n|a_n(t)|² for all n,t')
    plt.colorbar(im, ax=ax3, label='|a_n(t)|²')

    # Bottom right: weight conservation
    ax4 = axes[1, 1]
    total_weights = np.sum(weights_gs_basis, axis=1)
    ax4.plot(time_steps, total_weights, 'b-o', linewidth=3, markersize=5, label='Σ_n |a_n(t)|²')
    for k in [1, 2, 3, 5]:
        if k <= n_gs_operators:
            cum_weight = np.sum(weights_gs_basis[:, :k], axis=1)
            ax4.plot(time_steps, cum_weight, '--', linewidth=2, label=f'Σ(n=0 to {k - 1}) |a_n|²')
    ax4.set_xlabel('Time Step t')
    ax4.set_ylabel('Cumulative Weight')
    ax4.set_title('Weight Distribution in GS Basis\n(Conservation & concentration)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Show PSWAP properties
    # Show efficiency improvements
    theta_try = np.pi/15
    # Run the main analysis
    unitaries = create_brickwork_unitaries(8,theta_try)
    # Step 1: Get your spreading results first
    results = analyze_with_u1_symmetry(8, "ZIIIIIII", unitaries, 5, verbose=True)

    # Step 2: Do Gram-Schmidt orthogonalization
    analyzer = QuantumOperatorAnalyzer(8, symmetry='U1')
    gs_results = orthogonalize_evolved_operators(results['evolved_operators'], analyzer)
    plot_gs_basis_evolution(results, gs_results)
    plot_nonzero_operator_count(results)
    quick_plot(results)
    plot_final_weight_histogram(results)