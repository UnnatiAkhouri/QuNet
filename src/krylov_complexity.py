# For spin systems, operators can be represented as:
# - Dense matrices (small systems)
# - Sparse matrices (medium systems)
# - Matrix Product Operators (large systems)
import numpy as np
from scipy.linalg import expm
import csv
import numpy as np
from itertools import product
from typing import List, Tuple, Dict
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

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
                dim = len(self.pauli_basis[i])
                overlaps[i] = (1/2**dim)*(np.trace(operator @ pauli_matrix))

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
            dim = len(pauli_string)
            overlaps[i] = (1/2**(dim))*(np.trace(operator @ pauli_matrix))

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
            check_unitarity(unitary)
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

    def filter_linearly_independent(self, operators, tol=1e-12):
        vectors = [op.flatten() for op in operators]
        independent_ops = []
        basis = []
        for v, op in zip(vectors, operators):
            if not basis:
                basis.append(v)
                independent_ops.append(op)
            else:
                proj = sum(np.vdot(b, v) * b for b in basis)
                residue = v - proj
                if np.linalg.norm(residue) > tol:
                    basis.append(residue / np.linalg.norm(residue))
                    independent_ops.append(op)
        return independent_ops

    def robust_modified_gram_schmidt(self, operators):
        d = operators[0].shape[0]
        orthogonal_ops = []

        for i, op in enumerate(operators):
            v = op.copy()

            # Modified GS: orthogonalize against each previous vector
            for k in orthogonal_ops:
                proj = np.trace(k.conj().T @ v) / d
                v = v - proj * k

            norm = np.sqrt(np.real(np.trace(v.conj().T @ v)) / d)

            # Check if norm is large enough
            if norm > 1e-12:
                v = v / norm

                # Verify orthogonality: check overlaps with all previous vectors
                max_overlap = 0.0
                for j, k in enumerate(orthogonal_ops):
                    overlap = abs(np.trace(k.conj().T @ v) / d)
                    max_overlap = max(max_overlap, overlap)

                # Accept only if truly orthogonal
                if max_overlap < 1e-10:
                    orthogonal_ops.append(v)
                    print(f"✓ Kept operator {i} (norm={norm:.3e}, max_overlap={max_overlap:.3e})")
                else:
                    print(f"✗ Rejected operator {i}: max_overlap={max_overlap:.3e} > 1e-10")
            else:
                print(f"✗ Dropped operator {i}: norm={norm:.3e} too small")

        return orthogonal_ops

    def debug_modified_gram_schmidt(self, operators):
        d = operators[0].shape[0]
        orthogonal_ops = []

        for i, op in enumerate(operators):
            print(f"\n--- Processing operator {i} ---")
            v = op.copy()

            for j, k in enumerate(orthogonal_ops):
                # Check k is normalized
                k_norm_sq = np.trace(k.conj().T @ k) / d
                print(f"  k[{j}] norm²/d = {k_norm_sq:.6f}")

                proj = np.trace(k.conj().T @ v) / d
                print(f"  proj onto k[{j}] = {proj:.6f}")
                v = v - proj * k

            norm = np.sqrt(np.real(np.trace(v.conj().T @ v)) / d)
            print(f"  Final norm/√d = {norm:.6f}")

            if norm > 1e-12:
                v = v / norm
                orthogonal_ops.append(v)
            else:
                print(f"  DROPPED: norm too small")

        return orthogonal_ops

    # def gram_schmidt(self, operators):
    #     orthogonal_ops = []
    #     for op in operators:
    #         ortho_op = op.copy()
    #         for prev in orthogonal_ops:
    #             # Subtract projection using Hilbert-Schmidt inner product
    #             proj = np.trace(prev.conj().T @ ortho_op) / np.trace(prev.conj().T @ prev)
    #             ortho_op = ortho_op - proj * prev
    #         norm = np.sqrt(np.real(np.trace(ortho_op.conj().T @ ortho_op)))
    #         if norm > 1e-12:
    #             ortho_op = ortho_op / norm
    #             orthogonal_ops.append(ortho_op)
    #     return orthogonal_ops
    def gram_schmidt(self, operators):
        d = operators[0].shape[0]
        orthogonal_ops = []
        for op in operators:
            v = op.copy()
            for k in orthogonal_ops:
                proj = np.trace(k.conj().T @ v) / d
                v = v - proj * k
            # Normalize
            norm = np.sqrt(np.real(np.trace(v.conj().T @ v)) / d)
            if norm > 1e-12:
                v = v / norm
                orthogonal_ops.append(v)
        return orthogonal_ops

    def updated_express_in_orthogonal_basis(self, operators, orthogonal_operators):
        d = operators[0].shape[0]
        coefficients_list = []
        for operator in operators:
            coeff_per_op = []
            for Kj in orthogonal_operators:
                # Use Hilbert-Schmidt inner product and normalization
                coeff = np.trace(Kj.conj().T @ operator) / d
                coeff_per_op.append(coeff)
            coefficients_list.append(coeff_per_op)
        return coefficients_list



    def gram_schmidt_orthogonalization(self, operators: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """
        Perform Gram-Schmidt orthogonalization on a list of operators.

        Args:
            operators: List of operator matrices

        Returns:
            Tuple of (orthogonal_operators, transformation_matrix, coefficients_matrix)
        """
        orthogonal_ops = [operators[0]]
        #Normalize initial op
        K0 = (operators[0])/((np.trace(operators[0]*operators[0])))^0.5
        for j in range(1,len(operators)+1):
            overlaps_with_prev_basis=0
            for k in range(0,j):
                overlaps_with_prev_basis=overlaps_with_prev_basis+np.trace(operators[j] * orthogonal_ops[k])
            residue_operator = operators[j]-overlaps_with_prev_basis
            #Normalize new op
            orthogonal_ops.append((residue_operator)/((np.trace(residue_operator * residue_operator)))^0.5)









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

    def test_express_without_d(self, operators, orthogonal_operators):
        """Test version: remove /d from coefficient calculation"""
        coefficients_list = []
        for operator in operators:
            coeff_per_op = []
            for Kj in orthogonal_operators:
                # Remove /d here since orthogonal_operators are already normalized with /d
                coeff = np.trace(Kj.conj().T @ operator)  # NO /d
                coeff_per_op.append(coeff)
            coefficients_list.append(coeff_per_op)
        return coefficients_list

    # def updated_express_in_orthogonal_basis(self, operators, orthogonal_operators):
    #     d = operators[0].shape[0]
    #     coefficients_list = []
    #     for operator in operators:
    #         coeff_per_op = []
    #         for Kj in orthogonal_operators:
    #             coeff_per_op.append(np.trace(Kj.conj().T @ operator) / d)
    #         coefficients_list.append(coeff_per_op)
    #     return coefficients_list


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

# 1. Print Gram matrix of orthogonal basis
def print_gram_matrix(orthogonal_ops):
    d = orthogonal_ops[0].shape[0]
    n = len(orthogonal_ops)
    gram = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            gram[i, j] = np.trace(orthogonal_ops[i].conj().T @ orthogonal_ops[j]) / d
    print("Gram matrix of orthogonal basis (should be close to identity):")
    print(np.round(gram, 4))

# 2. Print norm of each basis operator
def print_basis_norms(orthogonal_ops):
    d = orthogonal_ops[0].shape[0]
    print("Norms of orthogonal basis operators (should be 1):")
    for idx, op in enumerate(orthogonal_ops):
        norm = np.sqrt(np.real(np.trace(op.conj().T @ op)) / d)
        print(f"Operator {idx}: norm = {norm:.6f}")

# 3. Check unitarity of time-evolution operator U
def check_unitarity(U):
    is_unitary = np.allclose(U.conj().T @ U, np.eye(U.shape[0]))
    print(f"Time-evolution operator is unitary: {is_unitary}")

# 4. Check for accidental overwriting/truncation during evolution
def print_evolved_operators_info(evolved_ops):
    for t, op in enumerate(evolved_ops):
        print(f"Step {t}: type={type(op)}, shape={op.shape}")

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
    #orthogonal_ops, transform_matrix, coefficients_matrix = orthogonalizer.gram_schmidt_orthogonalization(evolved_operators)
    independent_ops = orthogonalizer.filter_linearly_independent(evolved_operators)
    # Then perform Gram-Schmidt on the filtered set
    orthogonal_ops = orthogonalizer.robust_modified_gram_schmidt(evolved_operators)
    print_gram_matrix(orthogonal_ops)
    print_basis_norms(orthogonal_ops)
    # Express original operators in orthogonal basis
    coefficients = orthogonalizer.updated_express_in_orthogonal_basis(
        evolved_operators, orthogonal_ops
    )

    return {
        'orthogonal_operators': orthogonal_ops,
        #'transformation_matrix': transform_matrix,
        'coefficients_in_orthogonal_basis': coefficients,
        #'gs_in_original_coefficients': coefficients_matrix  # key added to invert coeff
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


def embed_two_qubit_gate(gate, i, j, n):
    # Permute qubits so i,j -> 0,1
    qubit_order = [i, j] + [q for q in range(n) if q != i and q != j]
    inv_order = np.argsort(qubit_order)
    dim = 2 ** n
    # Permutation matrix
    def permute_basis(order):
        perm = np.zeros((dim, dim), dtype=complex)
        for k in range(dim):
            bits = [(k >> l) & 1 for l in range(n)]
            permuted = [bits[order[m]] for m in range(n)]
            idx = sum([b << l for l, b in enumerate(permuted)])
            perm[idx, k] = 1
        return perm
    P = permute_basis(qubit_order)
    P_inv = permute_basis(inv_order)
    # Gate acts on first two qubits
    op = np.kron(gate, np.eye(2 ** (n - 2), dtype=complex))
    return P_inv @ op @ P

def create_general_circuit_unitaries(n_qubits, circuit_structure, theta):
    pswap = create_partial_swap_gate(theta)
    unitaries = []
    for pairs in circuit_structure:
        U = np.eye(2 ** n_qubits, dtype=complex)
        for pair in pairs:
            gate_full = embed_two_qubit_gate(pswap, pair[0], pair[1], n_qubits)
            U = gate_full @ U
        unitaries.append(U)
    return unitaries

def generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta):
    # Define two patterns for nearest-neighbor pairs
    pattern_dict = {
        'j': [(i, i+1) for i in range(0, n_qubits-1, 2)],  # even pairs: (0,1), (2,3), ...
        'g': [(i, i+1) for i in range(1, n_qubits-1, 2)]   # odd pairs: (1,2), (3,4), ...
    }
    pswap = create_partial_swap_gate(theta)
    unitaries = []
    for letter in pattern_string:
        pairs = pattern_dict[letter]
        U = np.eye(2 ** n_qubits, dtype=complex)
        for i, j in pairs:
            gate_full = embed_two_qubit_gate(pswap, i, j, n_qubits)
            U = gate_full @ U
        unitaries.append(U)
    return unitaries


import random

def generate_markov_chain_non_markovian_string(length, transition_matrix=None):
    """
    Generate a non-Markovian string using a Markov chain with memory.

    Args:
        length (int): Length of the string to generate.
        transition_matrix (dict): Transition probabilities based on last two characters.

    Returns:
        str: Generated non-Markovian string.
    """
    if transition_matrix is None:
        transition_matrix = {
            ('j', 'j'): {'j': 0.05, 'g': 0.95},
            ('j', 'g'): {'j': 0.1, 'g': 0.9},
            ('g', 'j'): {'j': 0.02, 'g': 0.98},
            ('g', 'g'): {'j': 0.02, 'g': 0.98},
        }

    result = ['j', 'g']  # Start with initial characters
    for _ in range(length - 2):
        last_two = tuple(result[-2:])
        probabilities = transition_matrix.get(last_two, {'j': 0.5, 'g': 0.5})
        next_char = random.choices(list(probabilities.keys()), weights=list(probabilities.values()))[0]
        result.append(next_char)

    return ''.join(result)


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
        print(f"U1 analysis completed")
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

def plot_final_weight_barchart(results):
    """
    Plot a histogram of nonzero weights at the last time step.

    Args:
        results: Output from analyze_with_z2_symmetry() or similar
        bins: Number of histogram bins
    """
    final_weights = np.array(results['weights_per_time'][-1])
    total_weight = np.sum(results['weights_per_time'], axis=0)
    non_zero_paulis=[]
    weight_0f_non_zero_pauli=[]
    for i in range(len(total_weight)):
        if total_weight[i] > 0:
            non_zero_paulis.append(i)
            weight_0f_non_zero_pauli.append(total_weight[i])
    plt.figure(figsize=(7, 4))
    plt.plot(non_zero_paulis, weight_0f_non_zero_pauli, color='navy', alpha=0.7)
    plt.xlabel('Pauli basis')
    plt.ylabel('Weight')
    plt.title('Distribution of weights on Pauli basis')
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

def plot_total_coefficient_sqaure(gs_results):
    coefficients_per_time = gs_results['coefficients_in_orthogonal_basis']
    coeff_squared = []
    time_steps_range = []
    for time_steps in range(1, len(coefficients_per_time)):
        square_val=0
        for coeff in coefficients_per_time[time_steps]:
            square_val=square_val+coeff**2
        coeff_squared.append(coeff_squared)
        time_steps_range.append(time_steps)
    plt.figure(figsize=(7, 4))
    plt.plot(time_steps_range, coeff_squared, color='navy', alpha=0.7)
    plt.xlabel('Pauli basis')
    plt.ylabel('Weight')
    plt.title('Distribution of weights on Pauli basis')
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

    # Top right
    ax2 = axes[0, 1]
    n_indices = np.arange(n_gs_operators)
    # Print the actual sum of squares values
    total_weights = np.sum(weights_gs_basis, axis=1)
    print("Sum of squares values:")
    for i, val in enumerate(total_weights):
        print(f"t={i}: {val:.15f}")

    print(f"\nDeviation from 1.0:")
    for i, val in enumerate(total_weights):
        deviation = val - 1.0
        print(f"t={i}: {deviation:.2e}")
    tot_sum = np.sum(weights_gs_basis, axis=1)
    ax2.plot(time_steps, tot_sum, 'o-', linewidth=2, color='purple')
    ax2.set_xlabel('Time Step t')
    ax2.set_ylabel(r'$\sum_n |a_n(t)|^2$')
    ax2.set_title(r'Weighted GS Index: $\sum_n|a_n(t)|^2$')
    ax2.grid(True, alpha=0.3)

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

def check_operator_norms(operators):
    d = operators[0].shape[0]
    for i, op in enumerate(operators):
        norm_sq = np.real(np.trace(op.conj().T @ op)) / d
        print(f"Operator {i}: ||O_{i}||²/d = {norm_sq:.6f}")


plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa'
})

# Beautiful pastel color palette
PASTEL_COLORS = [
    '#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF',
    '#E1BAFF', '#FFBAE1', '#C9FFBA', '#BAD4FF', '#FFDCBA'
]

# Create custom pastel colormaps
pastel_viridis = LinearSegmentedColormap.from_list('pastel_viridis',
                                                   ['#E8F4F8', '#B8E6B8', '#FFE5B4', '#FFCCCB'])
pastel_hot = LinearSegmentedColormap.from_list('pastel_hot',
                                               ['#F0F8FF', '#FFE4E1', '#FFB6C1', '#DDA0DD'])

# CONSISTENT NUMERICAL PRECISION THRESHOLD
WEIGHT_THRESHOLD = 1e-30  # Use same threshold across all functions


def plot_final_weight_histogram(results, bins=40, figsize=(8, 5)):
    """Enhanced histogram with better styling and log-normal fit overlay."""
    final_weights = np.array(results['weights_per_time'][-1])
    nonzero_weights = final_weights[final_weights > WEIGHT_THRESHOLD]  # CONSISTENT THRESHOLD

    fig, ax = plt.subplots(figsize=figsize)

    # Create histogram with beautiful styling
    n, bins_edges, patches = ax.hist(nonzero_weights, bins=bins, log=True,
                                     color='#BAE1FF', alpha=0.7, edgecolor='#7BB3D3', linewidth=0.5)

    # Color gradient for histogram bars
    for i, p in enumerate(patches):
        p.set_facecolor(plt.cm.Pastel1(i / len(patches)))

    ax.set_xlabel('Weight', fontweight='bold')
    ax.set_ylabel('Count (log scale)', fontweight='bold')
    ax.set_title('Distribution of Nonzero Weights at Final Time Step',
                 fontweight='bold', pad=20)

    # Add statistics text box
    mean_weight = np.mean(nonzero_weights)
    std_weight = np.std(nonzero_weights)
    textstr = f'Non-zero operators: {len(nonzero_weights)}\nMean weight: {mean_weight:.2e}\nStd: {std_weight:.2e}'
    props = dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()


def plot_pauli_weight_distribution(results, top_n=50, figsize=(12, 6)):
    """Enhanced bar chart showing weight distribution across Pauli basis."""
    final_weights = np.array(results['weights_per_time'][-1])
    total_weight = np.sum(results['weights_per_time'], axis=0)

    # Get top operators by total weight
    top_indices = np.argsort(total_weight)[-top_n:][::-1]
    top_weights = total_weight[top_indices]
    top_labels = [results['pauli_basis'][i] for i in top_indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Bar chart of top operators
    bars = ax1.bar(range(len(top_weights)), top_weights,
                   color=PASTEL_COLORS[:len(top_weights)], alpha=0.8,
                   edgecolor='gray', linewidth=0.5)

    ax1.set_xlabel('Pauli Operator Rank', fontweight='bold')
    ax1.set_ylabel('Total Weight', fontweight='bold')
    ax1.set_title(f'Top {top_n} Pauli Operators by Total Weight', fontweight='bold')
    ax1.set_yscale('log')

    # Add value labels on bars for top 10
    for i, (bar, weight) in enumerate(zip(bars[:10], top_weights[:10])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{weight:.1e}', ha='center', va='bottom', fontsize=8, rotation=45)

    # Right: Weight decay plot
    sorted_weights = np.sort(total_weight[total_weight > WEIGHT_THRESHOLD])[::-1]  # CONSISTENT THRESHOLD
    ax2.semilogy(range(len(sorted_weights)), sorted_weights,
                 'o-', color='#FF9999', markersize=3, linewidth=2, alpha=0.8)
    ax2.set_xlabel('Operator Index (sorted)', fontweight='bold')
    ax2.set_ylabel('Weight (log scale)', fontweight='bold')
    ax2.set_title('Weight Decay Profile', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_operator_spreading_evolution(results, figsize=(10, 6)):
    """Enhanced plot showing operator spreading over time."""
    weights_per_time = results['weights_per_time']
    n_time_steps = len(weights_per_time)

    # Calculate metrics
    nonzero_counts = [np.sum(np.array(w) > WEIGHT_THRESHOLD) for w in weights_per_time]  # CONSISTENT THRESHOLD
    entropy = []
    participation_ratio = []

    for weights in weights_per_time:
        w = np.array(weights)
        w_norm = w[w > WEIGHT_THRESHOLD] / np.sum(w[w > WEIGHT_THRESHOLD])  # CONSISTENT THRESHOLD

        # Shannon entropy
        entropy.append(-np.sum(w_norm * np.log(w_norm + 1e-16)))

        # Participation ratio (inverse participation ratio)
        participation_ratio.append(1 / np.sum(w_norm ** 2) if len(w_norm) > 0 else 0)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Operator Spreading Dynamics', fontsize=16, fontweight='bold')

    # Top left: Number of explored operators
    axes[0, 0].plot(range(n_time_steps), nonzero_counts, 'o-',
                    color='#FFB3BA', linewidth=3, markersize=6, alpha=0.8)
    axes[0, 0].set_ylabel('Active Operators', fontweight='bold')
    axes[0, 0].set_title('Operator Space Exploration', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Shannon entropy
    axes[0, 1].plot(range(n_time_steps), entropy, 's-',
                    color='#BAFFC9', linewidth=3, markersize=6, alpha=0.8)
    axes[0, 1].set_ylabel('Shannon Entropy', fontweight='bold')
    axes[0, 1].set_title('Weight Distribution Entropy', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom left: Participation ratio
    axes[1, 0].plot(range(n_time_steps), participation_ratio, '^-',
                    color='#BAE1FF', linewidth=3, markersize=6, alpha=0.8)
    axes[1, 0].set_xlabel('Time Step', fontweight='bold')
    axes[1, 0].set_ylabel('Participation Ratio', fontweight='bold')
    axes[1, 0].set_title('Effective Dimensionality', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom right: Weight concentration (top 10%)
    top_10_percent = []
    for weights in weights_per_time:
        w = np.array(weights)
        n_top = max(1, len(w) // 10)
        top_weights = np.sort(w)[-n_top:]
        top_10_percent.append(np.sum(top_weights) / np.sum(w))

    axes[1, 1].plot(range(n_time_steps), top_10_percent, 'D-',
                    color='#E1BAFF', linewidth=3, markersize=6, alpha=0.8)
    axes[1, 1].set_xlabel('Time Step', fontweight='bold')
    axes[1, 1].set_ylabel('Weight in Top 10%', fontweight='bold')
    axes[1, 1].set_title('Weight Concentration', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_top_operators_evolution(results, top_n=15, figsize=(12, 8)):
    """Beautiful evolution plot for top operators with enhanced styling."""
    weights_per_time = results['weights_per_time']
    pauli_basis = results['pauli_basis']
    n_time_steps = len(weights_per_time)

    # Find top operators by maximum weight
    all_max_weights = []
    for i, pauli_string in enumerate(pauli_basis):
        max_weight = max(weights_per_time[t][i] for t in range(n_time_steps))
        all_max_weights.append((max_weight, i, pauli_string))

    all_max_weights.sort(reverse=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Top plot: Linear scale with top operators
    for rank, (_, idx, pauli_string) in enumerate(all_max_weights[:top_n]):
        weights_over_time = [weights_per_time[t][idx] for t in range(n_time_steps)]
        color = PASTEL_COLORS[rank % len(PASTEL_COLORS)]

        ax1.plot(range(n_time_steps), weights_over_time, 'o-',
                 linewidth=2.5, color=color, label=f'{pauli_string}',
                 markersize=5, alpha=0.8)

    ax1.set_ylabel('Weight', fontweight='bold')
    ax1.set_title(f'Top {top_n} Operators: Weight Evolution (Linear Scale)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    # Bottom plot: Log scale
    for rank, (_, idx, pauli_string) in enumerate(all_max_weights[:top_n]):
        weights_over_time = [max(weights_per_time[t][idx], WEIGHT_THRESHOLD) for t in range(n_time_steps)]  # CONSISTENT THRESHOLD
        color = PASTEL_COLORS[rank % len(PASTEL_COLORS)]

        ax2.semilogy(range(n_time_steps), weights_over_time, 's-',
                     linewidth=2.5, color=color, markersize=4, alpha=0.8)

    ax2.set_xlabel('Time Step', fontweight='bold')
    ax2.set_ylabel('Weight (log scale)', fontweight='bold')
    ax2.set_title('Weight Evolution (Log Scale)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_pauli_heatmap(coefficients_matrix, pauli_basis, max_operators=50, figsize=(14, 8)):
    """Enhanced heatmap with better color scheme and annotations."""
    coefficients_magnitude = np.abs(coefficients_matrix)

    # Select top operators for better visualization
    max_weights = np.max(coefficients_magnitude, axis=0)
    top_indices = np.argsort(max_weights)[-max_operators:][::-1]

    selected_coeffs = coefficients_magnitude[:, top_indices]
    selected_labels = [pauli_basis[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap with custom colormap
    im = ax.imshow(selected_coeffs.T, aspect='auto', cmap=pastel_viridis,
                   origin='lower', interpolation='nearest')

    # Colorbar with better formatting
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('|Coefficient|', fontweight='bold', fontsize=12)

    ax.set_xlabel('Time Step', fontweight='bold', fontsize=12)
    ax.set_ylabel('Pauli Operator', fontweight='bold', fontsize=12)
    ax.set_title('Pauli String Coefficients Evolution', fontweight='bold', fontsize=14, pad=20)

    # Set ticks
    time_steps = coefficients_magnitude.shape[0]
    ax.set_xticks(range(0, time_steps, max(1, time_steps // 10)))
    ax.set_yticks(range(len(selected_labels)))
    ax.set_yticklabels(selected_labels, fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_gs_basis_evolution_enhanced(results, gs_results, figsize=(14, 10)):
    """Enhanced Gram-Schmidt evolution plot with beautiful styling."""
    coefficients_per_time = gs_results['coefficients_in_orthogonal_basis']
    n_time_steps = len(coefficients_per_time)
    n_gs_operators = len(coefficients_per_time[0]) if coefficients_per_time else 0

    weights_gs_basis = np.abs(np.array(coefficients_per_time)) ** 2
    time_steps = range(n_time_steps)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Gram-Schmidt Basis Evolution Analysis', fontsize=16, fontweight='bold')

    # Top left: Weighted GS index
    n_indices = np.arange(n_gs_operators)
    weighted_sum = np.sum(n_indices * weights_gs_basis, axis=1)
    axes[0, 0].plot(time_steps, weighted_sum, 'o-', linewidth=3,
                    color='#FF9999', markersize=6, alpha=0.8)
    axes[0, 0].set_xlabel('Time Step', fontweight='bold')
    axes[0, 0].set_ylabel(r'$\sum_n n|a_n(t)|^2$', fontweight='bold')
    axes[0, 0].set_title('Weighted GS Index', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Norm conservation
    total_weights = np.sum(weights_gs_basis, axis=1)
    deviation_from_one = total_weights - 1.0

    axes[0, 1].plot(time_steps, deviation_from_one * 1e12, 'o-', linewidth=3,
                    color='#99FF99', markersize=6, alpha=0.8)
    axes[0, 1].set_xlabel('Time Step', fontweight='bold')
    axes[0, 1].set_ylabel('Deviation × 10¹² ', fontweight='bold')
    axes[0, 1].set_title('Norm Conservation (×10¹² precision)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Bottom left: Enhanced heatmap
    im = axes[1, 0].imshow(weights_gs_basis.T, aspect='auto', cmap=pastel_hot,
                           origin='lower', interpolation='bilinear')
    axes[1, 0].set_xlabel('Time Step', fontweight='bold')
    axes[1, 0].set_ylabel('GS Operator Index', fontweight='bold')
    axes[1, 0].set_title('GS Basis Weight Heatmap', fontweight='bold')
    plt.colorbar(im, ax=axes[1, 0], label='|aₙ(t)|²', shrink=0.8)

    # Bottom right: Cumulative weight distribution
    axes[1, 1].plot(time_steps, total_weights, 'o-', linewidth=3,
                    markersize=6, color='#9999FF', label='Total', alpha=0.9)

    colors = ['#FFB3BA', '#FFDFBA', '#BAFFC9', '#BAE1FF', '#E1BAFF']
    for i, k in enumerate([1, 2, 3, 5, 8]):
        if k <= n_gs_operators:
            cum_weight = np.sum(weights_gs_basis[:, :k], axis=1)
            axes[1, 1].plot(time_steps, cum_weight, '--', linewidth=2.5,
                            color=colors[i % len(colors)], alpha=0.8,
                            label=f'First {k} operators')

    axes[1, 1].set_xlabel('Time Step', fontweight='bold')
    axes[1, 1].set_ylabel('Cumulative Weight', fontweight='bold')
    axes[1, 1].set_title('Weight Concentration Analysis', fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_final_weight_barchart(results, figsize=(15, 6), max_labels=50):
    """
    Enhanced barchart with automatic neat spacing for Pauli operator strings.
    """
    final_weights = np.array(results['weights_per_time'][-1])
    total_weight = np.sum(results['weights_per_time'], axis=0)
    pauli_basis = results['pauli_basis']

    # Find non-zero operators and sort by weight (descending)
    non_zero_data = []
    for i in range(len(total_weight)):
        if total_weight[i] > WEIGHT_THRESHOLD:  # CONSISTENT THRESHOLD
            non_zero_data.append((total_weight[i], pauli_basis[i]))

    # Sort by weight (highest first) for better visualization
    non_zero_data.sort(reverse=True)
    weights = [w for w, _ in non_zero_data]
    labels = [label for _, label in non_zero_data]

    # If too many operators, show top ones + indicate truncation
    if len(labels) > max_labels:
        weights = weights[:max_labels]
        labels = labels[:max_labels]
        truncated = True
    else:
        truncated = False

    fig, ax = plt.subplots(figsize=figsize)

    # Create evenly spaced x positions
    x_positions = np.arange(len(weights))

    # Bar plot instead of line plot for better categorical visualization
    bars = ax.bar(x_positions, weights,
                  color='#BAE1FF', alpha=0.8,
                  edgecolor='#7BB3D3', linewidth=1)

    # Color gradient for bars (highest to lowest)
    for i, bar in enumerate(bars):
        # Gradient from strong to light color
        intensity = 1 - (i / len(bars)) * 0.7  # From 1.0 to 0.3
        bar.set_facecolor(plt.cm.Pastel1(intensity))

    # Set Pauli string labels with automatic spacing
    ax.set_xticks(x_positions)

    # Calculate optimal rotation based on number of labels
    if len(labels) <= 10:
        rotation = 0
        ha = 'center'
        fontsize = 10
    elif len(labels) <= 25:
        rotation = 45
        ha = 'right'
        fontsize = 9
    else:
        rotation = 90
        ha = 'right'
        fontsize = 8

    ax.set_xticklabels(labels, rotation=rotation, ha=ha, fontsize=fontsize)

    ax.set_xlabel('Pauli Operator (sorted by weight)', fontweight='bold')
    ax.set_ylabel('Total Weight', fontweight='bold')
    ax.set_title('Distribution of Weights on Pauli Basis', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')  # Only horizontal grid lines
    ax.set_yscale('log')

    # Add value labels on top of bars for top 5
    for i, (bar, weight) in enumerate(zip(bars[:5], weights[:5])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height * 1.1,
                f'{weight:.2e}', ha='center', va='bottom', fontsize=8,
                fontweight='bold', color='darkblue')

    # Enhanced statistics
    total_operators = len(total_weight)
    active_operators = len([w for w in total_weight if w > WEIGHT_THRESHOLD])  # CONSISTENT THRESHOLD

    textstr = f'Active operators: {active_operators}/{total_operators}'
    if truncated:
        textstr += f'\nShowing top {max_labels}'
    textstr += f'\nMax weight: {max(weights):.2e}'
    textstr += f'\nMin weight: {min(weights):.2e}'

    props = dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()


def plot_cumulative_operator_exploration(results, figsize=(12, 8)):
    """
    Plot cumulative exploration of distinct Pauli operators over time.
    Shows both instantaneous and cumulative operator counts.
    """
    weights_per_time = results['weights_per_time']
    n_time_steps = len(weights_per_time)

    # Track cumulative distinct operators
    explored_operators = set()
    cumulative_counts = []
    instantaneous_counts = []

    for t, weights in enumerate(weights_per_time):
        # Count active operators at this time step
        active_at_t = set(i for i, w in enumerate(weights) if w > WEIGHT_THRESHOLD)  # CONSISTENT THRESHOLD
        instantaneous_counts.append(len(active_at_t))

        # Add to cumulative set
        explored_operators.update(active_at_t)
        cumulative_counts.append(len(explored_operators))

    # Calculate exploration rate (new operators per step)
    exploration_rate = [0]  # First step
    for t in range(1, n_time_steps):
        new_ops = cumulative_counts[t] - cumulative_counts[t - 1]
        exploration_rate.append(new_ops)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Pauli Operator Space Exploration Dynamics', fontsize=16, fontweight='bold')

    # Top left: Cumulative vs instantaneous
    axes[0, 0].plot(range(n_time_steps), cumulative_counts, 'o-',
                    linewidth=3, color='#FF9999', markersize=5, alpha=0.8,
                    label='Cumulative distinct')
    axes[0, 0].plot(range(n_time_steps), instantaneous_counts, 's-',
                    linewidth=3, color='#9999FF', markersize=5, alpha=0.8,
                    label='Active at time t')
    axes[0, 0].set_ylabel('Number of Operators', fontweight='bold')
    axes[0, 0].set_title('Cumulative vs Instantaneous Exploration', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Exploration rate
    axes[0, 1].bar(range(n_time_steps), exploration_rate,
                   color='#BAFFC9', alpha=0.8, edgecolor='#7BB3D3', linewidth=0.5)
    axes[0, 1].set_ylabel('New Operators Discovered', fontweight='bold')
    axes[0, 1].set_title('Rate of New Operator Discovery', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Bottom left: Exploration efficiency
    exploration_efficiency = [c / max(1, t + 1) for t, c in enumerate(cumulative_counts)]
    axes[1, 0].plot(range(n_time_steps), exploration_efficiency, '^-',
                    linewidth=3, color='#FFB3BA', markersize=5, alpha=0.8)
    axes[1, 0].set_xlabel('Time Step', fontweight='bold')
    axes[1, 0].set_ylabel('Operators per Time Step', fontweight='bold')
    axes[1, 0].set_title('Exploration Efficiency', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom right: Exploration phases
    # Detect different exploration phases
    phases = []
    current_phase = "Initial"

    for t in range(1, n_time_steps):
        rate = exploration_rate[t]
        if rate > 2:
            current_phase = "Rapid"
        elif rate > 0:
            current_phase = "Gradual"
        else:
            current_phase = "Saturated"
        phases.append(current_phase)

    # Color code the cumulative plot by phase
    phase_colors = {"Initial": '#E1BAFF', "Rapid": '#FFB3BA',
                    "Gradual": '#FFFFBA', "Saturated": '#BAE1FF'}

    for t in range(n_time_steps - 1):
        if t < len(phases):
            color = phase_colors.get(phases[t], '#CCCCCC')
            axes[1, 1].plot([t, t + 1],
                            [cumulative_counts[t], cumulative_counts[t + 1]],
                            'o-', linewidth=4, color=color, markersize=4, alpha=0.8)

    axes[1, 1].set_xlabel('Time Step', fontweight='bold')
    axes[1, 1].set_ylabel('Cumulative Operators', fontweight='bold')
    axes[1, 1].set_title('Exploration Phases', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    # Add legend for phases
    phase_handles = [plt.Line2D([0], [0], color=color, linewidth=3, alpha=0.8)
                     for color in phase_colors.values()]
    axes[1, 1].legend(phase_handles, phase_colors.keys(), fontsize=9)

    # Print summary statistics
    total_possible = len(weights_per_time[0])
    final_explored = cumulative_counts[-1]
    max_instantaneous = max(instantaneous_counts)

    print(f"\n📊 Exploration Summary:")
    print(f"Total operators explored: {final_explored}/{total_possible} ({100 * final_explored / total_possible:.1f}%)")
    print(f"Maximum instantaneous: {max_instantaneous}")
    print(f"Final instantaneous: {instantaneous_counts[-1]}")
    print(f"Exploration efficiency: {final_explored / n_time_steps:.1f} operators/step")

    plt.tight_layout()
    plt.show()

    return {
        'cumulative_counts': cumulative_counts,
        'instantaneous_counts': instantaneous_counts,
        'exploration_rate': exploration_rate,
        'total_explored': final_explored,
        'total_possible': total_possible
    }

# Main execution
if __name__ == "__main__":
    theta_try = np.pi/15
    # Run the main analysis
    #unitaries = create_brickwork_unitaries(8,theta_try)
    # Step 1: Get your spreading results first
    #results = analyze_with_u1_symmetry(8, "ZIIIIIII", unitaries, 2, verbose=True)
    #analyzer = QuantumOperatorAnalyzer(8, symmetry='U1')
    #gs_results = orthogonalize_evolved_operators(results['evolved_operators'], analyzer)
    #plot_gs_basis_evolution(results, gs_results)
    #plot_nonzero_operator_count(results)
    #quick_plot(results)
    #plot_final_weight_histogram(results)

    # Example usage:
    #circuit_structure = [
    #     [[0,1],[2,3],[4,5],[6,7]],  # time step 1
    #     [[1,4],[3,2],[5,6]],        # time step 2
    #     [[0,7]]                    # time step 3 (non-nearest neighbor)
    # ]
    #unitaries = create_general_circuit_unitaries(8, circuit_structure, theta=np.pi/15)
    #results = analyze_with_u1_symmetry(8, "ZIIIIIII", unitaries, 2, verbose=True)
    ##analyzer = QuantumOperatorAnalyzer(8, symmetry='U1')
    #gs_results = orthogonalize_evolved_operators(results['evolved_operators'], analyzer)
    #plot_gs_basis_evolution(results, gs_results)
    #plot_nonzero_operator_count(results)
    #quick_plot(results)
    #plot_final_weight_histogram(results)
    # Example usage:
    #n_qubits = 8
    #pattern_string = "jgjgjggggggggggggggggggggggggggggggg"
    #theta = np.pi / 15
    #unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
    #print(f"Generated {len(unitaries)} unitaries for pattern: {pattern_string}")
    #results = analyze_with_u1_symmetry(8, "ZIIIIIII", unitaries, 20, verbose=True)
    #analyzer = QuantumOperatorAnalyzer(8, symmetry='U1')
    #gs_results = orthogonalize_evolved_operators(results['evolved_operators'], analyzer)
    #plot_gs_basis_evolution(results, gs_results)
    #plot_nonzero_operator_count(results)
    #quick_plot(results)
    #plot_final_weight_histogram(results)


    n_qubits = 8
    #pattern_string = "jjjgjjjg"
    theta = np.pi /15
    #unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
    #print(f"Generated {len(unitaries)} unitaries for pattern: {pattern_string}")
    #for i in range(10):
    #    pattern_string = generate_markov_chain_non_markovian_string(100)
    #    print(f"Generated pattern string: {pattern_string}")
    #unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
    #results = analyze_with_u1_symmetry(8, "ZIIIIIII", unitaries, 20, verbose=True)
    #analyzer = QuantumOperatorAnalyzer(8, symmetry='U1')
    #gs_results = orthogonalize_evolved_operators(results['evolved_operators'], analyzer)
    #plot_gs_basis_evolution(results, gs_results)
    #plot_nonzero_operator_count(results)
    #quick_plot(results)
    #plot_final_weight_histogram(results)

    for rule in ["rule_0_0_2_2","rule_3_3_1_1","rule_1_0_1_3","rule_0_2_2_1"]:
    #for rule in ["rule_0_0_2_2"]:

        n_qubits = 8
        with open(f'../non_markovian_orders_list/{rule}.csv', 'r') as f:
            reader = csv.reader(f)
            NM_list = [row[0].strip() for row in reader if row and row[0].strip()]

        len(NM_list[1])
        pattern_string = NM_list[1]
        print(pattern_string)
        theta = np.pi / 15
        unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
        print(f"Generated {len(unitaries)} unitaries for pattern: {pattern_string}")
        results = analyze_with_u1_symmetry(8, "ZIIIIIII", unitaries, 100, verbose=True)
        plot_final_weight_barchart(results)
        analyzer = QuantumOperatorAnalyzer(8, symmetry='U1')
        print(results['evolved_operators'])
        print(type(results['evolved_operators']))
        print(len(results['evolved_operators']))
        check_operator_norms(results['evolved_operators'])

        print_evolved_operators_info(results['evolved_operators'])

        gs_results = orthogonalize_evolved_operators(results['evolved_operators'], analyzer)
        plot_gs_basis_evolution_enhanced(results, gs_results)
        plot_pauli_weight_distribution(results)
        plot_final_weight_histogram(results)
        plot_operator_spreading_evolution(results)
        plot_top_operators_evolution(results)
        plot_final_weight_barchart(results)
        plot_gs_basis_evolution_enhanced(results, gs_results)
        plot_cumulative_operator_exploration(results)

        #plot_nonzero_operator_count(results)
        #quick_plot(results)
        #plot_final_weight_histogram(results)


