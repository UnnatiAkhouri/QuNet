
import numpy as np
from scipy.linalg import expm
from itertools import product, combinations
from typing import List, Tuple, Dict, Optional
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
import numpy as np
from scipy.linalg import expm
from itertools import product, combinations
from typing import List, Tuple, Dict, Optional
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random

# Global constants
WEIGHT_THRESHOLD = 1e-30
PASTEL_COLORS = [
    '#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF',
    '#E1BAFF', '#FFBAE1', '#C9FFBA', '#BAD4FF', '#FFDCBA'
]
#New_pastel = ["#656a95", "#d9ead3", "#ffe599", "#a2c4c9", "#9fc5e8", "#b4a7d6"]

New_pastel = ["#b5e2da","#656a95","#db95a6", "#f7c59f", "#d9ead3", "#ffe599", "#a2c4c9", "#9fc5e8", "#b4a7d6"]


class QuantumOperatorAnalyzer:
    """Efficient analyzer for operator spreading in quantum circuits with corrected symmetries."""

    def __init__(self, n_qubits: int, symmetry: Optional[str] = None):
        self.n_qubits = n_qubits
        self.symmetry = symmetry

        # Determine which basis to use based on symmetry
        if symmetry == 'U1_sbasis':
            self.basis_type = 'S+S-'
            self.operator_matrices = self._get_spin_matrices()
        else:
            self.basis_type = 'IXYZ'
            self.operator_matrices = self._get_pauli_matrices()

        self.pauli_basis = self._generate_pauli_basis()

        # Only pre-compute for small systems to avoid memory issues
        if len(self.pauli_basis) < 10:
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

    def _get_spin_matrices(self) -> Dict[str, np.ndarray]:
        """Generate the spin operator matrices for S+/S- basis."""
        return {
            'I': np.array([[1, 0], [0, 1]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
            'S+': np.array([[0, 1], [0, 0]], dtype=complex),  # |1⟩⟨0|
            'S-': np.array([[0, 0], [1, 0]], dtype=complex)  # |0⟩⟨1|
        }

    def _generate_pauli_basis(self) -> List[str]:
        """Generate operator strings respecting symmetries."""
        if self.symmetry is None:
            return self._generate_full_basis()
        elif self.symmetry == 'U1':
            return self._generate_u1_basis()
        elif self.symmetry == 'U1_sbasis':
            return self._generate_u1_sbasis()
        else:
            raise ValueError(f"Unknown symmetry: {self.symmetry}")

    def _generate_full_basis(self) -> List[str]:
        """Generate all possible operator strings for N qubits."""
        if self.basis_type == 'IXYZ':
            operator_chars = ['I', 'X', 'Y', 'Z']
        else:  # S+S- basis
            operator_chars = ['I', 'Z', 'S+', 'S-']
        return [''.join(p) for p in product(operator_chars, repeat=self.n_qubits)]

    def _generate_u1_basis(self) -> List[str]:
        """Generate U(1)-symmetric operator strings (even X+Y count in IXYZ basis)."""
        if self.basis_type != 'IXYZ':
            raise ValueError("U1 symmetry (IXYZ) only applies to IXYZ basis")

        operator_chars = ['I', 'X', 'Y', 'Z']
        u1_strings = []

        for operator_tuple in product(operator_chars, repeat=self.n_qubits):
            xy_count = operator_tuple.count('X') + operator_tuple.count('Y')
            if xy_count % 2 == 0:  # U(1) constraint: even X+Y count
                u1_strings.append(''.join(operator_tuple))

        return u1_strings

    def _generate_u1_sbasis(self) -> List[str]:
        """Generate U(1)-symmetric operator strings in S+/S- basis (equal S+ and S- count)."""
        if self.basis_type != 'S+S-':
            raise ValueError("U1_sbasis only applies to S+/S- basis")

        u1_strings = []
        max_spin_pairs = self.n_qubits // 2

        for num_splus in range(max_spin_pairs + 1):
            num_sminus = num_splus  # U1 symmetry: equal S+ and S-
            num_iz = self.n_qubits - num_splus - num_sminus

            if num_iz < 0:
                continue

            # Choose positions for S+'s
            for splus_positions in combinations(range(self.n_qubits), num_splus):
                remaining_positions = [i for i in range(self.n_qubits) if i not in splus_positions]

                # Choose positions for S-'s from remaining positions
                for sminus_positions in combinations(remaining_positions, num_sminus):
                    iz_positions = [i for i in remaining_positions if i not in sminus_positions]

                    # For each way to assign I and Z to remaining positions
                    for num_i in range(len(iz_positions) + 1):
                        # Choose positions for I's from I/Z positions
                        for i_positions in combinations(iz_positions, num_i):
                            z_positions = [i for i in iz_positions if i not in i_positions]

                            # Construct the operator string
                            operator_string = ['I'] * self.n_qubits

                            for pos in splus_positions:
                                operator_string[pos] = 'S+'
                            for pos in sminus_positions:
                                operator_string[pos] = 'S-'
                            for pos in z_positions:
                                operator_string[pos] = 'Z'

                            u1_strings.append(''.join(operator_string))

        return u1_strings

    def _precompute_optimization_data(self):
        """Pre-compute data structures for optimization."""
        print(f"Pre-computing optimization data for {len(self.pauli_basis):,} operator strings...")
        start_time = time.time()

        dim = 2 ** self.n_qubits
        self.pauli_tensor = np.zeros((len(self.pauli_basis), dim, dim), dtype=complex)

        for i, operator_string in enumerate(self.pauli_basis):
            self.pauli_tensor[i] = self.operator_string_to_matrix(operator_string)

        print(f"Pre-computation completed in {time.time() - start_time:.2f}s")

    def operator_string_to_matrix(self, operator_string: str) -> np.ndarray:
        """Convert an operator string to its matrix representation.

        This method should work with ANY operator string regardless of the symmetry setting.
        The symmetry only affects the decomposition basis, not the allowed initial operators.
        """
        if len(operator_string) != self.n_qubits:
            raise ValueError(f"Operator string length must be {self.n_qubits}")

        # ALWAYS try IXYZ format first, regardless of self.basis_type
        # This ensures any standard Pauli string works with any symmetry
        if self._is_ixyz_string(operator_string):
            # Use standard Pauli matrices (always available)
            pauli_matrices = self._get_pauli_matrices()
            result = pauli_matrices[operator_string[0]]
            for op_char in operator_string[1:]:
                result = np.kron(result, pauli_matrices[op_char])
            return result

        # Only if it contains S+/S- operators, parse accordingly
        elif self._contains_spin_operators(operator_string):
            operators = self._parse_operator_string(operator_string)
            if len(operators) != self.n_qubits:
                raise ValueError(
                    f"Parsed {len(operators)} operators from '{operator_string}', but need {self.n_qubits}")

            # Use the appropriate operator matrices for this basis
            result = self.operator_matrices[operators[0]]
            for op in operators[1:]:
                result = np.kron(result, self.operator_matrices[op])
            return result

        else:
            raise ValueError(
                f"Unrecognized operator format: '{operator_string}'. Use IXYZ format like 'XIIIIII' or S+/S- format like 'S+S-IIII'")

    def _is_ixyz_string(self, operator_string: str) -> bool:
        """Check if string contains only I, X, Y, Z characters."""
        return all(char in 'IXYZ' for char in operator_string) and len(operator_string) > 0

    def _contains_spin_operators(self, operator_string: str) -> bool:
        """Check if string contains S+ or S- operators."""
        return 'S+' in operator_string or 'S-' in operator_string

    def _parse_operator_string(self, operator_string: str) -> List[str]:
        """Parse operator string handling multi-character operators like S+ and S-."""
        operators = []
        i = 0
        while i < len(operator_string):
            if i < len(operator_string) - 1 and operator_string[i:i + 2] in ['S+', 'S-']:
                operators.append(operator_string[i:i + 2])
                i += 2
            else:
                operators.append(operator_string[i])
                i += 1
        return operators

    def pauli_string_to_matrix(self, pauli_string: str) -> np.ndarray:
        """Legacy method name for backward compatibility."""
        return self.operator_string_to_matrix(pauli_string)

    def compute_overlap_vectorized(self, operator: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Vectorized overlap computation using Einstein summation."""
        if self.pauli_tensor is None:
            if verbose:
                print("Pre-computed tensor not available, using batched computation...")
            return self.compute_overlap_batched(operator, verbose=verbose)

        if verbose:
            print(f"Computing {len(self.pauli_basis):,} overlaps (vectorized)...")
            start_time = time.time()

        # Vectorized trace computation: trace(O @ P_i) for all i
        # Normalize by Hilbert space dimension
        d = operator.shape[0]
        overlaps = np.einsum('ij,kji->k', operator, self.pauli_tensor) / d

        if verbose:
            total_time = time.time() - start_time
            print(f"Vectorized computation completed in {total_time:.3f}s")

        return overlaps

    def compute_overlap_batched(self, operator: np.ndarray, batch_size: int = 1000,
                                verbose: bool = False) -> np.ndarray:
        """Batched computation for large systems."""
        if verbose:
            print(f"Computing {len(self.pauli_basis):,} overlaps (batched, size={batch_size})...")
            start_time = time.time()

        overlaps = np.zeros(len(self.pauli_basis), dtype=complex)
        n_batches = (len(self.pauli_basis) + batch_size - 1) // batch_size
        d = operator.shape[0]  # Hilbert space dimension

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.pauli_basis))

            if verbose and batch_idx % 5 == 0:
                progress = 100 * start_idx / len(self.pauli_basis)
                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx + 1}/{n_batches}: {progress:.1f}% ({elapsed:.1f}s)")

            # Compute overlaps for this batch
            for i in range(start_idx, end_idx):
                try:
                    # This is where the error occurs - creating matrices from basis strings
                    basis_operator_string = self.pauli_basis[i]

                    # Handle different basis types properly
                    if self.basis_type == 'S+S-':
                        # For S+/S- basis, we need to use the spin matrices
                        operator_matrix = self._create_spin_basis_matrix(basis_operator_string)
                    else:
                        # For IXYZ basis, use standard conversion
                        operator_matrix = self.operator_string_to_matrix(basis_operator_string)

                    overlaps[i] = np.trace(operator @ operator_matrix) / d

                except Exception as e:
                    if verbose:
                        print(f"Error processing basis operator {i}: '{self.pauli_basis[i]}' - {e}")
                    overlaps[i] = 0.0  # Set to zero if conversion fails

        if verbose:
            total_time = time.time() - start_time
            print(f"Batched computation completed in {total_time:.3f}s")

        return overlaps

    def _create_spin_basis_matrix(self, operator_string: str) -> np.ndarray:
        """Create matrix for S+/S- basis operator strings."""
        # Parse the S+/S- operator string
        operators = self._parse_operator_string(operator_string)

        if len(operators) != self.n_qubits:
            raise ValueError(f"Parsed {len(operators)} operators from '{operator_string}', expected {self.n_qubits}")

        # Create matrix using spin operator matrices
        result = self.operator_matrices[operators[0]]
        for op in operators[1:]:
            result = np.kron(result, self.operator_matrices[op])
        return result

    def apply_unitary_to_operator(self, operator: np.ndarray, unitary: np.ndarray) -> np.ndarray:
        """Apply unitary evolution: U† O U."""
        return unitary.conj().T @ operator @ unitary

    def compute_weight_distribution(self, overlaps: np.ndarray) -> np.ndarray:
        """Compute weight distribution from overlaps."""
        return np.abs(overlaps) ** 2

    def evolve_operator(self, initial_operator_string: str, unitaries: List[np.ndarray],
                        time_steps: int, verbose: bool = False,
                        method: str = 'vectorized') -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Time evolve an initial operator and track its spreading.

        The initial_operator_string can be in any format (IXYZ or S+/S-) regardless of
        the symmetry setting. The symmetry only affects which basis operators are used
        for decomposition and analysis.
        """

        if verbose:
            print(f"Starting evolution using {method} method")
            print(f"System: {self.n_qubits} qubits, {len(self.pauli_basis):,} basis states")
            print(f"Symmetry: {self.symmetry}, Basis: {self.basis_type}")
            print(f"Initial operator: {initial_operator_string}")

            # Show which format was detected
            if self._is_ixyz_string(initial_operator_string):
                print(f"Detected IXYZ format initial operator")
            elif self._contains_spin_operators(initial_operator_string):
                print(f"Detected S+/S- format initial operator")

        # Choose overlap computation method
        if method == 'vectorized':
            overlap_func = self.compute_overlap_vectorized
        elif method == 'batched':
            overlap_func = self.compute_overlap_batched
        else:
            raise ValueError(f"Unknown method: {method}")

        # Initialize with the initial operator (handles any format automatically)
        current_operator = self.operator_string_to_matrix(initial_operator_string)

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
            current_operator = self.apply_unitary_to_operator(current_operator, unitary)

            # Store evolved operator
            evolved_operators.append(current_operator.copy())

            # Compute overlaps and weights
            overlaps = overlap_func(current_operator, verbose=verbose)
            weights = self.compute_weight_distribution(overlaps)

            overlaps_per_time.append(overlaps)
            weights_per_time.append(weights)

        if verbose:
            print(f"\n✓ Evolution completed using {method} method!")

        return evolved_operators, overlaps_per_time, weights_per_time

import numpy as np
from typing import List, Callable, Tuple
from typing import List, Callable, Tuple
import numpy as np

class GramSchmidtOrthogonalizer:
    """Gram-Schmidt orthogonalization and Lanczos routines for evolved operators."""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def robust_modified_gram_schmidt(
            self,
            operators: list,
            norm_threshold: float = 1e-15,
            overlap_threshold: float = 1e-10,
            verbose: bool = True
    ) -> list:
        """Robust modified Gram-Schmidt orthogonalization with tunable thresholds."""
        d = operators[0].shape[0]
        orthogonal_ops = []

        for i, op in enumerate(operators):
            v = op.copy()
            for k in orthogonal_ops:
                proj = np.trace(k.conj().T @ v) / d
                v = v - proj * k

            norm = np.sqrt(np.real(np.trace(v.conj().T @ v)) / d)
            if norm > norm_threshold:
                v = v / norm
                max_overlap = 0.0
                for k in orthogonal_ops:
                    overlap = abs(np.trace(k.conj().T @ v) / d)
                    max_overlap = max(max_overlap, overlap)
                if max_overlap < overlap_threshold:
                    orthogonal_ops.append(v)
                    if verbose:
                        print(f"✓ Kept operator {i} (norm={norm:.3e}, max_overlap={max_overlap:.3e})")
                else:
                    if verbose:
                        print(f"✗ Rejected operator {i}: max_overlap={max_overlap:.3e} > {overlap_threshold:.1e}")
            else:
                if verbose:
                    print(f"✗ Dropped operator {i}: norm={norm:.3e} < {norm_threshold:.1e}")


        return orthogonal_ops

    def express_in_orthogonal_basis(self, operators: List[np.ndarray],
                                    orthogonal_operators: List[np.ndarray]) -> List[List[complex]]:
        """Express operators in orthogonal basis."""
        d = operators[0].shape[0]
        coefficients_list = []

        for operator in operators:
            coeff_per_op = []
            for Kj in orthogonal_operators:
                coeff = np.trace(Kj.conj().T @ operator) / d
                coeff_per_op.append(coeff)
            coefficients_list.append(coeff_per_op)

        return coefficients_list

    def lanczos_evolution(
            self,
            initial_op: np.ndarray,
            circuit_evolution: Callable,
            max_steps: int = 100,
            tol: float = 1e-15,
            verbose: bool = True
    ) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """
        Standard Lanczos construction.
        Returns (orthogonal_basis, alpha_coeffs, beta_coeffs).
        """
        d = initial_op.shape[0]
        K_prev = None
        K_curr = initial_op / np.linalg.norm(initial_op, 'fro')
        basis = [K_curr.copy()]
        alpha_coeffs = []
        beta_coeffs = []

        for n in range(max_steps):
            if verbose and n % 10 == 0:
                print(f"Lanczos step {n}")

            LK = circuit_evolution(K_curr)
            alpha_n = np.real(np.trace(K_curr.conj().T @ LK)) / d
            alpha_coeffs.append(alpha_n)

            w = LK - alpha_n * K_curr
            if K_prev is not None and len(beta_coeffs) > 0:
                w = w - beta_coeffs[-1] * K_prev

            beta_n = np.linalg.norm(w, 'fro')
            beta_coeffs.append(beta_n)
            if verbose:
                print(f"  α_{n} = {alpha_n:.6f}, β_{n} = {beta_n:.6e}")

            if beta_n < tol:
                if verbose:
                    print(f"Krylov space saturated at step {n} (β = {beta_n:.2e})")
                break

            K_prev = K_curr.copy()
            K_curr = w / beta_n
            basis.append(K_curr.copy())

        return basis, alpha_coeffs, beta_coeffs

    def layer_by_layer_lanczos(
            self,
            evolved_operators: List[np.ndarray],
            verbose: bool = True
    ) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """
        Perform layer-by-layer Lanczos using the actual sequence of evolved operators.
        Returns:
            - basis: List of orthonormal basis vectors.
            - alpha_coeffs: Diagonal elements of the tridiagonal matrix.
            - beta_coeffs: Off-diagonal elements of the tridiagonal matrix.
        """
        d = evolved_operators[0].shape[0]
        basis = []
        alpha_coeffs = []
        beta_coeffs = []

        # Normalize the first operator
        O0 = evolved_operators[0] / np.sqrt(np.trace(evolved_operators[0].conj().T @ evolved_operators[0]) / d)
        basis.append(O0)

        if verbose:
            print(f"Step 0: norm={np.linalg.norm(O0, 'fro'):.3e}")

        for n in range(1, len(evolved_operators)):
            v = evolved_operators[n].copy()

            # Orthogonalize against the previous two basis vectors
            if n > 1:
                proj_prev2 = np.trace(basis[n - 2].conj().T @ v) / d
                v -= proj_prev2 * basis[n - 2]
                if verbose:
                    print(f"  Remove proj to O_{n-2}: {proj_prev2:.3e}")

            proj_prev1 = np.trace(basis[n - 1].conj().T @ v) / d
            v -= proj_prev1 * basis[n - 1]
            if verbose:
                print(f"  Remove proj to O_{n-1}: {proj_prev1:.3e}")

            # Normalize the new vector
            norm = np.sqrt(np.trace(v.conj().T @ v) / d)
            if norm < 1e-14:
                if verbose:
                    print(f"  Stopping at step {n}: norm too small ({norm:.2e})")
                break
            v /= norm
            basis.append(v)

            # Compute alpha and beta coefficients
            alpha_n = proj_prev1
            beta_n = norm
            alpha_coeffs.append(alpha_n)
            beta_coeffs.append(beta_n)

            if verbose:
                print(f"Step {n}: alpha={alpha_n:.3e}, beta={beta_n:.3e}")

        return basis, alpha_coeffs, beta_coeffs

    def validate_lanczos_basis(self, basis: List[np.ndarray], verbose: bool = True) -> bool:
        """Validate that Lanczos basis is orthonormal."""
        d = basis[0].shape[0]
        n_basis = len(basis)
        max_error = 0.0

        for i in range(n_basis):
            for j in range(n_basis):
                overlap = np.trace(basis[i].conj().T @ basis[j]) / d
                expected = 1.0 if i == j else 0.0
                error = abs(overlap - expected)
                max_error = max(max_error, error)
                if error > 1e-10 and verbose:
                    print(f"  Warning: ⟨K_{i}|K_{j}⟩ = {overlap:.2e} (expected {expected})")

        if verbose:
            print(f"  Max orthogonality error: {max_error:.2e}")

        return max_error < 1e-8

    def compute_krylov_complexity(
            self,
            alpha_coeffs: List[float],
            beta_coeffs: List[float],
            time_steps: int
    ) -> List[float]:
        """
        Compute Krylov complexity using the tridiagonal matrix.
        Compatible with existing plotting code.
        """
        import scipy.linalg

        dim = len(alpha_coeffs)
        if dim == 0:
            return [0.0] * (time_steps + 1)

        T = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            T[i, i] = alpha_coeffs[i]
        for i in range(dim - 1):
            T[i, i + 1] = beta_coeffs[i]
            T[i + 1, i] = beta_coeffs[i]

        psi_0 = np.zeros(dim)
        psi_0[0] = 1.0
        complexities = []

        for t in range(time_steps + 1):
            if t == 0:
                psi_t = psi_0.copy()
            else:
                psi_t = scipy.linalg.expm(-1j * T * t) @ psi_0
            complexity = sum(n * abs(psi_t[n]) ** 2 for n in range(dim))
            complexities.append(complexity.real)

        return complexities



# Circuit construction utilities (unchanged)
def create_partial_swap_gate(theta: float) -> np.ndarray:
    """Create a partial SWAP gate with parameter theta."""
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), 1j * np.sin(theta), 0],
        [0, 1j * np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ], dtype=complex)


def embed_two_qubit_gate(gate: np.ndarray, i: int, j: int, n: int) -> np.ndarray:
    """Embed a 2-qubit gate into an n-qubit system."""
    if i >= j or i < 0 or j >= n:
        raise ValueError(f"Invalid qubit pair ({i}, {j}) for {n} qubits")

    # Create permutation to move qubits i,j to positions 0,1
    qubit_order = [i, j] + [q for q in range(n) if q != i and q != j]
    inv_order = np.argsort(qubit_order)
    dim = 2 ** n

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


def create_brickwork_unitaries(n_qubits: int, theta: float) -> List[np.ndarray]:
    """Create brickwork circuit unitaries."""
    if n_qubits % 2 != 0:
        print(f"Warning: {n_qubits} is odd. Last qubit will not participate in all gates.")

    pswap = create_partial_swap_gate(theta)

    # Even layer: (0,1), (2,3), (4,5), (6,7)
    U_even = np.eye(2 ** n_qubits, dtype=complex)
    for i in range(0, n_qubits - 1, 2):
        gate_full = embed_two_qubit_gate(pswap, i, i + 1, n_qubits)
        U_even = gate_full @ U_even

    # Odd layer: (1,2), (3,4), (5,6), (7,8)
    U_odd = np.eye(2 ** n_qubits, dtype=complex)
    for i in range(1, n_qubits - 1, 2):
        gate_full = embed_two_qubit_gate(pswap, i, i + 1, n_qubits)
        U_odd = gate_full @ U_odd

    return [U_even, U_odd]


def create_general_circuit_unitaries(n_qubits: int, circuit_structure: List[List[Tuple[int, int]]],
                                     theta: float) -> List[np.ndarray]:
    """Create unitaries for general circuit structures."""
    pswap = create_partial_swap_gate(theta)
    unitaries = []

    for pairs in circuit_structure:
        U = np.eye(2 ** n_qubits, dtype=complex)
        for pair in pairs:
            gate_full = embed_two_qubit_gate(pswap, pair[0], pair[1], n_qubits)
            U = gate_full @ U
        unitaries.append(U)

    return unitaries


def generate_nonmarkovian_circuit_unitaries(n_qubits: int, pattern_string: str,
                                            theta: float) -> List[np.ndarray]:
    """Generate unitaries for non-Markovian circuits."""
    pattern_dict = {
        'j': [(i, i + 1) for i in range(0, n_qubits - 1, 2)],  # even pairs
        'g': [(i, i + 1) for i in range(1, n_qubits - 1, 2)]  # odd pairs
    }

    pswap = create_partial_swap_gate(theta)
    unitaries = []

    for letter in pattern_string:
        if letter not in pattern_dict:
            raise ValueError(f"Unknown pattern letter: {letter}")

        pairs = pattern_dict[letter]
        U = np.eye(2 ** n_qubits, dtype=complex)
        for i, j in pairs:
            gate_full = embed_two_qubit_gate(pswap, i, j, n_qubits)
            U = gate_full @ U
        unitaries.append(U)

    return unitaries


def generate_markov_chain_non_markovian_string(length: int,
                                               transition_matrix: Optional[Dict] = None) -> str:
    """Generate a non-Markovian string using a Markov chain with memory."""
    if transition_matrix is None:
        transition_matrix = {
            ('j', 'j'): {'j': 0.05, 'g': 0.95},
            ('j', 'g'): {'j': 0.1, 'g': 0.9},
            ('g', 'j'): {'j': 0.02, 'g': 0.98},
            ('g', 'g'): {'j': 0.02, 'g': 0.98},
        }

    if length < 2:
        return 'jg'[:length]

    result = ['j', 'g']  # Start with initial characters
    for _ in range(length - 2):
        last_two = tuple(result[-2:])
        probabilities = transition_matrix.get(last_two, {'j': 0.5, 'g': 0.5})
        next_char = random.choices(list(probabilities.keys()),
                                   weights=list(probabilities.values()))[0]
        result.append(next_char)

    return ''.join(result)




def analyze_operator_spreading(n_qubits: int, initial_operator_string: str,
                               unitaries: List[np.ndarray], time_steps: int,
                               symmetry: Optional[str] = None, verbose: bool = False,
                               method: str = 'vectorized',
                               compute_lanczos: bool = True) -> Dict:
    """Main function to analyze operator spreading with corrected symmetries."""
    analyzer = QuantumOperatorAnalyzer(n_qubits, symmetry)

    if verbose:
        print(f"Initialized analyzer with {symmetry} symmetry")
        print(f"Basis type: {analyzer.basis_type}")
        print(f"Basis size: {len(analyzer.pauli_basis):,}")

    evolved_ops, overlaps, weights = analyzer.evolve_operator(
        initial_operator_string, unitaries, time_steps, verbose=verbose, method=method
    )

    results = {
        'evolved_operators': evolved_ops,
        'overlaps_per_time': overlaps,
        'weights_per_time': weights,
        'pauli_basis': analyzer.pauli_basis,
        'basis_size': len(analyzer.pauli_basis),
        'basis_type': analyzer.basis_type,
        'symmetry': symmetry,
        'method': method
    }

    if compute_lanczos:
        orthogonalizer = GramSchmidtOrthogonalizer(analyzer)
        basis, alpha, beta = orthogonalizer.layer_by_layer_lanczos(evolved_ops, verbose=verbose)
        results['lanczos_basis'] = basis
        results['lanczos_alpha'] = alpha
        results['lanczos_beta'] = beta

    return results


def orthogonalize_evolved_operators(evolved_operators: List[np.ndarray],
                                    analyzer: QuantumOperatorAnalyzer,
                                    use_lanczos: bool = False,
                                    verbose: bool = True) -> Dict:
    """Perform Gram-Schmidt or layer-by-layer Lanczos orthogonalization on evolved operators."""
    orthogonalizer = GramSchmidtOrthogonalizer(analyzer)

    if use_lanczos:
        basis, alpha, beta = orthogonalizer.layer_by_layer_lanczos(evolved_operators, verbose=verbose)
        return {
            'lanczos_basis': basis,
            'lanczos_alpha': alpha,
            'lanczos_beta': beta
        }
    else:
        orthogonal_ops = orthogonalizer.robust_modified_gram_schmidt(evolved_operators, verbose=verbose)
        coefficients = orthogonalizer.express_in_orthogonal_basis(evolved_operators, orthogonal_ops)
        return {
            'orthogonal_operators': orthogonal_ops,
            'coefficients_in_orthogonal_basis': coefficients
        }


# Updated validation function - now only validates decomposition basis compatibility
def validate_initial_operator(initial_operator_string: str, symmetry: Optional[str] = None) -> bool:
    """Validate that initial operator is properly formatted.

    Note: ANY operator can be used with ANY symmetry setting. The symmetry only
    affects the decomposition basis, not the allowed initial operators.
    This function just checks for proper formatting.
    """
    if symmetry is None:
        return True

    # Just check that the string is properly formatted
    if len(initial_operator_string) == 0:
        print(f"Warning: Empty operator string")
        return False

    # Check for valid characters
    valid_chars = set('IXYZ')
    if 'S+' in initial_operator_string or 'S-' in initial_operator_string:
        # S+/S- format - more complex validation
        i = 0
        while i < len(initial_operator_string):
            if i < len(initial_operator_string) - 1 and initial_operator_string[i:i + 2] in ['S+', 'S-']:
                i += 2
            elif initial_operator_string[i] in 'IZ':
                i += 1
            else:
                print(f"Warning: Invalid character '{initial_operator_string[i]}' in S+/S- format")
                return False
    else:
        # IXYZ format
        for char in initial_operator_string:
            if char not in valid_chars:
                print(f"Warning: Invalid character '{char}' in IXYZ format")
                return False

    print(f"✓ Initial operator '{initial_operator_string}' is properly formatted")
    print(f"  Can be used with {symmetry} symmetry (or any other symmetry)")
    return True

def express_in_orthogonal_basis(operators, orthogonal_operators):
    d = operators[0].shape[0]
    coefficients_list = []
    for operator in operators:
        coeff_per_op = []
        for Kj in orthogonal_operators:
            coeff = np.trace(Kj.conj().T @ operator) / d
            coeff_per_op.append(coeff)
        coefficients_list.append(coeff_per_op)
    return np.array(coefficients_list, dtype=np.complex128)

def compute_gs_complexity_and_active(gs_coeffs, weight_threshold=1e-8):
    gs_probs = np.abs(gs_coeffs)**2
    gs_complexity = np.sum(gs_probs * np.arange(gs_probs.shape[1]), axis=1)
    gs_active = np.sum(gs_probs > weight_threshold, axis=1)
    return gs_complexity, gs_active

# --- Plotting Function (2-panel, publication style) ---
def moving_average(x, window=30):
    return np.convolve(x, np.ones(window)/window, mode='same')

def plot_gs_and_pauli_2panel(
    labels, gs_complexities, active_paulis, pastel_colors=None, line_styles=None, avg_window=30
):
    """
    Panel 1: Smoothed GS-basis complexity (sum n |a_n|^2)
    Panel 2: Smoothed number of active Pauli operators (weight > threshold)
    """
    if pastel_colors is None:
        pastel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    if line_styles is None:
        line_styles = ['-'] * len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for i, label in enumerate(labels):
        color = pastel_colors[i % len(pastel_colors)]
        ls = line_styles[i % len(line_styles)]
        smoothed_gs = moving_average(gs_complexities[i], window=avg_window)
        smoothed_pauli = moving_average(active_paulis[i], window=avg_window)
        axes[0].plot(smoothed_gs, label=label, color=color, linestyle=ls, linewidth=2, alpha=0.85)
        axes[1].plot(smoothed_pauli, label=label, color=color, linestyle=ls, linewidth=2, alpha=0.85)
    axes[0].set_title('Time-averaged GS-basis Complexity')
    axes[1].set_title('Time-averaged Active Pauli Operators')
    axes[0].set_xlabel('Time step', fontsize=15)
    axes[1].set_xlabel('Time step', fontsize=15)
    axes[0].set_ylabel('GS Complexity', fontsize=15)
    axes[1].set_ylabel('Active Pauli ops', fontsize=15)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Plotting functions with corrected symmetry labels
def plot_operator_evolution(results: Dict, top_n: int = 20, figsize: Tuple = (12, 8)):
    """Plot evolution of top operators with corrected symmetry labels."""
    weights_per_time = results['weights_per_time']
    pauli_basis = results['pauli_basis']
    symmetry = results.get('symmetry', 'None')
    basis_type = results.get('basis_type', 'IXYZ')
    n_time_steps = len(weights_per_time)

    # Find top operators by maximum weight
    all_max_weights = []
    for i, operator_string in enumerate(pauli_basis):
        max_weight = max(weights_per_time[t][i] for t in range(n_time_steps))
        all_max_weights.append((max_weight, i, operator_string))

    all_max_weights.sort(reverse=True)

    plt.figure(figsize=figsize)

    for rank, (_, idx, operator_string) in enumerate(all_max_weights[:top_n]):
        weights_over_time = [weights_per_time[t][idx] for t in range(n_time_steps)]
        color = PASTEL_COLORS[rank % len(PASTEL_COLORS)]

        plt.plot(range(n_time_steps), weights_over_time, 'o-',
                 linewidth=2, color=color, label=operator_string,
                 markersize=4, alpha=0.8)

    plt.xlabel('Time Step')
    plt.ylabel('Weight')
    plt.title(f'Top {top_n} Operators: Weight Evolution\n(Symmetry: {symmetry}, Basis: {basis_type})')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


    # Additional utility functions with corrected symmetry handling


def compute_complexity_metrics(results: Dict) -> Dict:
    """Compute various complexity metrics from the spreading results."""
    weights_per_time = results['weights_per_time']
    n_time_steps = len(weights_per_time)

    metrics = {
        'nonzero_counts': [],
        'shannon_entropy': [],
        'participation_ratio': [],
        'weight_concentration_top10': []
    }

    for weights in weights_per_time:
        w = np.array(weights)
        active_mask = w > WEIGHT_THRESHOLD
        w_active = w[active_mask]

        # Number of active operators
        metrics['nonzero_counts'].append(np.sum(active_mask))

        if len(w_active) > 0:
            # Normalize for entropy calculation
            w_norm = w_active / np.sum(w_active)

            # Shannon entropy
            entropy = -np.sum(w_norm * np.log(w_norm + 1e-16))
            metrics['shannon_entropy'].append(entropy)

            # Participation ratio (inverse participation ratio)
            participation = 1 / np.sum(w_norm ** 2)
            metrics['participation_ratio'].append(participation)

            # Weight concentration in top 10%
            n_top = max(1, len(w) // 10)
            top_weights = np.sort(w)[-n_top:]
            concentration = np.sum(top_weights) / np.sum(w)
            metrics['weight_concentration_top10'].append(concentration)
        else:
            metrics['shannon_entropy'].append(0)
            metrics['participation_ratio'].append(0)
            metrics['weight_concentration_top10'].append(0)

    return metrics


def analyze_convergence(weights_per_time: List[np.ndarray], window_size: int = 5) -> Dict:
    """Analyze convergence properties of the weight evolution."""
    if len(weights_per_time) < window_size + 1:
        return {'converged': False, 'message': 'Insufficient time steps for convergence analysis'}

    # Calculate relative change in weight distribution
    relative_changes = []
    for t in range(window_size, len(weights_per_time)):
        w_curr = np.array(weights_per_time[t])
        w_prev = np.array(weights_per_time[t - 1])

        # Avoid division by zero
        denominator = np.maximum(w_prev, WEIGHT_THRESHOLD)
        rel_change = np.mean(np.abs(w_curr - w_prev) / denominator)
        relative_changes.append(rel_change)

    # Check if converged (small changes in recent window)
    if len(relative_changes) >= window_size:
        recent_changes = relative_changes[-window_size:]
        avg_change = np.mean(recent_changes)
        converged = avg_change < 1e-6

        return {
            'converged': converged,
            'average_recent_change': avg_change,
            'relative_changes': relative_changes,
            'convergence_threshold': 1e-6
        }

    return {'converged': False, 'message': 'Insufficient data for convergence analysis'}


def plot_complexity_metrics(results: Dict, figsize: Tuple = (14, 10)):
    """Plot comprehensive complexity metrics with corrected symmetry labels."""
    metrics = compute_complexity_metrics(results)
    symmetry = results.get('symmetry', 'None')
    basis_type = results.get('basis_type', 'IXYZ')
    n_time_steps = len(metrics['nonzero_counts'])
    time_steps = range(n_time_steps)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Operator Spreading Complexity Metrics\n(Symmetry: {symmetry}, Basis: {basis_type})',
                 fontsize=16, fontweight='bold')

    # Top left: Active operators
    axes[0, 0].plot(time_steps, metrics['nonzero_counts'], 'o-',
                    color='#FFB3BA', linewidth=2, markersize=5)
    axes[0, 0].set_ylabel('Active Operators')
    axes[0, 0].set_title('Operator Space Exploration')
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Shannon entropy
    axes[0, 1].plot(time_steps, metrics['shannon_entropy'], 's-',
                    color='#BAFFC9', linewidth=2, markersize=5)
    axes[0, 1].set_ylabel('Shannon Entropy')
    axes[0, 1].set_title('Weight Distribution Entropy')
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom left: Participation ratio
    axes[1, 0].plot(time_steps, metrics['participation_ratio'], '^-',
                    color='#BAE1FF', linewidth=2, markersize=5)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Participation Ratio')
    axes[1, 0].set_title('Effective Dimensionality')
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom right: Weight concentration
    axes[1, 1].plot(time_steps, metrics['weight_concentration_top10'], 'D-',
                    color='#E1BAFF', linewidth=2, markersize=5)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Weight in Top 10%')
    axes[1, 1].set_title('Weight Concentration')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return metrics


def save_results(results: Dict, filename: str):
    """Save results to a compressed numpy file."""
    import pickle
    try:
        with open(f"{filename}.pkl", 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {filename}.pkl")
    except Exception as e:
        print(f"Failed to save results: {e}")


def load_results(filename: str) -> Dict:
    """Load results from a compressed numpy file."""
    import pickle
    try:
        with open(f"{filename}.pkl", 'rb') as f:
            results = pickle.load(f)
        print(f"Results loaded from {filename}.pkl")
        return results
    except Exception as e:
        print(f"Failed to load results: {e}")
        return {}


# Updated plotting functions for Pauli operator statistics with corrected labels
def plot_pauli_operator_counts(results, figsize=(12, 5)):
    """Plot total and active operator counts over time with corrected symmetry labels."""
    weights_per_time = results['weights_per_time']
    symmetry = results.get('symmetry', 'None')
    basis_type = results.get('basis_type', 'IXYZ')
    total_operators = len(weights_per_time[0])
    time_steps = range(len(weights_per_time))

    # Count active operators at each time step
    active_counts = []
    for weights in weights_per_time:
        active = np.sum(np.array(weights) > 1e-30)  # Non-zero threshold
        active_counts.append(active)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'Operator Space Analysis (Symmetry: {symmetry}, Basis: {basis_type})', fontsize=14)

    # Left plot: Active vs Total
    ax1.plot(time_steps, active_counts, 'o-', linewidth=2,
             color='#FF6B6B', markersize=6, label='Active operators')
    ax1.axhline(y=total_operators, color='#4ECDC4', linestyle='--',
                linewidth=2, label=f'Total operators ({total_operators})')

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Number of Operators')
    ax1.set_title('Operator Space Exploration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Percentage active
    percentage_active = [100 * count / total_operators for count in active_counts]
    ax2.plot(time_steps, percentage_active, 's-', linewidth=2,
             color='#45B7D1', markersize=6)

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Percentage Active (%)')
    ax2.set_title('Fraction of Operator Space Explored')
    ax2.grid(True, alpha=0.3)

    # Add text annotations
    final_active = active_counts[-1]
    final_percentage = percentage_active[-1]

    ax1.text(0.02, 0.98, f'Final: {final_active}/{total_operators}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.text(0.02, 0.98, f'Final: {final_percentage:.1f}%',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"Operator Summary ({symmetry} symmetry, {basis_type} basis):")
    print(f"  Total operators in basis: {total_operators}")
    print(f"  Initial active: {active_counts[0]}")
    print(f"  Final active: {final_active}")
    print(f"  Exploration: {final_percentage:.1f}% of operator space")

    return active_counts


def plot_nonzero_weight_histogram(results, time_step=-1, bins=50, figsize=(10, 6)):
    """Plot histogram of non-zero operator weights with corrected symmetry labels."""
    weights = np.array(results['weights_per_time'][time_step])
    pauli_basis = results['pauli_basis']
    symmetry = results.get('symmetry', 'None')
    basis_type = results.get('basis_type', 'IXYZ')

    # Get non-zero weights
    nonzero_mask = weights > 1e-30
    nonzero_weights = weights[nonzero_mask]
    nonzero_labels = [pauli_basis[i] for i in range(len(pauli_basis)) if nonzero_mask[i]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(f'Non-Zero Operator Weights (Symmetry: {symmetry}, Basis: {basis_type})', fontsize=14)

    # Top: Histogram of weights
    n, bin_edges, patches = ax1.hist(nonzero_weights, bins=bins,
                                     color='#FFB6C1', alpha=0.7,
                                     edgecolor='#FF69B4', linewidth=0.5)

    # Color gradient for histogram bars
    for i, p in enumerate(patches):
        p.set_facecolor(plt.cm.viridis(i / len(patches)))

    ax1.set_xlabel('Weight')
    ax1.set_ylabel('Count')
    ax1.set_title(
        f'Distribution of Non-Zero Weights (t={time_step if time_step >= 0 else len(results["weights_per_time"]) - 1})')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Add statistics
    mean_weight = np.mean(nonzero_weights)
    max_weight = np.max(nonzero_weights)
    min_weight = np.min(nonzero_weights)

    stats_text = f'Non-zero operators: {len(nonzero_weights)}\n'
    stats_text += f'Mean: {mean_weight:.2e}\n'
    stats_text += f'Max: {max_weight:.2e}\n'
    stats_text += f'Min: {min_weight:.2e}'

    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Bottom: Log-scale histogram
    ax2.hist(nonzero_weights, bins=bins, color='#87CEEB', alpha=0.7,
             edgecolor='#4682B4', linewidth=0.5)
    ax2.set_xlabel('Weight')
    ax2.set_ylabel('Count')
    ax2.set_title('Same Distribution (Log-Log Scale)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print top operators
    print(f"\nTop 20 operators by weight ({symmetry} symmetry, {basis_type} basis):")
    sorted_indices = np.argsort(nonzero_weights)[::-1]
    for i in range(min(20, len(sorted_indices))):
        idx = sorted_indices[i]
        weight = nonzero_weights[idx]
        operator_str = nonzero_labels[idx]
        print(f"  {i + 1:2d}. {operator_str}: {weight:.3e}")

    return nonzero_weights, nonzero_labels


def plot_weight_evolution_heatmap(results, max_operators=50, figsize=(12, 8)):
    """Plot heatmap showing weight evolution of top operators with corrected symmetry labels."""
    weights_per_time = results['weights_per_time']
    pauli_basis = results['pauli_basis']
    symmetry = results.get('symmetry', 'None')
    basis_type = results.get('basis_type', 'IXYZ')
    n_time_steps = len(weights_per_time)

    # Convert to numpy array for easier manipulation
    weight_matrix = np.array(weights_per_time).T  # Shape: (n_operators, n_time)

    # Find operators with highest maximum weight
    max_weights = np.max(weight_matrix, axis=1)
    top_indices = np.argsort(max_weights)[::-1][:max_operators]

    # Select top operators
    top_weights = weight_matrix[top_indices, :]
    top_labels = [pauli_basis[i] for i in top_indices]

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(top_weights, aspect='auto', cmap='hot',
                   origin='lower', interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Weight', fontsize=12)

    # Labels and title
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Operator', fontsize=12)
    ax.set_title(f'Weight Evolution of Top {len(top_labels)} Operators\n'
                 f'(Symmetry: {symmetry}, Basis: {basis_type})',
                 fontsize=14, pad=20)

    # Set ticks
    ax.set_xticks(range(0, n_time_steps, max(1, n_time_steps // 10)))
    ax.set_yticks(range(len(top_labels)))
    ax.set_yticklabels(top_labels, fontsize=8)

    plt.tight_layout()
    plt.show()

    return top_weights, top_labels


# Updated convenience functions
def analyze_no_symmetry(n_qubits: int, initial_operator: str, unitaries: List[np.ndarray],
                        time_steps: int, **kwargs) -> Dict:
    """Analyze spreading with no symmetry constraints.

    initial_operator can be any valid operator string (IXYZ or S+/S- format).
    """
    validate_initial_operator(initial_operator, None)
    return analyze_operator_spreading(n_qubits, initial_operator, unitaries, time_steps,
                                      symmetry=None, **kwargs)


def analyze_u1_symmetry(n_qubits: int, initial_operator: str, unitaries: List[np.ndarray],
                        time_steps: int, **kwargs) -> Dict:
    """Analyze spreading with U(1) symmetry (even X+Y count in decomposition basis).

    initial_operator can be any valid operator string (IXYZ or S+/S- format).
    The U(1) symmetry only affects which basis operators are used for decomposition.
    """
    validate_initial_operator(initial_operator, 'U1')
    return analyze_operator_spreading(n_qubits, initial_operator, unitaries, time_steps,
                                      symmetry='U1', **kwargs)


def analyze_u1_sbasis(n_qubits: int, initial_operator: str, unitaries: List[np.ndarray],
                      time_steps: int, **kwargs) -> Dict:
    """Analyze spreading with U(1) symmetry in S+/S- decomposition basis.

    initial_operator can be any valid operator string (IXYZ or S+/S- format).
    The U(1)_sbasis symmetry only affects which basis operators are used for decomposition.
    """
    validate_initial_operator(initial_operator, 'U1_sbasis')
    return analyze_operator_spreading(n_qubits, initial_operator, unitaries, time_steps,
                                      symmetry='U1_sbasis', **kwargs)


def plot_comprehensive_pauli_analysis(results, figsize=(16, 12)):
    """
    Comprehensive plotting function that shows all Pauli operator statistics.

    Args:
        results: Output from analyze_operator_spreading()
        figsize: Figure size tuple
    """
    fig = plt.figure(figsize=figsize)

    # Create a 2x3 subplot grid
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)

    weights_per_time = results['weights_per_time']
    pauli_basis = results['pauli_basis']
    total_operators = len(pauli_basis)
    time_steps = range(len(weights_per_time))

    # 1. Active operator count over time
    ax1 = fig.add_subplot(gs[0, 0])
    active_counts = [np.sum(np.array(w) > 1e-30) for w in weights_per_time]
    ax1.plot(time_steps, active_counts, 'o-', linewidth=2, color='#FF6B6B', markersize=4)
    ax1.set_ylabel('Active Operators')
    ax1.set_title('Active Pauli Operators vs Time')
    ax1.grid(True, alpha=0.3)

    # 2. Percentage of space explored
    ax2 = fig.add_subplot(gs[0, 1])
    percentage = [100 * count / total_operators for count in active_counts]
    ax2.plot(time_steps, percentage, 's-', linewidth=2, color='#45B7D1', markersize=4)
    ax2.set_ylabel('% of Pauli Space')
    ax2.set_title('Pauli Space Exploration')
    ax2.grid(True, alpha=0.3)

    # 3. Weight histogram (final time)
    ax3 = fig.add_subplot(gs[1, 0])
    final_weights = np.array(weights_per_time[-1])
    nonzero_weights = final_weights[final_weights > 1e-30]
    ax3.hist(nonzero_weights, bins=30, color='#98D8C8', alpha=0.7, edgecolor='#2F7D32')
    ax3.set_xlabel('Weight')
    ax3.set_ylabel('Count')
    ax3.set_title('Final Weight Distribution')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # 4. Log-log weight histogram
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(nonzero_weights, bins=30, color='#FFB74D', alpha=0.7, edgecolor='#F57C00')
    ax4.set_xlabel('Weight')
    ax4.set_ylabel('Count')
    ax4.set_title('Final Weight Distribution (Log-Log)')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    # 5. Heatmap of top operators (spans bottom row)
    ax5 = fig.add_subplot(gs[2, :])

    # Get top 20 operators for heatmap
    weight_matrix = np.array(weights_per_time).T
    max_weights = np.max(weight_matrix, axis=1)
    top_indices = np.argsort(max_weights)[::-1][:20]
    top_weights = weight_matrix[top_indices, :]
    top_labels = [pauli_basis[i] for i in top_indices]

    im = ax5.imshow(top_weights, aspect='auto', cmap='plasma', origin='lower')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Top Pauli Operators')
    ax5.set_title('Weight Evolution Heatmap (Top 20 Operators)')
    ax5.set_yticks(range(len(top_labels)))
    ax5.set_yticklabels(top_labels, fontsize=8)

    # Colorbar for heatmap
    cbar = plt.colorbar(im, ax=ax5, shrink=0.6)
    cbar.set_label('Weight')

    # Add summary statistics
    final_active = active_counts[-1]
    final_percentage = percentage[-1]
    max_weight = np.max(final_weights)

    summary_text = f'Summary:\n'
    summary_text += f'Total operators: {total_operators}\n'
    summary_text += f'Final active: {final_active}\n'
    summary_text += f'Space explored: {final_percentage:.1f}%\n'
    summary_text += f'Max weight: {max_weight:.2e}'

    fig.text(0.02, 0.98, summary_text, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.show()

    return {
        'active_counts': active_counts,
        'nonzero_weights': nonzero_weights,
        'top_operators': list(zip(top_labels, np.max(top_weights, axis=1)))
    }


def plot_dynamic_active_operators(results, threshold=1e-6, figsize=(12, 8)):
    """Show which operators are active at each time step"""
    weights_per_time = results['weights_per_time']
    pauli_basis = results['pauli_basis']

    # Create active operator matrix
    active_matrix = []
    active_operators_per_time = []
    all_active_operators = set()

    for t, weights in enumerate(weights_per_time):
        active_indices = [i for i, w in enumerate(weights) if w > threshold]
        active_operators_per_time.append(active_indices)
        all_active_operators.update(active_indices)

        # Create row for this time step
        row = np.zeros(len(pauli_basis))
        for i in active_indices:
            row[i] = weights[i]
        active_matrix.append(row)

    active_matrix = np.array(active_matrix)

    # Sort active operators by first appearance time
    sorted_active = sorted(all_active_operators,
                           key=lambda x: next(t for t, indices in enumerate(active_operators_per_time) if x in indices))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Full heatmap
    im1 = ax1.imshow(active_matrix, aspect='auto', cmap='hot', origin='lower')
    ax1.set_xlabel('Pauli Operator Index')
    ax1.set_ylabel('Time Step')
    ax1.set_title('All Active Operators (Weight > {:.0e})'.format(threshold))
    plt.colorbar(im1, ax=ax1, label='Weight')

    # Focused heatmap on active operators only
    if len(sorted_active) > 0:
        focused_matrix = active_matrix[:, sorted_active]
        im2 = ax2.imshow(focused_matrix, aspect='auto', cmap='hot', origin='lower')
        ax2.set_xlabel('Active Operator Index')
        ax2.set_ylabel('Time Step')
        ax2.set_title(f'Focused View ({len(sorted_active)} Active Operators)')

        # Add Pauli string labels if not too many
        if len(sorted_active) <= 20:
            ax2.set_xticks(range(len(sorted_active)))
            ax2.set_xticklabels([pauli_basis[i] for i in sorted_active], rotation=45, ha='right')

        plt.colorbar(im2, ax=ax2, label='Weight')

    plt.tight_layout()
    plt.show()

    # Print active operators summary
    print(f"\nActive Operators Summary (threshold > {threshold:.0e}):")
    print(f"Total active operators: {len(all_active_operators)}")
    print(f"Top 10 most persistent operators:")

    # Count how many times each operator is active
    persistence_count = {}
    for indices in active_operators_per_time:
        for idx in indices:
            persistence_count[idx] = persistence_count.get(idx, 0) + 1

    top_persistent = sorted(persistence_count.items(), key=lambda x: x[1], reverse=True)[:10]
    for idx, count in top_persistent:
        print(f"  {pauli_basis[idx]}: active {count}/{len(weights_per_time)} steps")

    return active_operators_per_time, sorted_active


def plot_pauli_structure_evolution(results, threshold=1e-6, figsize=(14, 10)):
    """Track evolution by Pauli string characteristics"""
    weights_per_time = results['weights_per_time']
    pauli_basis = results['pauli_basis']

    # Categorize operators by structure and track which strings contribute
    structure_weights = {'I-only': [], 'X-only': [], 'Y-only': [], 'Z-only': [],
                         'Mixed': [], 'XY-mixed': [], 'XZ-mixed': [], 'YZ-mixed': []}
    structure_strings = {'I-only': set(), 'X-only': set(), 'Y-only': set(), 'Z-only': set(),
                         'Mixed': set(), 'XY-mixed': set(), 'XZ-mixed': set(), 'YZ-mixed': set()}

    for weights in weights_per_time:
        categories = {key: 0 for key in structure_weights.keys()}

        for i, (pauli_str, weight) in enumerate(zip(pauli_basis, weights)):
            if weight > threshold:
                x_count = pauli_str.count('X')
                y_count = pauli_str.count('Y')
                z_count = pauli_str.count('Z')
                i_count = pauli_str.count('I')

                if i_count == len(pauli_str):
                    categories['I-only'] += weight
                    structure_strings['I-only'].add(pauli_str)
                elif x_count > 0 and y_count == 0 and z_count == 0:
                    categories['X-only'] += weight
                    structure_strings['X-only'].add(pauli_str)
                elif x_count == 0 and y_count > 0 and z_count == 0:
                    categories['Y-only'] += weight
                    structure_strings['Y-only'].add(pauli_str)
                elif x_count == 0 and y_count == 0 and z_count > 0:
                    categories['Z-only'] += weight
                    structure_strings['Z-only'].add(pauli_str)
                elif x_count > 0 and y_count > 0 and z_count == 0:
                    categories['XY-mixed'] += weight
                    structure_strings['XY-mixed'].add(pauli_str)
                elif x_count > 0 and y_count == 0 and z_count > 0:
                    categories['XZ-mixed'] += weight
                    structure_strings['XZ-mixed'].add(pauli_str)
                elif x_count == 0 and y_count > 0 and z_count > 0:
                    categories['YZ-mixed'] += weight
                    structure_strings['YZ-mixed'].add(pauli_str)
                else:
                    categories['Mixed'] += weight
                    structure_strings['Mixed'].add(pauli_str)

        for key in structure_weights:
            structure_weights[key].append(categories[key])

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    time_steps = range(len(weights_per_time))

    # Plot first 3 categories in separate subplots
    colors = PASTEL_COLORS[:len(structure_weights)]
    category_items = list(structure_weights.items())

    for i in range(3):
        if i < len(category_items):
            category, weights = category_items[i]
            axes[i // 2, i % 2].plot(time_steps, weights, 'o-',
                                     color=colors[i], linewidth=2, markersize=4)
            axes[i // 2, i % 2].set_ylabel('Total Weight')
            axes[i // 2, i % 2].set_title(f'{category} Operators')
            axes[i // 2, i % 2].grid(True, alpha=0.3)

    # Combined plot in bottom-right
    for i, (category, weights) in enumerate(structure_weights.items()):
        if max(weights) > threshold:  # Only plot if category has activity
            axes[1, 1].plot(time_steps, weights, 'o-',
                            color=colors[i % len(colors)], linewidth=2, markersize=3,
                            label=category, alpha=0.8)

    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Total Weight')
    axes[1, 1].set_title('All Categories Combined')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print string breakdown by category
    print(f"\nPauli String Breakdown by Structure (threshold > {threshold:.0e}):")
    for category, strings in structure_strings.items():
        if strings:
            print(f"\n{category} ({len(strings)} unique strings):")
            sorted_strings = sorted(strings)
            for i, s in enumerate(sorted_strings[:10]):  # Show first 10
                print(f"  {s}")
            if len(sorted_strings) > 10:
                print(f"  ... and {len(sorted_strings) - 10} more")

    return structure_weights, structure_strings


def plot_operator_lifecycles(results, threshold=1e-6, figsize=(12, 8)):
    """Show operator activation/deactivation patterns"""
    weights_per_time = results['weights_per_time']
    pauli_basis = results['pauli_basis']

    # Find lifecycle events
    lifecycles = []
    for i, pauli_str in enumerate(pauli_basis):
        weights = [w[i] for w in weights_per_time]
        active_times = [t for t, w in enumerate(weights) if w > threshold]

        if active_times:
            lifecycles.append({
                'operator': pauli_str,
                'index': i,
                'birth': min(active_times),
                'death': max(active_times),
                'peak_weight': max(weights),
                'peak_time': weights.index(max(weights))
            })

    # Sort by birth time
    lifecycles.sort(key=lambda x: x['birth'])

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Top left: Lifecycle timeline
    n_show = min(30, len(lifecycles))  # Show top 30 for visibility
    for i, lc in enumerate(lifecycles[:n_show]):
        y_pos = i
        axes[0, 0].barh(y_pos, lc['death'] - lc['birth'],
                        left=lc['birth'], height=0.8,
                        color=PASTEL_COLORS[i % len(PASTEL_COLORS)], alpha=0.7)
        axes[0, 0].plot(lc['peak_time'], y_pos, 'ko', markersize=3)

    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Operator Index (by birth time)')
    axes[0, 0].set_title(f'Operator Lifecycles (Top {n_show})')
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Birth/death events
    births = [lc['birth'] for lc in lifecycles]
    deaths = [lc['death'] for lc in lifecycles]

    time_steps = range(len(weights_per_time))
    birth_counts = [births.count(t) for t in time_steps]
    death_counts = [deaths.count(t) for t in time_steps]

    axes[0, 1].plot(time_steps, birth_counts, 'o-', color='#BAFFC9',
                    linewidth=2, markersize=4, label='Births')
    axes[0, 1].plot(time_steps, death_counts, 's-', color='#FFB3BA',
                    linewidth=2, markersize=4, label='Deaths')

    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Number of Events')
    axes[0, 1].set_title('Birth/Death Events')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom left: Peak weight distribution
    peak_weights = [lc['peak_weight'] for lc in lifecycles]
    axes[1, 0].hist(peak_weights, bins=20, color='#FFDFBA', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Peak Weight')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Peak Weight Distribution')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom right: Lifespan distribution
    lifespans = [lc['death'] - lc['birth'] + 1 for lc in lifecycles]
    axes[1, 1].hist(lifespans, bins=20, color='#BAD4FF', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Lifespan (steps)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Operator Lifespan Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print detailed lifecycle information
    print(f"\nOperator Lifecycle Analysis (threshold > {threshold:.0e}):")
    print(f"Total operators with activity: {len(lifecycles)}")

    # Early birds (born at t=0)
    early_birds = [lc for lc in lifecycles if lc['birth'] == 0]
    print(f"\nEarly operators (born at t=0): {len(early_birds)}")
    for lc in early_birds[:5]:
        print(f"  {lc['operator']}: peak={lc['peak_weight']:.3e} at t={lc['peak_time']}")

    # Late bloomers (born after t=2)
    late_bloomers = [lc for lc in lifecycles if lc['birth'] > 2]
    print(f"\nLate bloomers (born after t=2): {len(late_bloomers)}")
    for lc in late_bloomers[:5]:
        print(f"  {lc['operator']}: born t={lc['birth']}, peak={lc['peak_weight']:.3e}")

    # Most persistent (longest lifespan)
    persistent = sorted(lifecycles, key=lambda x: x['death'] - x['birth'], reverse=True)
    print(f"\nMost persistent operators:")
    for lc in persistent[:5]:
        lifespan = lc['death'] - lc['birth'] + 1
        print(f"  {lc['operator']}: {lifespan} steps, peak={lc['peak_weight']:.3e}")

    return lifecycles


def plot_weight_flow_network(results, threshold=1e-6, figsize=(12, 8)):
    """Visualize weight transfer between operator classes"""
    weights_per_time = results['weights_per_time']
    pauli_basis = results['pauli_basis']

    # Define operator classes by locality (number of non-I operators)
    locality_weights = {}
    locality_strings = {}  # Track which strings contribute to each locality
    max_locality = 0

    # First pass: find max locality
    for weights in weights_per_time:
        for i, (pauli_str, weight) in enumerate(zip(pauli_basis, weights)):
            if weight > threshold:
                locality = len(pauli_str) - pauli_str.count('I')
                max_locality = max(max_locality, locality)

    # Handle case where no operators are above threshold
    if max_locality == 0 and not any(any(w > threshold for w in weights) for weights in weights_per_time):
        max_locality = -1  # No active operators

    # Initialize all localities
    localities = list(range(max_locality + 1)) if max_locality >= 0 else []
    for locality in localities:
        locality_weights[locality] = []
        locality_strings[locality] = set()

    # Second pass: collect weights and strings
    for step_idx, weights in enumerate(weights_per_time):
        step_weights = {loc: 0 for loc in localities}

        for i, (pauli_str, weight) in enumerate(zip(pauli_basis, weights)):
            if weight > threshold:
                locality = len(pauli_str) - pauli_str.count('I')
                if locality in step_weights:
                    step_weights[locality] += weight
                    locality_strings[locality].add(pauli_str)

        # Add weights for this time step
        for locality in localities:
            locality_weights[locality].append(step_weights[locality])

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    time_steps = range(len(weights_per_time))

    # Top left: Stacked area plot
    if localities:
        weights_matrix = [locality_weights[loc] for loc in localities]

        axes[0, 0].stackplot(time_steps, *weights_matrix,
                             labels=[f'Locality {loc}' for loc in localities],
                             colors=PASTEL_COLORS[:len(localities)], alpha=0.7)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[0, 0].text(0.5, 0.5, 'No operators above threshold',
                        transform=axes[0, 0].transAxes, ha='center', va='center')

    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Total Weight')
    axes[0, 0].set_title('Weight Distribution by Locality')
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Individual locality evolution
    for locality in localities:
        if locality_weights[locality] and max(locality_weights[locality]) > threshold:
            axes[0, 1].plot(time_steps, locality_weights[locality], 'o-',
                            color=PASTEL_COLORS[locality % len(PASTEL_COLORS)],
                            linewidth=2, markersize=4, label=f'Locality {locality}')

    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Weight')
    axes[0, 1].set_title('Individual Locality Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom left: Locality distribution over time
    locality_counts = []
    for t in range(len(weights_per_time)):
        counts = {loc: 0 for loc in localities}
        for i, (pauli_str, weight) in enumerate(zip(pauli_basis, weights_per_time[t])):
            if weight > threshold:
                locality = len(pauli_str) - pauli_str.count('I')
                if locality in counts:
                    counts[locality] += 1
        locality_counts.append(counts)

    for locality in localities:
        counts = [locality_counts[t][locality] for t in range(len(weights_per_time))]
        if max(counts) > 0:  # Only plot if there are active operators
            axes[1, 0].plot(time_steps, counts, 'o-',
                            color=PASTEL_COLORS[locality % len(PASTEL_COLORS)],
                            linewidth=2, markersize=4, label=f'Locality {locality}')

    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Number of Active Operators')
    axes[1, 0].set_title('Active Operator Count by Locality')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom right: Text summary
    axes[1, 1].axis('off')

    # Add text summary
    if localities:
        summary_text = f"Locality Analysis (threshold > {threshold:.0e}):\n\n"
        for locality in localities:
            n_strings = len(locality_strings[locality])
            max_weight = max(locality_weights[locality]) if locality_weights[locality] else 0
            summary_text += f"Locality {locality}: {n_strings} operators, max weight {max_weight:.3e}\n"
    else:
        summary_text = "No operators above threshold"

    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.show()

    # Print detailed string breakdown
    print(f"\nLocality-based String Analysis (threshold > {threshold:.0e}):")
    if localities:
        for locality in localities:
            strings = sorted(locality_strings[locality])
            if strings:  # Only print if there are strings
                print(f"\nLocality {locality} ({len(strings)} unique strings):")
                for i, s in enumerate(strings[:8]):  # Show first 8
                    print(f"  {s}")
                if len(strings) > 8:
                    print(f"  ... and {len(strings) - 8} more")
    else:
        print("No operators above threshold found.")

    return locality_weights, locality_strings

    # if weight > threshold:
    #     locality = len(pauli_str) - pauli_str.count('I')
    #     max_locality = max(max_locality, locality)
    #     if locality not in step_weights:
    #         step_weights[locality] = 0
    #         step_strings[locality] = []
    #     step_weights[locality] += weight
    #     step_strings[locality].append((pauli_str, weight))
    #
    # for locality in range(max_locality + 1):
    #     if locality not in locality_weights:
    #         locality_weights[locality] = [0] * len(weights_per_time)
    #         locality_strings[locality] = set()
    #
    #     # Update weights for this time step
    # step_idx = len([w for w in weights_per_time if id(w) == id(weights)]) - 1
    # for locality in range(max_locality + 1):
    #     if locality in step_weights:
    #         locality_weights[locality][step_idx] = step_weights[locality]
    #         locality_strings[locality].update([s for s, w in step_strings[locality]])
    #
    # # Ensure all localities have the same length
    # n_steps = len(weights_per_time)
    # for locality in locality_weights:
    #     while len(locality_weights[locality]) < n_steps:
    #         locality_weights[locality].append(0)
    #
    # # Create plots
    # fig, axes = plt.subplots(2, 2, figsize=figsize)
    # time_steps = range(n_steps)
    #
    # # Top left: Stacked area plot
    # if locality_weights:
    #     localities = sorted(locality_weights.keys())
    #     weights_matrix = [locality_weights[loc] for loc in localities]
    #
    #     axes[0, 0].stackplot(time_steps, *weights_matrix,
    #                          labels=[f'Locality {loc}' for loc in localities],
    #                          colors=PASTEL_COLORS[:len(localities)], alpha=0.7)
    #     axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # else:
    #     axes[0, 0].text(0.5, 0.5, 'No operators above threshold',
    #                     transform=axes[0, 0].transAxes, ha='center', va='center')
    #
    # axes[0, 0].set_xlabel('Time Step')
    # axes[0, 0].set_ylabel('Total Weight')
    # axes[0, 0].set_title('Weight Distribution by Locality')
    # axes[0, 0].grid(True, alpha=0.3)
    #
    # # Top right: Individual locality evolution
    # for locality in localities:
    #     if max(locality_weights[locality]) > threshold:
    #         axes[0, 1].plot(time_steps, locality_weights[locality], 'o-',
    #                         color=PASTEL_COLORS[locality % len(PASTEL_COLORS)],
    #                         linewidth=2, markersize=4, label=f'Locality {locality}')
    #
    # axes[0, 1].set_xlabel('Time Step')
    # axes[0, 1].set_ylabel('Weight')
    # axes[0, 1].set_title('Individual Locality Evolution')
    # axes[0, 1].legend()
    # axes[0, 1].grid(True, alpha=0.3)
    #
    # # Bottom left: Locality distribution over time
    # locality_counts = []
    # for t in range(n_steps):
    #     counts = {}
    #     for i, (pauli_str, weight) in enumerate(zip(pauli_basis, weights_per_time[t])):
    #         if weight > threshold:
    #             locality = len(pauli_str) - pauli_str.count('I')
    #             counts[locality] = counts.get(locality, 0) + 1
    #     locality_counts.append(counts)
    #
    # for locality in localities:
    #     counts = [locality_counts[t].get(locality, 0) for t in range(n_steps)]
    #     axes[1, 0].plot(time_steps, counts, 'o-',
    #                     color=PASTEL_COLORS[locality % len(PASTEL_COLORS)],
    #                     linewidth=2, markersize=4, label=f'Locality {locality}')
    #
    # axes[1, 0].set_xlabel('Time Step')
    # axes[1, 0].set_ylabel('Number of Active Operators')
    # axes[1, 0].set_title('Active Operator Count by Locality')
    # axes[1, 0].legend()
    # axes[1, 0].grid(True, alpha=0.3)
    #
    # # Bottom right: Clear for text summary
    # axes[1, 1].axis('off')
    #
    # # Add text summary
    # summary_text = f"Locality Analysis (threshold > {threshold:.0e}):\n\n"
    # for locality in sorted(localities):
    #     n_strings = len(locality_strings[locality])
    #     max_weight = max(locality_weights[locality])
    #     summary_text += f"Locality {locality}: {n_strings} operators, max weight {max_weight:.3e}\n"
    #
    # axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
    #                 fontsize=10, verticalalignment='top', fontfamily='monospace')
    #
    # plt.tight_layout()
    # plt.show()

# def local_analysis(localities,threshold)
#     print(f"\nLocality-based String Analysis (threshold > {threshold:.0e}):")
#     for locality in sorted(localities):
#         strings = sorted(locality_strings[locality])
#         print(f"\nLocality {locality} ({len(strings)} unique strings):")
#         for i, s in enumerate(strings[:8]):  # Show first 8
#             print(f"  {s}")
#         if len(strings) > 8:
#             print(f"  ... and {len(strings) - 8} more")
#
#     return locality_weights, locality_strings


def compute_exploration_metrics(results, threshold=1e-6):
    """Compute Active Pauli Growth Rate, Weight Diffusion, Weight Concentration"""
    weights_per_time = results['weights_per_time']

    # Active Pauli Growth Rate
    active_counts = [sum(1 for w in weights if w > threshold) for weights in weights_per_time]
    cumulative_active = np.cumsum(active_counts)

    # Weight Diffusion (Shannon entropy)
    entropies = []
    for weights in weights_per_time:
        w = np.array(weights)
        w_norm = w / np.sum(w)
        w_norm = w_norm[w_norm > 0]  # Remove zeros
        if len(w_norm) > 0:
            entropy = -np.sum(w_norm * np.log(w_norm))
        else:
            entropy = 0
        entropies.append(entropy)

    # Weight Concentration (weight in top 10%)
    concentrations = []
    for weights in weights_per_time:
        w = np.array(weights)
        if np.sum(w) > 0:
            n_top = max(1, len(w) // 10)
            top_weights = np.sort(w)[-n_top:]
            concentration = np.sum(top_weights) / np.sum(w)
        else:
            concentration = 0
        concentrations.append(concentration)

    return {
        'active_counts': active_counts,
        'cumulative_active': cumulative_active,
        'entropies': entropies,
        'concentrations': concentrations
    }


def plot_exploration_metrics(results, threshold=1e-6, figsize=(12, 9)):
    """Plot exploration metrics"""
    metrics = compute_exploration_metrics(results, threshold)
    time_steps = range(len(metrics['active_counts']))

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Active Pauli Growth Rate
    axes[0, 0].plot(time_steps, metrics['active_counts'], 'o-',
                    color='#FFB3BA', linewidth=2, markersize=4, label='Per Step')
    axes[0, 0].plot(time_steps, metrics['cumulative_active'], 's-',
                    color='#BAFFC9', linewidth=2, markersize=4, label='Cumulative')
    axes[0, 0].set_ylabel('Active Operators')
    axes[0, 0].set_title('Active Pauli Growth Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Weight Diffusion (Entropy)
    axes[0, 1].plot(time_steps, metrics['entropies'], '^-',
                    color='#BAE1FF', linewidth=2, markersize=4)
    axes[0, 1].set_ylabel('Shannon Entropy')
    axes[0, 1].set_title('Weight Diffusion Rate')
    axes[0, 1].grid(True, alpha=0.3)

    # Weight Concentration
    axes[1, 0].plot(time_steps, metrics['concentrations'], 'D-',
                    color='#E1BAFF', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Weight in Top 10%')
    axes[1, 0].set_title('Weight Concentration')
    axes[1, 0].grid(True, alpha=0.3)

    # Combined normalized view
    # Normalize each metric to [0,1] for comparison
    max_active = max(metrics['active_counts']) if max(metrics['active_counts']) > 0 else 1
    max_entropy = max(metrics['entropies']) if max(metrics['entropies']) > 0 else 1

    norm_active = np.array(metrics['active_counts']) / max_active
    norm_entropy = np.array(metrics['entropies']) / max_entropy
    norm_concentration = 1 - np.array(metrics['concentrations'])  # Invert for comparison

    axes[1, 1].plot(time_steps, norm_active, 'o-',
                    color='#FFB3BA', linewidth=2, markersize=3, label='Active Growth')
    axes[1, 1].plot(time_steps, norm_entropy, '^-',
                    color='#BAE1FF', linewidth=2, markersize=3, label='Entropy')
    axes[1, 1].plot(time_steps, norm_concentration, 'D-',
                    color='#E1BAFF', linewidth=2, markersize=3, label='Deconcentration')

    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Normalized Value')
    axes[1, 1].set_title('Exploration Metrics (Normalized)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return metrics
def time_average(data, window=50):
    """Moving time average for plotting."""
    return [np.mean(data[max(0, i - window + 1):i + 1]) for i in range(len(data))]

def plot_publication_2panel(
    labels: List[str],
    krylov_complexities: List[np.ndarray],
    active_operators: List[np.ndarray],
    pastel_colors=None,
    line_styles=None,
):
    """Panel 1: Time-averaged Krylov complexity and active operator number, with colors and line styles."""
    import matplotlib.pyplot as plt
    if pastel_colors is None:
        pastel_colors = PASTEL_COLORS
    if line_styles is None:
        line_styles = ['-'] * len(labels)
    krylov_avg = [time_average(kc) for kc in krylov_complexities]
    active_avg = [time_average(ao) for ao in active_operators]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for i, label in enumerate(labels):
        color = pastel_colors[i % len(pastel_colors)]
        ls = line_styles[i % len(line_styles)]
        axes[0].plot(
            krylov_avg[i], label=label, color=color, linestyle=ls, linewidth=2, alpha=0.85
        )
        axes[1].plot(
            active_avg[i], label=label, color=color, linestyle=ls, linewidth=2, alpha=0.85
        )
    axes[0].set_title('Time-Averaged Krylov Complexity')
    axes[1].set_title('Time-Averaged Active Operators')
    axes[0].set_xlabel('Time step',fontsize=15)
    axes[1].set_xlabel('Time step',fontsize=15)
    axes[0].set_ylabel('Krylov complexity',fontsize=15)
    axes[1].set_ylabel('Active operators',fontsize=15)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    #for ax in axes:
    #    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    #    ax.grid(True)
    plt.tight_layout()
    plt.show()

def hamming_weight(pauli_string: str) -> int:
    """Count number of non-identity characters in a Pauli string (works for IXYZ and S+/S-)."""
    # For S+/S-, treat S+ S- and Z as non-identity, for IXYZ, all except I
    non_id_count = 0
    i = 0
    while i < len(pauli_string):
        if pauli_string[i] == 'I':
            i += 1
        elif pauli_string[i] == 'S':
            # Expect S+ or S-
            if pauli_string[i:i + 2] in ['S+', 'S-']:
                non_id_count += 1
                i += 2
            else:
                i += 1  # If malformed, skip
        else:
            non_id_count += 1
            i += 1
    return non_id_count

def compute_gs_complexity(gs_coeffs):
    gs_probs = np.abs(gs_coeffs) ** 2
    gs_complexity = np.sum(gs_probs * np.arange(gs_probs.shape[1]), axis=1)
    return gs_complexity

def compute_weight_distribution_by_hamming(
    pauli_basis: List[str], weights: np.ndarray, max_body: int = 6
) -> np.ndarray:
    """
    Returns an array of shape (len(weights), max_body)
    Each column is the total weight for that hamming weight (body) at a time step.
    """
    n_time = len(weights)
    result = np.zeros((n_time, max_body))
    hamming_weights = [hamming_weight(ps) for ps in pauli_basis]
    for t in range(n_time):
        for i, h in enumerate(hamming_weights):
            if 1 <= h <= max_body:
                result[t, h-1] += weights[t][i]
    return result

def plot_weight_distribution_stacked(pauli_basis, weights_per_time, rule_label, pastel_colors=None):
    """Plot stacked bar chart of weight distribution by hamming weight for one rule."""
    if pastel_colors is None:
        pastel_colors = PASTEL_COLORS
    max_body = min(6, max(hamming_weight(ps) for ps in pauli_basis))
    weight_dist = compute_weight_distribution_by_hamming(pauli_basis, weights_per_time, max_body)
    timesteps = np.arange(weight_dist.shape[0])
    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros(weight_dist.shape[0])
    labels = [f"{i+1}-body" for i in range(max_body)]
    for i in range(max_body):
        ax.bar(
            timesteps,
            weight_dist[:, i],
            bottom=bottoms,
            color=pastel_colors[i % len(pastel_colors)],
            label=labels[i],
            width=1.0
        )
        bottoms += weight_dist[:, i]
    ax.set_title(f"Weight Distribution by Hamming Weight: {rule_label}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Total weight")
    ax.legend()
    plt.tight_layout()
    plt.show()

def auto_subplot_layout(n):
    """Find rows, cols for n subplots to fit all in a grid."""
    import math
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols

def plot_all_weight_distributions(rules, all_pauli_basis, all_weights, pastel_colors=None):
    """Plot all rules' weight distributions in an optimized subplot layout."""
    n_rules = len(rules)
    rows, cols = auto_subplot_layout(n_rules)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*3))
    if n_rules == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()
    for idx, (rule, pauli_basis, weights_per_time) in enumerate(zip(rules, all_pauli_basis, all_weights)):
        max_body = min(6, max(hamming_weight(ps) for ps in pauli_basis))
        weight_dist = compute_weight_distribution_by_hamming(pauli_basis, weights_per_time, max_body)
        timesteps = np.arange(weight_dist.shape[0])
        bottoms = np.zeros(weight_dist.shape[0])
        labels = [f"{i+1}-body" for i in range(max_body)]
        for i in range(max_body):
            axes[idx].bar(
                timesteps,
                weight_dist[:, i],
                bottom=bottoms,
                color=New_pastel[i % len(New_pastel)],
                label=labels[i],
                width=1.0
            )
            bottoms += weight_dist[:, i]
        axes[idx].set_title(f"{rule}")
        axes[0].set_xlabel("Time step", fontsize=15)
        #axes[15].set_xlabel("Time step",fontsize=15)
        #axes[16].set_xlabel("Time step", fontsize=15)
        #axes[17].set_xlabel("Time step", fontsize=15)
        #axes[18].set_xlabel("Time step", fontsize=15)
        #axes[19].set_xlabel("Time step", fontsize=15)
        axes[0].set_ylabel("Total weight",fontsize=15)
        #axes[5].set_ylabel("Total weight",fontsize=15)
        #axes[8].set_ylabel("Total weight",fontsize=15)
        #axes[8].set_ylabel("Total weight",fontsize=15)

        axes[idx].set_ylim(0, 1.05)
        axes[0].legend(fontsize=10)
    # Hide unused axes if any
    for a in axes[n_rules:]:
        a.axis('off')
    plt.tight_layout()
    plt.show()
from matplotlib.colors import LogNorm

# python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_pauli_gs_heatmaps_all_rules(rules, n_qubits, initial_operator, time_steps, theta):
    pauli_weights_all = []
    gs_weights_all = []
    rule_labels = []

    for rule in rules:
        # --- Generate unitaries for each rule ---
        if rule == "random":
            pattern_string = ''.join(random.choice('gj') for _ in range(time_steps))
            unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
        elif rule == "brickwork":
            unitaries = create_brickwork_unitaries(n_qubits, theta)
        else:
            filename = f"../Thesis_plots/C2/config_list_CS3_{rule}_steps_10_500.csv"
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for i, row in enumerate(reader):
                    if i == 0:
                        pattern_string = ''.join(cell.strip().replace('0', 'j').replace('1', 'g') for cell in row[37:])
                        print(f"Pat is {pattern_string}")
                        unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
                        break

        # --- Run quantum circuit analysis ---
        results = analyze_operator_spreading(
            n_qubits, initial_operator, unitaries, time_steps, symmetry="U1"
        )
        weights_per_time = results['weights_per_time']
        pauli_weights = np.array(weights_per_time).T  # shape: (n_pauli, n_time)
        pauli_weights_all.append(pauli_weights)
        rule_labels.append(rule)

        # --- GS analysis ---
        analyzer = QuantumOperatorAnalyzer(n_qubits, symmetry="U1")
        orthogonalizer = GramSchmidtOrthogonalizer(analyzer)
        gs_coeffs = orthogonalizer.express_in_orthogonal_basis(
            results['evolved_operators'], orthogonalizer.robust_modified_gram_schmidt(results['evolved_operators'])
        )
        gs_weights = np.abs(np.array(gs_coeffs)) ** 2  # shape: (n_time, n_gs)
        gs_weights = gs_weights.T  # shape: (n_gs, n_time)
        gs_weights_all.append(gs_weights)

    n_rules = len(rules)
    fig, axes = plt.subplots(n_rules, 2, figsize=(14, 4 * n_rules))
    if n_rules == 1:
        axes = np.array([axes])

    for idx, rule in enumerate(rules):
        # Pauli weights heatmap
        im1 = axes[idx, 0].imshow(pauli_weights_all[idx], aspect='auto', cmap='plasma',
                                  norm=LogNorm(vmax=1, vmin=1e-8), origin='lower')
        axes[idx, 0].set_title(f"{rule}: Pauli Operator Spreading")
        axes[idx, 0].set_xlabel("Time Step")
        axes[idx, 0].set_ylabel("Pauli Index")
        plt.colorbar(im1, ax=axes[idx, 0], shrink=0.7, label="Weight (log)")

        # GS operator heatmap
        im2 = axes[idx, 1].imshow(gs_weights_all[idx], aspect='auto', cmap='YlGnBu',
                                  norm=LogNorm(vmax=1, vmin=1e-8), origin='lower')
        axes[idx, 1].set_title(f"{rule}: GS Operator Heatmap")
        axes[idx, 1].set_xlabel("Time Step")
        axes[idx, 1].set_ylabel("GS Index")
        plt.colorbar(im2, ax=axes[idx, 1], shrink=0.7, label="|a_n(t)|² (log)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # # Parameters
    n_qubits = 6


    def S1(N):
        """Long blocks near-swap (m=8 blocks)"""
        base_block = "j" * 8 + "g" * 8
        repeats = N // len(base_block)
        remainder = N % len(base_block)
        seq = base_block * repeats
        seq += "j" * remainder  # pad with j if needed
        return seq[:N]


    def S2(N):
        """Echo / identity cycle for A-bonds (7+8=15)"""
        base_block = "j" * 7 + "g" * 8
        repeats = N // len(base_block)
        remainder = N % len(base_block)
        seq = base_block * repeats
        seq += "j" * (remainder if remainder <= 7 else 7) + "g" * (remainder - 7 if remainder > 7 else 0)
        return seq[:N]


    def S3(N):
        """Explicit echo with small g paddings (A identity every cycle)"""
        base_block = "j" * 7 + "g" * 1 + "j" * 8 + "g" * 1  # cycle length 17
        repeats = N // len(base_block)
        remainder = N % len(base_block)
        seq = base_block * repeats
        # Pad remainder: split leftover as best as possible
        if remainder > 0:
            if remainder <= 7:
                seq += "j" * remainder
            elif remainder <= 8:
                seq += "j" * 7 + "g" * (remainder - 7)
            else:
                seq += "j" * 7 + "g" * 1 + "j" * (remainder - 8)
        return seq[:N]


    def S4(N):
        """Staggered run-lengths (A near-swap, B moderate)"""
        base_block = "j" * 8 + "g" * 4
        repeats = N // len(base_block)
        remainder = N % len(base_block)
        seq = base_block * repeats
        seq += "j" * remainder  # pad with j
        return seq[:N]


    def S5(N):
        """Correlated near-swap bias (many m=7 runs)"""
        base_block = "j" * 7 + "g" * 7
        repeats = N // len(base_block)
        remainder = N % len(base_block)
        seq = base_block * repeats
        seq += "j" * remainder
        return seq[:N]


    def S6(N):
        """Micro-echo / short-pattern interference"""
        base_block = "jggjjggj"
        repeats = N // len(base_block)
        remainder = N % len(base_block)
        seq = base_block * repeats
        seq += base_block[:remainder]
        return seq[:N]


    # ---- Additional variants ----

    def Frobenium(N):
        """Highly frustrated alternating short runs"""
        base_block = "j" * 3 + "g" * 5 + "j" * 5 + "g" * 3
        repeats = N // len(base_block)
        remainder = N % len(base_block)
        seq = base_block * repeats + base_block[:remainder]
        return seq[:N]


    def generate_palindromic_sequence(N):
        """Generate a sequence of N characters using palindromes of length 10"""
        import random

        def single_palindrome():
            """Generate one palindrome of length 10"""
            prefix_len = 5
            prefix = ''.join(random.choice(['j', 'g']) for _ in range(prefix_len))
            return prefix + prefix[::-1]  # 5 chars + 5 chars reversed = 10

        num_palindromes = N // 10
        remainder = N % 10

        sequence = ""
        for _ in range(num_palindromes):
            sequence += single_palindrome()

        # Handle remainder by truncating a final palindrome if needed
        if remainder > 0:
            sequence += single_palindrome()[:remainder]

        return sequence[:N]  # Ensure exact length N


    def Mirror1(N):
        """Mirror sequence with alternating runs"""
        base_block = "j" * 4 + "g" * 4
        half = N // 2
        seq = base_block * (half // len(base_block))
        seq = seq[:half]
        seq += seq[::-1]  # mirror second half
        return seq[:N]


    def Mirror2(N):
        """Long blocks mirrored"""
        base_block = "j" * 8 + "g" * 8
        half = N // 2
        seq = base_block * (half // len(base_block))
        seq = seq[:half]
        seq += seq[::-1]
        return seq[:N]


    def Frustrated(N):
        """Mix of near-swap and identity cycles with irregular lengths"""
        blocks = ["j" * 7 + "g" * 8, "j" * 8 + "g" * 7, "j" * 6 + "g" * 6]
        seq = ""
        i = 0
        while len(seq) < N:
            seq += blocks[i % len(blocks)]
            i += 1
        return seq[:N]


    def Frustrated2(N):
        """
        Generate a string of length N using blocks of the form
        'j'*j1 + 'g'*g + 'j'*j2, cycling through a fixed distribution.
        """
        blocks = [
            (9, 8, 6), (5, 7, 2), (4, 2, 8), (1, 5, 8), (1, 9, 6), (4, 2, 3), (9, 6, 3), (7, 5, 5),
            (3, 2, 6), (7, 3, 7), (4, 5, 9), (1, 2, 6), (5, 6, 8), (1, 1, 5), (8, 3, 1), (9, 8, 3),
            (1, 2, 5), (6, 2, 9), (9, 7, 2), (4, 2, 6), (4, 1, 1), (5, 6, 5), (4, 2, 4), (8, 7, 7),
            (6, 9, 4), (1, 1, 9), (4, 7, 8), (2, 3, 3), (7, 6, 7), (7, 1, 1), (8, 2, 2), (8, 8, 8),
            (4, 5, 3), (7, 8, 3), (7, 2, 1), (3, 4, 1), (3, 5, 1), (2, 4, 1), (1, 5, 9), (3, 7, 6),
            (7, 7, 9), (9, 3, 7), (3, 3, 6), (2, 7, 7), (5, 7, 1), (4, 4, 2), (1, 8, 3), (7, 8, 8),
            (2, 2, 7), (9, 4, 3), (1, 1, 6), (1, 4, 6), (3, 5, 2), (3, 2, 5), (6, 6, 3), (2, 5, 5),
            (8, 8, 5), (6, 2, 8), (2, 9, 8), (1, 1, 6), (7, 6, 6), (6, 8, 9), (4, 4, 4), (9, 5, 8),
            (2, 4, 9), (2, 4, 2), (3, 4, 3), (1, 7, 9), (4, 1, 4), (7, 8, 1), (2, 1, 7), (3, 9, 1),
            (9, 2, 5), (8, 4, 9), (8, 6, 6), (5, 3, 6), (4, 6, 8), (3, 2, 2), (9, 3, 4), (5, 7, 4),
            (8, 6, 2), (1, 3, 2), (5, 2, 1), (6, 3, 7), (2, 4, 6), (2, 1, 3), (7, 2, 1), (2, 2, 1),
            (5, 5, 6), (6, 1, 2), (1, 6, 6), (4, 6, 6), (3, 4, 7), (7, 8, 3), (3, 2, 4), (5, 2, 8),
            (2, 1, 5), (1, 6, 7), (3, 1, 5), (7, 4, 7)
        ]
        seq = ""
        i = 0
        while len(seq) < N:
            j1, g, j2 = blocks[i % len(blocks)]
            seq += "j" * j1 + "g" * g + "j" * j2
            i += 1
        return seq[:N]


    def Fibonacci(N):
        """
        Fibonacci-inspired gate sequence:
        - Use run lengths following Fibonacci numbers: 1,1,2,3,5,8,13,...
        - Alternate j and g runs.
        - Truncate to length N.
        """
        # Generate Fibonacci numbers until sum exceeds N
        fibs = [1, 1]
        while sum(fibs) < N:
            fibs.append(fibs[-1] + fibs[-2])

        seq = ""
        toggle = True  # start with j
        for f in fibs:
            block = "j" * f if toggle else "g" * f
            seq += block
            toggle = not toggle
            if len(seq) >= N:
                break
        return seq[:N]


    def AlternatingJG10(N):
        """
        Generate a sequence of length N with alternating 'j' and 'g' blocks,
        such that any snapshot of 3 consecutive blocks (j, g, j) sums to 10.
        """
        # Precompute all valid (j1, g, j2) with j1, g, j2 >= 1 and j1+g+j2=10
        triplets = []
        for j1 in range(1, 10):
            for g in range(1, 10):
                j2 = 10 - j1 - g
                if j2 >= 1:
                    triplets.append((j1, g, j2))

        seq = ""
        i = 0
        while len(seq) < N:
            j1, g, j2 = triplets[i % len(triplets)]
            seq += "j" * j1 + "g" * g + "j" * j2
            i += 1
        return seq[:N]


    def repeated_jg_pattern(N: int, p_peak: int, g_len: int = 1) -> str:
        """
        Generate a string of length N with repeated blocks of the form:
        'j'*n1 + 'g'*g_len + 'j'*n2, where n1 + n2 = 15, n1 peaked at p_peak.
        """
        seq = ""
        while len(seq) < N:
            n1 = int(np.round(np.random.normal(loc=p_peak, scale=2)))
            n1 = max(0, min(15, n1))
            n2 = 15 - n1
            block = 'j' * n1 + 'g' * g_len + 'j' * n2
            seq += block
        return seq[:N]


    def oscillatory_jg_pattern(N: int, theta_fix: float, A: float, theta: float) -> str:
        """
        Generate a string of length N with alternating 'j' and 'g' blocks.
        Each block's length is given by: int(round(theta_fix + A * sin(theta * t)))
        where t is the block index (layer).
        """
        seq = ""
        t = 0
        toggle = True  # Start with 'j'
        while len(seq) < N:
            block_len = int(round(theta_fix + A * np.sin(theta * t)))
            block_len = max(1, block_len)  # Ensure at least 1
            block = ('j' if toggle else 'g') * block_len
            seq += block
            toggle = not toggle
            t += 1
        return seq[:N]

    def random_jg_alternating_string(total_j: int, total_g: int, peak_j: int = 7, peak_g: int = 7) -> str:
        """
        Generate a string of the form: 'j'*n1 + 'g'*m1 + 'j'*n2 + 'g'*m2,
        where n1 + n2 = total_j, m1 + m2 = total_g.
        n1, n2, m1, m2 are chosen randomly, peaked at peak_j and peak_g.
        """
        # Randomly split total_j into n1 and n2, peaked at peak_j
        n1 = int(np.round(np.random.normal(loc=peak_j, scale=2)))
        n1 = max(0, min(total_j, n1))
        n2 = total_j - n1

        # Randomly split total_g into m1 and m2, peaked at peak_g
        m1 = int(np.round(np.random.normal(loc=peak_g, scale=2)))
        m1 = max(0, min(total_g, m1))
        m2 = total_g - m1

        return 'j' * n1 + 'g' * m1 + 'j' * n2 + 'g' * m2


    def random_gate_string(N, num_j_layers=None, num_g_layers=None, brickwork_start='j'):
        """
        Place blocks of j and g randomly, split into user-specified layers,
        and fill leftover with brickwork.

        Parameters:
            N: total desired string length
            num_j_layers: number of segments to split j blocks (optional)
            num_g_layers: number of segments to split g blocks (optional)
            brickwork_start: 'j' or 'g'

        Returns:
            gate_string: the final string
            details: dict with block details
        """
        # Step 1: Find largest multiple of 15
        max_mult_15 = (N // 15) * 15
        leftover = N - max_mult_15
        total_blocks = N // 15

        # Step 2: Randomly split blocks between j and g
        # k: number of 15-j blocks, m: number of 15-g blocks
        k = np.random.randint(0, total_blocks + 1)
        m = total_blocks - k

        # Step 3: Split j's and g's into layers
        if num_j_layers is None:
            num_j_layers = 1 if k == 0 else min(k, 3)
        if num_g_layers is None:
            num_g_layers = 1 if m == 0 else min(m, 3)

        # Random multinomial splits
        j_layer_sizes = np.random.multinomial(15 * k, [1 / num_j_layers] * num_j_layers) if k else []
        g_layer_sizes = np.random.multinomial(15 * m, [1 / num_g_layers] * num_g_layers) if m else []

        # Step 4: Create blocks and shuffle
        blocks = ['j' * size for size in j_layer_sizes] + ['g' * size for size in g_layer_sizes]
        np.random.shuffle(blocks)
        main_str = ''.join(blocks)

        # Step 5: Brickwork for leftover
        brickwork = ''
        pattern = ['j', 'g']
        idx = 0 if brickwork_start == 'j' else 1
        for i in range(leftover):
            brickwork += pattern[(idx + i) % 2]

        final_str = main_str + brickwork
        details = {
            'j_layer_sizes': j_layer_sizes,
            'g_layer_sizes': g_layer_sizes,
            'k': k,
            'm': m,
            'leftover': leftover,
            'brickwork_pattern': brickwork
        }
        return final_str, details


    time_steps = 10
    salt3= random_gate_string(1000, num_j_layers=640, num_g_layers=350, brickwork_start='j')
    salt1 = random_jg_alternating_string(total_j=15, total_g=15, peak_j=8, peak_g=8)
    salt2 = random_jg_alternating_string(total_j=30, total_g=30, peak_j=15, peak_g=15)
    stringjg = repeated_jg_pattern(1000, p_peak=15,g_len=2)
    altstring=AlternatingJG10(1000)
    thetastring=oscillatory_jg_pattern(1000, theta_fix=0.3, A=0.2, theta=np.pi/4)
    print(stringjg[500:])
    S1=S1(time_steps)
    S2 = S2(time_steps)
    S3 = S3(time_steps)
    S4 = S4(time_steps)
    S5 = S5(time_steps)
    S6 = S6(time_steps)
    Fib=Fibonacci(time_steps)
    Frob=Frustrated(time_steps)
    Frus=Frustrated2(time_steps)
    Pal=generate_palindromic_sequence(time_steps)
    Mir1=Mirror1(time_steps)
    Mir2=Mirror2(time_steps)

    #rules=["brickwork","random",altstring]
           #S1, S2, S3, S4, S5, S6,Fib, Frob,Pal, Frus,Mir1, Mir2]
    #"rule_0_2_2_1",, "rule_1_2_0_0","rule_0_1_0_0",, "rule_1_0_3_3", "rule_1_0_1_3"

    #rules= ["brickwork","random","rule_0_2_1_0","rule_0_3_1_1","rule_1_0_1_3","rule_0_3_1_1", "rule_1_2_3_3","rule_1_3_1_3","rule_1_3_3_2","rule_2_0_0_3","rule_2_1_2_3","rule_2_2_0_0", "rule_2_3_0_1", "rule_2_3_3_0", "rule_3_0_3_2", "rule_3_1_0_2", "rule_3_1_1_2","rule_3_1_2_2","rule_3_1_3_2","rule_3_3_1_1"]#,Mir1,Mir2]
    #rules= ["brickwork","random","rule_2_0_0_3", "rule_1_0_1_3", "rule_2_3_0_1","rule_0_2_1_0","rule_0_3_1_1", "rule_1_3_3_2", "rule_3_1_1_2", "rule_3_0_3_2", "rule_3_3_1_1"]
    #rules= ["brickwork","random","R1","R2","R3","R4","R5"]
    #,stringjg,altstring,thetastring,salt1,salt2,salt3]
    #["brickwork","random","rule_0_2_1_0","rule_0_3_1_1","rule_1_0_1_3","rule_0_3_1_1", "rule_1_2_3_3","rule_1_3_1_3","rule_1_3_3_2","rule_2_0_0_3","rule_2_1_2_3","rule_2_2_0_0", "rule_2_3_0_1", "rule_2_3_3_0", "rule_3_0_3_2", "rule_3_1_0_2", "rule_3_1_1_2","rule_3_1_2_2","rule_3_1_3_2","rule_3_3_1_1"]
               #,"rule_0_3_1_1","rule_1_0_1_3","rule_1_0_3_3","rule_1_2_0_0","rule_1_2_3_3",
               #"rule_1_3_1_3","rule_1_3_3_2","rule_2_0_0_0","rule_2_0_0_3","rule_2_1_2_3",
               #"rule_2_2_0_0","rule_2_3_0_1","rule_2_3_3_0","rule_3_0_3_2","rule_3_1_0_2","rule_3_1_1_2",
               #"rule_3_1_2_2","rule_3_1_3_2","rule_3_3_1_1"]

    theta = 0.07
    initial_operator = "ZIIIII"
    N_TRIALS=100
    time_steps = 200
    n_qubits = 6
    #rules = ["brickwork", "random", "R1", "R2", "R3","R4","R5"]
    #rules= ["brickwork","random","rule_2_0_0_3", "rule_1_0_1_3", "rule_2_3_0_1","rule_0_2_1_0","rule_0_3_1_1", "rule_1_3_3_2", "rule_3_1_1_2", "rule_3_0_3_2", "rule_3_3_1_1"]
    rules= ["brickwork",Pal]
    #plot_pauli_gs_heatmaps_all_rules(rules, n_qubits, initial_operator, time_steps, theta)

    # krylov_complexities_ensemble = {rule: [] for rule in rules}
    # active_operators_ensemble = {rule: [] for rule in rules}
    # all_pauli_basis_ensemble = {rule: [] for rule in rules}
    # all_weights_per_time_ensemble = {rule: [] for rule in rules}

    # for rule in rules:
    #     n_trials = N_TRIALS if rule == "random" or not (rule == "brickwork") else 1
    #
    #     for trial_index in range(n_trials):
    #         if rule == "random":
    #             pattern_string = ''.join(random.choice('gj') for _ in range(time_steps))
    #             unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
    #         elif rule == "brickwork":
    #             unitaries = create_brickwork_unitaries(n_qubits, theta)
    #         else:
    #             # For CSV-based rule
    #             filename = f"../Thesis_plots/C2/config_list_CS3_{rule}_steps_10_500.csv"
    #             with open(filename, 'r') as f:
    #                 reader = csv.reader(f)
    #                 next(reader)  # Skip header
    #                 for i, row in enumerate(reader):
    #                     if i == trial_index:
    #                         pattern_string = ''.join(
    #                             cell.strip().replace('0', 'j').replace('1', 'g') for cell in row[1:]
    #                         )
    #                         unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
    #                         break
    #
    #         # -- Run the quantum circuit analysis --
    #         results = analyze_operator_spreading(
    #             n_qubits, initial_operator, unitaries, time_steps, symmetry="U1"
    #         )
    #         gs_results = orthogonalize_evolved_operators(
    #             results['evolved_operators'],
    #             QuantumOperatorAnalyzer(n_qubits, symmetry="U1")
    #         )
    #         weights_per_time = results['weights_per_time']
    #         norm0 = np.sum(weights_per_time[0])
    #         norms = [np.sum(w) for w in weights_per_time]
    #         active = [np.sum(np.array(w) > WEIGHT_THRESHOLD) for w in weights_per_time]
    #         analyzer = QuantumOperatorAnalyzer(n_qubits, symmetry="U1")
    #         orthogonalizer = GramSchmidtOrthogonalizer(analyzer)
    #         basis, alpha_coeffs, beta_coeffs = orthogonalizer.layer_by_layer_lanczos(
    #             results['evolved_operators'], verbose=False
    #         )
    #         krylov = orthogonalizer.compute_krylov_complexity(alpha_coeffs, beta_coeffs, time_steps)
    #
    #         # -- Store for averaging --
    #         krylov_complexities_ensemble[rule].append(np.asarray(krylov))
    #         active_operators_ensemble[rule].append(np.asarray(active))
    #         all_pauli_basis_ensemble[rule].append(results['pauli_basis'])
    #         all_weights_per_time_ensemble[rule].append(results['weights_per_time'])
    #
    # # -- Compute ensemble averages --
    # krylov_complexities = []
    # active_operators = []
    # all_pauli_basis = []
    # all_weights_per_time = []
    #
    # for rule in rules:
    #     krylov_arr = np.stack(krylov_complexities_ensemble[rule])
    #     active_arr = np.stack(active_operators_ensemble[rule])
    #     avg_krylov = np.mean(krylov_arr, axis=0)
    #     avg_active = np.mean(active_arr, axis=0)
    #     krylov_complexities.append(avg_krylov)
    #     active_operators.append(avg_active)
    #
    #     # For pauli_basis and weights_per_time, just use the first trial (structure doesn't change)
    #     all_pauli_basis.append(all_pauli_basis_ensemble[rule][0])
    #     # Average weights per time
    #     weights_stacked = np.stack([np.array(w) for w in all_weights_per_time_ensemble[rule]])
    #     avg_weights_per_time = np.mean(weights_stacked, axis=0)
    #     all_weights_per_time.append(avg_weights_per_time)
    #
    # # --- PUBLICATION STYLE PLOTS ---
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    #           '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
    #           '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5']
    # line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--',
    #                '-.', ':', '-']
    #
    # plot_publication_2panel(
    #     rules, krylov_complexities, active_operators,
    #     pastel_colors=colors,
    #     line_styles=line_styles
    # )
    # plot_all_weight_distributions(
    #     rules, all_pauli_basis, all_weights_per_time, pastel_colors=colors
    # )

    ############################ Use for N trials with out Lanczos, with GS #####################################
    # krylov_complexities_ensemble = {rule: [] for rule in rules}
    # active_operators_ensemble = {rule: [] for rule in rules}
    # all_pauli_basis_ensemble = {rule: [] for rule in rules}
    # all_weights_per_time_ensemble = {rule: [] for rule in rules}

    # for rule in rules:
    #     n_trials = N_TRIALS if rule == "random" or not (rule == "brickwork") else 1
    #
    #     for trial_index in range(n_trials):
    #         if rule == "random":
    #             pattern_string = ''.join(random.choice('gj') for _ in range(time_steps))
    #             unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
    #         elif rule == "brickwork":
    #             unitaries = create_brickwork_unitaries(n_qubits, theta)
    #         else:
    #             # For CSV-based rule
    #             filename = f"../Thesis_plots/C2/config_list_CS3_{rule}_steps_10_500.csv"
    #             with open(filename, 'r') as f:
    #                 reader = csv.reader(f)
    #                 next(reader)  # Skip header
    #                 for i, row in enumerate(reader):
    #                     if i == trial_index:
    #                         pattern_string = ''.join(
    #                             cell.strip().replace('0', 'j').replace('1', 'g') for cell in row[1:]
    #                         )
    #                         unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
    #                         break
    #
    #         # -- Run the quantum circuit analysis --
    #         results = analyze_operator_spreading(
    #             n_qubits, initial_operator, unitaries, time_steps, symmetry="U1"
    #         )
    #         gs_results = orthogonalize_evolved_operators(
    #             results['evolved_operators'],
    #             QuantumOperatorAnalyzer(n_qubits, symmetry="U1")
    #         )
    #         weights_per_time = results['weights_per_time']
    #         norm0 = np.sum(weights_per_time[0])
    #         norms = [np.sum(w) for w in weights_per_time]
    #         active = [np.sum(np.array(w) > WEIGHT_THRESHOLD) for w in weights_per_time]
    #         analyzer = QuantumOperatorAnalyzer(n_qubits, symmetry="U1")
    #         orthogonalizer = GramSchmidtOrthogonalizer(analyzer)
    #         basis, alpha_coeffs, beta_coeffs = orthogonalizer.layer_by_layer_lanczos(
    #             results['evolved_operators'], verbose=False
    #         )
    #         krylov = orthogonalizer.compute_krylov_complexity(alpha_coeffs, beta_coeffs, time_steps)
    #
    #         # -- Store for averaging --
    #         krylov_complexities_ensemble[rule].append(np.asarray(krylov))
    #         active_operators_ensemble[rule].append(np.asarray(active))
    #         all_pauli_basis_ensemble[rule].append(results['pauli_basis'])
    #         all_weights_per_time_ensemble[rule].append(results['weights_per_time'])
    #
    # # -- Compute ensemble averages --
    # krylov_complexities = []
    # active_operators = []
    # all_pauli_basis = []
    # all_weights_per_time = []
    #
    # for rule in rules:
    #     krylov_arr = np.stack(krylov_complexities_ensemble[rule])
    #     active_arr = np.stack(active_operators_ensemble[rule])
    #     avg_krylov = np.mean(krylov_arr, axis=0)
    #     avg_active = np.mean(active_arr, axis=0)
    #     krylov_complexities.append(avg_krylov)
    #     active_operators.append(avg_active)
    #
    #     # For pauli_basis and weights_per_time, just use the first trial (structure doesn't change)
    #     all_pauli_basis.append(all_pauli_basis_ensemble[rule][0])
    #     # Average weights per time
    #     weights_stacked = np.stack([np.array(w) for w in all_weights_per_time_ensemble[rule]])
    #     avg_weights_per_time = np.mean(weights_stacked, axis=0)
    #     all_weights_per_time.append(avg_weights_per_time)
    #
    # # --- PUBLICATION STYLE PLOTS ---
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    #           '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
    #           '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5']
    # line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--',
    #                '-.', ':', '-']
    #
    # plot_publication_2panel(
    #     rules, krylov_complexities, active_operators,
    #     pastel_colors=colors,
    #     line_styles=line_styles
    # )
    # plot_all_weight_distributions(
    #     rules, all_pauli_basis, all_weights_per_time, pastel_colors=colors
    # )

    ####################################  USE FOR ! TRIAL  with Lanczos WRONG#####################################
    gs_complexities_ensemble = {rule: [] for rule in rules}
    active_paulis_ensemble = {rule: [] for rule in rules}
    all_pauli_basis_ensemble = {rule: [] for rule in rules}
    all_weights_per_time_ensemble = {rule: [] for rule in rules}

    for rule in rules:
        n_trials = N_TRIALS if rule == "random" or not (rule == "brickwork") else 1

        for trial_index in range(n_trials):
            if rule == "random":
                pattern_string = ''.join(random.choice('gj') for _ in range(time_steps))
                unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
            elif rule == "brickwork":
                unitaries = create_brickwork_unitaries(n_qubits, theta)
            else:
                pattern_string=rule
                unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
                # with open(f'../non_markovian_orders_list/{rule}.csv', 'r') as f:
                #     reader = csv.reader(f)
                #     NM_list = [row[0].strip() for row in reader if row and row[0].strip()]
                #     pattern_string = NM_list[trial_index][0:100]

                    #unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)

                    # filename = f"../Thesis_plots/C2/config_list_CS3_{rule}_steps_10_500.csv"
                # with open(filename, 'r') as f:
                #     reader = csv.reader(f)
                #     next(reader)
                #     for i, row in enumerate(reader):
                #         if i == trial_index:
                #             pattern_string = ''.join(
                #                 cell.strip().replace('0', 'j').replace('1', 'g') for cell in row[1:]
                #             )
                            #unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
                            #break

            # -- Run the quantum circuit analysis --
            results = analyze_operator_spreading(
                n_qubits, initial_operator, unitaries, time_steps, symmetry="U1"
            )
            gs_results = orthogonalize_evolved_operators(
                results['evolved_operators'],
                QuantumOperatorAnalyzer(n_qubits, symmetry="U1"),
                use_lanczos=False
            )
            gs_coeffs = np.array(gs_results['coefficients_in_orthogonal_basis'], dtype=np.complex128)
            gs_complexity = compute_gs_complexity(gs_coeffs)
            gs_complexities_ensemble[rule].append(gs_complexity)

            # Number of active Pauli strings at each time step
            weights_per_time = results['weights_per_time']
            active_pauli = [np.sum(np.array(w) > WEIGHT_THRESHOLD) for w in weights_per_time]
            active_paulis_ensemble[rule].append(active_pauli)

            all_pauli_basis_ensemble[rule].append(results['pauli_basis'])
            all_weights_per_time_ensemble[rule].append(results['weights_per_time'])

    # -- Compute ensemble averages --
    gs_complexities = []
    active_paulis = []
    all_pauli_basis = []
    all_weights_per_time = []

    for rule in rules:
        gs_arr = np.stack(gs_complexities_ensemble[rule])
        pauli_arr = np.stack(active_paulis_ensemble[rule])
        avg_gs = np.mean(gs_arr, axis=0)
        avg_pauli = np.mean(pauli_arr, axis=0)
        gs_complexities.append(avg_gs)
        active_paulis.append(avg_pauli)

        all_pauli_basis.append(all_pauli_basis_ensemble[rule][0])
        weights_stacked = np.stack([np.array(w) for w in all_weights_per_time_ensemble[rule]])
        avg_weights_per_time = np.mean(weights_stacked, axis=0)
        all_weights_per_time.append(avg_weights_per_time)

    # --- PUBLICATION STYLE PLOTS ---
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
    plot_gs_and_pauli_2panel(
        rules, gs_complexities, active_paulis, pastel_colors=colors, line_styles=line_styles)

####################################################
    # krylov_complexities = []
    # active_operators = []
    # lanczos_coeffs_all = []
    #
    # for rule in rules:
    #     # No inner loop: only one trial per rule
    #     krylov_trials = []
    #     active_trials = []
    #     lanczos_trials = []
    #
    #     # Generate circuit/unitaries for this rule
    #     if rule == "random":
    #         pattern_string = ''.join(random.choice('gj') for _ in range(time_steps))
    #         unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
    #     elif rule == "brickwork":
    #         unitaries = create_brickwork_unitaries(n_qubits, theta)
    #     else:
    #         with open(f'../non_markovian_orders_list/{rule}.csv', 'r') as f:
    #             reader = csv.reader(f)
    #             NM_list = [row[0].strip() for row in reader if row and row[0].strip()]
    #             pattern_string = NM_list[0]  # implement this as needed
    #             print(pattern_string)
    #             unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
    #
    #     results = analyze_operator_spreading(
    #         n_qubits, initial_operator, unitaries, time_steps, symmetry="U1_sbasis"
    #     )
    #     gs_results = orthogonalize_evolved_operators(
    #         results['evolved_operators'],
    #         QuantumOperatorAnalyzer(n_qubits, symmetry="U1_sbasis")
    #     )
    #
    #     weights_per_time = results['weights_per_time']
    #     active = [np.sum(np.array(w) > 1e-30) for w in weights_per_time]
    #     analyzer = QuantumOperatorAnalyzer(n_qubits, symmetry="U1_sbasis")
    #     orthogonalizer = GramSchmidtOrthogonalizer(analyzer)
    #     basis, alpha_coeffs, beta_coeffs = orthogonalizer.layer_by_layer_lanczos(
    #         results['evolved_operators'], verbose=False
    #     )
    #     krylov = orthogonalizer.compute_krylov_complexity(alpha_coeffs, beta_coeffs, time_steps)
    #
    #     krylov_trials.append(krylov)
    #     active_trials.append(active)
    #     lanczos_trials.append(beta_coeffs)
    #
    #     # Ensemble average (axis=0: average over trials)
    #     # Here, each *_trials list has only one entry, but averaging for consistency
    #     krylov_complexities.append(np.mean(krylov_trials, axis=0))
    #     active_operators.append(np.mean(active_trials, axis=0))
    #     lanczos_coeffs_all.append(np.mean(lanczos_trials, axis=0))
    #
    #     # Plotting (unchanged)
    #     plot_gs_basis_evolution(results, gs_results)
    #     plot_comprehensive_pauli_analysis(results)
    #
    #     krylov = orthogonalizer.compute_krylov_complexity(alpha_coeffs, beta_coeffs, time_steps)
    #     plt.figure()
    #     plt.plot(np.arange(len(krylov)), krylov, label="kry", color='teal')
    #     plt.xlabel('Time step')
    #     plt.ylabel('Krylov complexity')
    #     plt.title('Krylov Complexity Comparison')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()
    #
    # # Plotting side by side
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # labels = ["brickwork","random","rule_0_2_1_0","rule_0_3_1_1","rule_1_0_1_3","rule_0_3_1_1", "rule_1_2_3_3","rule_1_3_1_3","rule_1_3_3_2","rule_2_0_0_3","rule_2_1_2_3","rule_2_2_0_0", "rule_2_3_0_1", "rule_2_3_3_0", "rule_3_0_3_2", "rule_3_1_0_2", "rule_3_1_1_2","rule_3_1_2_2","rule_3_1_3_2","rule_3_3_1_1"]
    # # "rule_1_3_1_3", "rule_1_3_3_2", "rule_2_0_0_3", "rule_2_1_2_3", "rule_2_2_0_0", "rule_2_3_0_1",
    # # "rule_2_3_3_0", "rule_3_0_3_2", "rule_3_1_0_2", "rule_3_1_1_2", "rule_3_1_2_2", "rule_3_1_3_2",
    # # "rule_3_3_1_1"]  # ,"Mir1", "Mir2"]
    # # ,"stringjg","altstring","thetastring","salt1","salt2","salt3"]
    # # ["brickwork","random","rule_0_2_1_0","rule_0_3_1_1","rule_1_0_1_3","rule_0_3_1_1", "rule_1_2_3_3","rule_1_3_1_3","rule_1_3_3_2","rule_2_0_0_3","rule_2_1_2_3","rule_2_2_0_0", "rule_2_3_0_1", "rule_2_3_3_0", "rule_3_0_3_2", "rule_3_1_0_2", "rule_3_1_1_2","rule_3_1_2_2","rule_3_1_3_2","rule_3_3_1_1"]
    #
    # # ["brickwork", "random","rule_0_0_2_2","rule_0_1_0_0","rule_0_1_2_1","rule_0_2_1_0","rule_0_2_2_1"]
    # # "string","S2", "S3", "S4", "S5","S6","Fib","Frob", "Pal", "Frus","Mir1", "Mir2"]
    # # ,"rule_0_3_1_1", "rule_1_0_1_3", "rule_1_0_3_3", "rule_1_2_0_0", "rule_1_2_3_3",
    # # "rule_1_3_1_3", "rule_1_3_3_2", "rule_2_0_0_0", "rule_2_0_0_3", "rule_2_1_2_3",
    # # "rule_2_2_0_0", "rule_2_3_0_1", "rule_2_3_3_0", "rule_3_0_3_2", "rule_3_1_0_2", "rule_3_1_1_2",
    # # "rule_3_1_2_2", "rule_3_1_3_2", "rule_3_3_1_1"]
    #
    # # Create 26 distinct colors and line styles
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    #           '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
    #           '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5', '#393b79', '#637939', '#8c6d31', '#843c39',
    #           '#7b4173', '#5254a3']
    # line_styles = ['-', '--', '-.', ':', '-', '--'] * 5
    # time_avg = lambda data, w=50: [np.mean(data[max(0, i - w + 1):i + 1]) for i in range(len(data))]
    # krylov_avg = [time_avg(krylov_complexities[i]) for i in range(len(labels))]
    # active_avg = [time_avg(active_operators[i]) for i in range(len(labels))]
    # lanczos_avg = [time_avg(lanczos_coeffs_all[i]) for i in range(len(labels))]
    #
    # # Plot
    # fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    # for i in range(len(labels)):
    #     style = {'color': colors[i], 'linestyle': line_styles[i], 'linewidth': 1.5, 'alpha': 0.8}
    #     axes[0].plot(krylov_avg[i], label=labels[i], **style)
    #     axes[1].plot(active_avg[i], label=labels[i], **style)
    #     axes[2].plot(lanczos_avg[i], label=labels[i], **style)
    #
    # axes[0].set_title('Time-Averaged Krylov Complexity')
    # axes[1].set_title('Time-Averaged Active Operators')
    # axes[2].set_title('Time-Averaged Lanczos Coefficients')
    # for ax in axes: ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    # plt.tight_layout()
    # plt.show()
####################################################
    # for rule in rules:
    #     with open(f'../non_markovian_orders_list/{rule}.csv', 'r') as f:
    #          reader = csv.reader(f)
    #          NM_list_0_0_2_2 = [row[0].strip() for row in reader if row and row[0].strip()]
    #
    # #     len(NM_list_0_0_2_2[1])
    # pattern_string = NM_list_0_0_2_2[10][0:100]
    # theta = np.pi / 15
    # initial_operator = "ZIIIII"
    # #
    # #     print(f"Pattern: {pattern_string}")
    # #     print(f"Length: {len(pattern_string)} steps")
    # #
    # #     # Create circuit from pattern
    # unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
    # print("=== Debug Info ===")
    # print(f"n_qubits: {n_qubits}")
    # print(f"initial_operator: '{initial_operator}' (length: {len(initial_operator)})")
    #
    # # Test the analyzer creation
    # analyzer = QuantumOperatorAnalyzer(n_qubits, symmetry='U1_sbasis')
    # print(f"Analyzer basis_type: {analyzer.basis_type}")
    # print(f"Analyzer basis_size: {len(analyzer.pauli_basis)}")
    #
    # # Test format detection
    # is_ixyz = analyzer._is_ixyz_string(initial_operator)
    # contains_spin = analyzer._contains_spin_operators(initial_operator)
    # print(f"is_ixyz_string: {is_ixyz}")
    # print(f"contains_spin_operators: {contains_spin}")
    #
    # # Try to convert to matrix
    # try:
    #     matrix = analyzer.operator_string_to_matrix(initial_operator)
    #     print(f"✓ Matrix conversion successful, shape: {matrix.shape}")
    # except Exception as e:
    #     print(f"✗ Matrix conversion failed: {e}")
    #     import traceback
    #
    #     traceback.print_exc()
    #
    # # Now try your original call
    # results = analyze_operator_spreading(
    #     n_qubits, initial_operator, unitaries, len(pattern_string),
    #     symmetry='U1_sbasis', verbose=True
    # )
    # #     print(f"Created {len(unitaries)} unitaries")
    # #
    # #     # Run analysis
    # #     # results = analyze_operator_spreading(
    # #     #     n_qubits, "ZIIIIIII", unitaries, len(pattern_string),
    # #     #     symmetry=None, verbose=True
    # #     # )
    # plot_dynamic_active_operators(results, threshold=1e-6)
    # plot_pauli_structure_evolution(results, threshold=1e-6)
    # plot_operator_lifecycles(results, threshold=1e-6)
    # plot_weight_flow_network(results, threshold=1e-6)
    # plot_exploration_metrics(results, threshold=1e-6)
    # active_ops, sorted_active = plot_dynamic_active_operators(results, threshold=1e-6)
    # structure_weights, structure_strings = plot_pauli_structure_evolution(results, threshold=1e-6)
    # lifecycles = plot_operator_lifecycles(results, threshold=1e-6)
    # locality_weights, locality_strings = plot_weight_flow_network(results, threshold=1e-6)
    # plot_exploration_metrics(results, threshold=1e-6)
    #
    # # Access specific string information
    # print("X-only operators:", structure_strings['X-only'])
    # if 2 in locality_strings:
    #     print("Locality 2 operators:", locality_strings[2])
    # if lifecycles:
    #     print("First operator to appear:", lifecycles[0]['operator'])
    # # Gram-Schmidt
    # analyzer = QuantumOperatorAnalyzer(n_qubits, symmetry="U1_sbasis")
    # gs_results = orthogonalize_evolved_operators(results['evolved_operators'], analyzer)
    #
    # # Plot results
    # plot_operator_evolution(results)
    # plot_gs_basis_evolution(results, gs_results)
    # plot_comprehensive_pauli_analysis(results)
    #
    #
    #     # U(1) in IXYZ basis (32K operators for 8-qubit)
    #     results = analyze_operator_spreading(
    #         n_qubits, "ZIIIIIII", unitaries, len(pattern_string),
    #         symmetry='U1', verbose=True
    #     )
    #
    #     plot_dynamic_active_operators(results, threshold=1e-6)
    #     plot_pauli_structure_evolution(results, threshold=1e-6)
    #     plot_operator_lifecycles(results, threshold=1e-6)
    #     plot_weight_flow_network(results, threshold=1e-6)
    #     plot_exploration_metrics(results, threshold=1e-6)
    #     active_ops, sorted_active = plot_dynamic_active_operators(results, threshold=1e-6)
    #     structure_weights, structure_strings = plot_pauli_structure_evolution(results, threshold=1e-6)
    #     lifecycles = plot_operator_lifecycles(results, threshold=1e-6)
    #     locality_weights, locality_strings = plot_weight_flow_network(results, threshold=1e-6)
    #     plot_exploration_metrics(results, threshold=1e-6)
    #
    #     # Access specific string information
    #     print("X-only operators:", structure_strings['X-only'])
    #     if 2 in locality_strings:
    #         print("Locality 2 operators:", locality_strings[2])
    #     if lifecycles:
    #         print("First operator to appear:", lifecycles[0]['operator'])
    #     # Gram-Schmidt
    #     analyzer = QuantumOperatorAnalyzer(n_qubits, symmetry='U1')
    #     gs_results = orthogonalize_evolved_operators(results['evolved_operators'], analyzer)
    #
    #     # Plot results
    #     plot_operator_evolution(results)
    #     plot_gs_basis_evolution(results, gs_results)
    #     plot_comprehensive_pauli_analysis(results)
    #
    #     # U(1) in S+/S- basis (13K operators for 8-qubit - fastest!)
    #     results = analyze_operator_spreading(
    #         n_qubits, "ZIIIIIII", unitaries, len(pattern_string),
    #         symmetry='U1_sbasis', verbose=True
    #     )
    #
    #     plot_dynamic_active_operators(results, threshold=1e-6)
    #     plot_pauli_structure_evolution(results, threshold=1e-6)
    #     plot_operator_lifecycles(results, threshold=1e-6)
    #     plot_weight_flow_network(results, threshold=1e-6)
    #     plot_exploration_metrics(results, threshold=1e-6)
    #     active_ops, sorted_active = plot_dynamic_active_operators(results, threshold=1e-6)
    #     structure_weights, structure_strings = plot_pauli_structure_evolution(results, threshold=1e-6)
    #     lifecycles = plot_operator_lifecycles(results, threshold=1e-6)
    #     locality_weights, locality_strings = plot_weight_flow_network(results, threshold=1e-6)
    #     plot_exploration_metrics(results, threshold=1e-6)
    #
    #     # Access specific string information
    #     print("X-only operators:", structure_strings['X-only'])
    #     if 2 in locality_strings:
    #         print("Locality 2 operators:", locality_strings[2])
    #     if lifecycles:
    #         print("First operator to appear:", lifecycles[0]['operator'])
    #     # Gram-Schmidt
    #     analyzer = QuantumOperatorAnalyzer(n_qubits, symmetry='U1_sbasis')
    #     gs_results = orthogonalize_evolved_operators(results['evolved_operators'], analyzer)
    #
    #     # Plot results
    #     plot_operator_evolution(results)
    #     plot_gs_basis_evolution(results, gs_results)
    #     plot_comprehensive_pauli_analysis(results)
    # Add this to your code to test:
    # Add this right before your analyze_operator_spreading call:




