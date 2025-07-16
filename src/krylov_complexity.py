
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


class QuantumOperatorAnalyzer:
    """Efficient analyzer for operator spreading in quantum circuits."""

    def __init__(self, n_qubits: int, symmetry: Optional[str] = None):
        self.n_qubits = n_qubits
        self.symmetry = symmetry
        self.pauli_matrices = self._get_pauli_matrices()
        self.pauli_basis = self._generate_pauli_basis()

        # Only pre-compute for small systems to avoid memory issues
        if len(self.pauli_basis) < 5000:
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

    def _generate_pauli_basis(self) -> List[str]:
        """Generate Pauli strings respecting symmetries."""
        if self.symmetry is None:
            return self._generate_full_basis()
        elif self.symmetry == 'Z2':
            return self._generate_z2_basis()
        elif self.symmetry == 'U1':
            return self._generate_u1_basis()
        else:
            raise ValueError(f"Unknown symmetry: {self.symmetry}")

    def _generate_full_basis(self) -> List[str]:
        """Generate all possible Pauli strings for N qubits."""
        pauli_chars = ['I', 'X', 'Y', 'Z']
        return [''.join(p) for p in product(pauli_chars, repeat=self.n_qubits)]

    def _generate_z2_basis(self) -> List[str]:
        """Generate Z2-symmetric Pauli strings (even X+Y count)."""
        pauli_chars = ['I', 'X', 'Y', 'Z']
        z2_strings = []

        for pauli_tuple in product(pauli_chars, repeat=self.n_qubits):
            xy_count = pauli_tuple.count('X') + pauli_tuple.count('Y')
            if xy_count % 2 == 0:
                z2_strings.append(''.join(pauli_tuple))

        return z2_strings

    def _generate_u1_basis(self) -> List[str]:
        """Generate U1-symmetric Pauli strings (equal X and Y count)."""
        u1_strings = []
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

                            u1_strings.append(''.join(pauli_string))

        return u1_strings

    def _precompute_optimization_data(self):
        """Pre-compute data structures for optimization."""
        print(f"Pre-computing optimization data for {len(self.pauli_basis):,} Pauli strings...")
        start_time = time.time()

        dim = 2 ** self.n_qubits
        self.pauli_tensor = np.zeros((len(self.pauli_basis), dim, dim), dtype=complex)

        for i, pauli_string in enumerate(self.pauli_basis):
            self.pauli_tensor[i] = self.pauli_string_to_matrix(pauli_string)

        print(f"Pre-computation completed in {time.time() - start_time:.2f}s")

    def pauli_string_to_matrix(self, pauli_string: str) -> np.ndarray:
        """Convert a Pauli string to its matrix representation."""
        if len(pauli_string) != self.n_qubits:
            raise ValueError(f"Pauli string length must be {self.n_qubits}")

        result = self.pauli_matrices[pauli_string[0]]
        for pauli_char in pauli_string[1:]:
            result = np.kron(result, self.pauli_matrices[pauli_char])
        return result

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
                pauli_matrix = self.pauli_string_to_matrix(self.pauli_basis[i])
                overlaps[i] = np.trace(operator @ pauli_matrix) / d

        if verbose:
            total_time = time.time() - start_time
            print(f"Batched computation completed in {total_time:.3f}s")

        return overlaps

    def apply_unitary_to_operator(self, operator: np.ndarray, unitary: np.ndarray) -> np.ndarray:
        """Apply unitary evolution: U† O U."""
        return unitary.conj().T @ operator @ unitary

    def compute_weight_distribution(self, overlaps: np.ndarray) -> np.ndarray:
        """Compute weight distribution from overlaps."""
        return np.abs(overlaps) ** 2

    def evolve_operator(self, initial_pauli_string: str, unitaries: List[np.ndarray],
                        time_steps: int, verbose: bool = False,
                        method: str = 'vectorized') -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Time evolve an initial Pauli operator and track its spreading."""

        if verbose:
            print(f"Starting evolution using {method} method")
            print(f"System: {self.n_qubits} qubits, {len(self.pauli_basis):,} basis states")

        # Choose overlap computation method
        if method == 'vectorized':
            overlap_func = self.compute_overlap_vectorized
        elif method == 'batched':
            overlap_func = self.compute_overlap_batched
        else:
            raise ValueError(f"Unknown method: {method}")

        # Initialize with the initial Pauli operator
        current_operator = self.pauli_string_to_matrix(initial_pauli_string)

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


class GramSchmidtOrthogonalizer:
    """Gram-Schmidt orthogonalization for evolved operators."""

    def __init__(self, analyzer: QuantumOperatorAnalyzer):
        self.analyzer = analyzer

    def robust_modified_gram_schmidt(self, operators: List[np.ndarray]) -> List[np.ndarray]:
        """Robust modified Gram-Schmidt orthogonalization."""
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

                # Verify orthogonality
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


# Circuit construction utilities
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


# Analysis functions
def analyze_operator_spreading(n_qubits: int, initial_pauli_string: str,
                               unitaries: List[np.ndarray], time_steps: int,
                               symmetry: Optional[str] = None, verbose: bool = False,
                               method: str = 'vectorized') -> Dict:
    """Main function to analyze operator spreading."""
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


def orthogonalize_evolved_operators(evolved_operators: List[np.ndarray],
                                    analyzer: QuantumOperatorAnalyzer) -> Dict:
    """Perform Gram-Schmidt orthogonalization on evolved operators."""
    orthogonalizer = GramSchmidtOrthogonalizer(analyzer)

    # Perform orthogonalization
    orthogonal_ops = orthogonalizer.robust_modified_gram_schmidt(evolved_operators)

    # Express original operators in orthogonal basis
    coefficients = orthogonalizer.express_in_orthogonal_basis(
        evolved_operators, orthogonal_ops
    )

    return {
        'orthogonal_operators': orthogonal_ops,
        'coefficients_in_orthogonal_basis': coefficients
    }


# Plotting functions
def plot_operator_evolution(results: Dict, top_n: int = 15, figsize: Tuple = (12, 8)):
    """Plot evolution of top operators."""
    weights_per_time = results['weights_per_time']
    pauli_basis = results['pauli_basis']
    n_time_steps = len(weights_per_time)

    # Find top operators by maximum weight
    all_max_weights = []
    for i, pauli_string in enumerate(pauli_basis):
        max_weight = max(weights_per_time[t][i] for t in range(n_time_steps))
        all_max_weights.append((max_weight, i, pauli_string))

    all_max_weights.sort(reverse=True)

    plt.figure(figsize=figsize)

    for rank, (_, idx, pauli_string) in enumerate(all_max_weights[:top_n]):
        weights_over_time = [weights_per_time[t][idx] for t in range(n_time_steps)]
        color = PASTEL_COLORS[rank % len(PASTEL_COLORS)]

        plt.plot(range(n_time_steps), weights_over_time, 'o-',
                 linewidth=2, color=color, label=pauli_string,
                 markersize=4, alpha=0.8)

    plt.xlabel('Time Step')
    plt.ylabel('Weight')
    plt.title(f'Top {top_n} Operators: Weight Evolution')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_gs_basis_evolution(results: Dict, gs_results: Dict, figsize: Tuple = (12, 8)):
    """Plot Gram-Schmidt basis evolution."""
    coefficients_per_time = gs_results['coefficients_in_orthogonal_basis']
    n_time_steps = len(coefficients_per_time)
    n_gs_operators = len(coefficients_per_time[0]) if coefficients_per_time else 0

    weights_gs_basis = np.abs(np.array(coefficients_per_time)) ** 2
    time_steps = range(n_time_steps)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Gram-Schmidt Basis Evolution', fontsize=16)

    # Top left: Weighted GS index
    n_indices = np.arange(n_gs_operators)
    weighted_sum = np.sum(n_indices * weights_gs_basis, axis=1)
    axes[0, 0].plot(time_steps, weighted_sum, 'o-', linewidth=2, color='darkorange')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel(r'$\sum_n n|a_n(t)|^2$')
    axes[0, 0].set_title('Weighted GS Index')
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Norm conservation
    total_weights = np.sum(weights_gs_basis, axis=1)
    deviation_from_one = total_weights - 1.0
    axes[0, 1].plot(time_steps, deviation_from_one * 1e12, 'o-', linewidth=2, color='green')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Deviation × 10¹²')
    axes[0, 1].set_title('Norm Conservation')
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom left: Heatmap
    im = axes[1, 0].imshow(weights_gs_basis.T, aspect='auto', cmap='hot', origin='lower')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('GS Operator Index')
    axes[1, 0].set_title('GS Evolution Heatmap')
    plt.colorbar(im, ax=axes[1, 0], label='|a_n(t)|²')

    # Bottom right: Weight conservation
    axes[1, 1].plot(time_steps, total_weights, 'b-o', linewidth=2, label='Total')
    for k in [1, 2, 3, 5]:
        if k <= n_gs_operators:
            cum_weight = np.sum(weights_gs_basis[:, :k], axis=1)
            axes[1, 1].plot(time_steps, cum_weight, '--', linewidth=2,
                            label=f'First {k} operators')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Cumulative Weight')
    axes[1, 1].set_title('Weight Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Additional utility functions and improvements

def validate_initial_operator(initial_pauli_string: str, symmetry: Optional[str] = None) -> bool:
    """Validate that initial operator respects the specified symmetry."""
    if symmetry is None:
        return True

    x_count = initial_pauli_string.count('X')
    y_count = initial_pauli_string.count('Y')

    if symmetry == 'Z2':
        xy_count = x_count + y_count
        valid = xy_count % 2 == 0
        if not valid:
            print(
                f"Warning: Initial operator '{initial_pauli_string}' has odd X+Y count ({xy_count}), violates Z2 symmetry")
        return valid

    elif symmetry == 'U1':
        valid = x_count == y_count
        if not valid:
            print(
                f"Warning: Initial operator '{initial_pauli_string}' has X={x_count}, Y={y_count}, violates U1 symmetry")
        return valid

    return True


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
    """Plot comprehensive complexity metrics."""
    metrics = compute_complexity_metrics(results)
    n_time_steps = len(metrics['nonzero_counts'])
    time_steps = range(n_time_steps)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Operator Spreading Complexity Metrics', fontsize=16, fontweight='bold')

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


"""
Functions to plot Pauli operator statistics and weight histograms
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_pauli_operator_counts(results, figsize=(12, 5)):
    """
    Plot total and active Pauli operator counts over time.

    Args:
        results: Output from analyze_operator_spreading()
        figsize: Figure size tuple
    """
    weights_per_time = results['weights_per_time']
    total_operators = len(weights_per_time[0])
    time_steps = range(len(weights_per_time))

    # Count active operators at each time step
    active_counts = []
    for weights in weights_per_time:
        active = np.sum(np.array(weights) > 1e-30)  # Non-zero threshold
        active_counts.append(active)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Active vs Total
    ax1.plot(time_steps, active_counts, 'o-', linewidth=2,
             color='#FF6B6B', markersize=6, label='Active operators')
    ax1.axhline(y=total_operators, color='#4ECDC4', linestyle='--',
                linewidth=2, label=f'Total operators ({total_operators})')

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Number of Operators')
    ax1.set_title('Pauli Operator Space Exploration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Percentage active
    percentage_active = [100 * count / total_operators for count in active_counts]
    ax2.plot(time_steps, percentage_active, 's-', linewidth=2,
             color='#45B7D1', markersize=6)

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Percentage Active (%)')
    ax2.set_title('Fraction of Pauli Space Explored')
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
    print(f"Pauli Operator Summary:")
    print(f"  Total operators in basis: {total_operators}")
    print(f"  Initial active: {active_counts[0]}")
    print(f"  Final active: {final_active}")
    print(f"  Exploration: {final_percentage:.1f}% of Pauli space")

    return active_counts


def plot_nonzero_weight_histogram(results, time_step=-1, bins=50, figsize=(10, 6)):
    """
    Plot histogram of non-zero Pauli operator weights.

    Args:
        results: Output from analyze_operator_spreading()
        time_step: Which time step to plot (-1 for final)
        bins: Number of histogram bins
        figsize: Figure size tuple
    """
    weights = np.array(results['weights_per_time'][time_step])
    pauli_basis = results['pauli_basis']

    # Get non-zero weights
    nonzero_mask = weights > 1e-30
    nonzero_weights = weights[nonzero_mask]
    nonzero_labels = [pauli_basis[i] for i in range(len(pauli_basis)) if nonzero_mask[i]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

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
        f'Distribution of Non-Zero Pauli Weights (t={time_step if time_step >= 0 else len(results["weights_per_time"]) - 1})')
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
    print(f"\nTop 10 Pauli operators by weight:")
    sorted_indices = np.argsort(nonzero_weights)[::-1]
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        weight = nonzero_weights[idx]
        pauli_op = nonzero_labels[idx]
        print(f"  {i + 1:2d}. {pauli_op}: {weight:.3e}")

    return nonzero_weights, nonzero_labels


def plot_weight_evolution_heatmap(results, max_operators=50, figsize=(12, 8)):
    """
    Plot heatmap showing weight evolution of top Pauli operators.

    Args:
        results: Output from analyze_operator_spreading()
        max_operators: Maximum number of operators to show
        figsize: Figure size tuple
    """
    weights_per_time = results['weights_per_time']
    pauli_basis = results['pauli_basis']
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
    ax.set_ylabel('Pauli Operator', fontsize=12)
    ax.set_title(f'Weight Evolution of Top {len(top_labels)} Pauli Operators',
                 fontsize=14, pad=20)

    # Set ticks
    ax.set_xticks(range(0, n_time_steps, max(1, n_time_steps // 10)))
    ax.set_yticks(range(len(top_labels)))
    ax.set_yticklabels(top_labels, fontsize=8)

    plt.tight_layout()
    plt.show()

    return top_weights, top_labels


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


if __name__ == "__main__":
    # Parameters
    n_qubits = 8
    rules= ["rule_0_0_2_2","rule_1_3_3_2","rule_3_3_1_1"]
    for rule in rules:
        with open(f'../non_markovian_orders_list/{rule}.csv', 'r') as f:
            reader = csv.reader(f)
            NM_list_0_0_2_2 = [row[0].strip() for row in reader if row and row[0].strip()]

        len(NM_list_0_0_2_2[1])
        pattern_string = NM_list_0_0_2_2[1][0:5]
        theta = np.pi / 15
        initial_operator = "ZIIIIIII"

        print(f"Pattern: {pattern_string}")
        print(f"Length: {len(pattern_string)} steps")

        # Create circuit from pattern
        unitaries = generate_nonmarkovian_circuit_unitaries(n_qubits, pattern_string, theta)
        print(f"Created {len(unitaries)} unitaries")

        # Run analysis
        results = analyze_operator_spreading(
            n_qubits, initial_operator, unitaries, len(pattern_string),
            symmetry='U1', verbose=True
        )

        # Gram-Schmidt
        analyzer = QuantumOperatorAnalyzer(n_qubits, symmetry='U1')
        gs_results = orthogonalize_evolved_operators(results['evolved_operators'], analyzer)

        # Plot results
        plot_operator_evolution(results)
        plot_gs_basis_evolution(results, gs_results)
        plot_comprehensive_pauli_analysis(results)

