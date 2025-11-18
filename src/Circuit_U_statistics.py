import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from sklearn.cluster import KMeans

# --- Pauli utilities ---
def pauli(label):
    paulis = {'I': np.eye(2), 'X': np.array([[0,1],[1,0]]), 'Y': np.array([[0,-1j],[1j,0]]), 'Z': np.array([[1,0],[0,-1]])}
    return paulis[label]

def kron_paulis(pauli_string):
    result = pauli(pauli_string[0])
    for p in pauli_string[1:]:
        result = np.kron(result, pauli(p))
    return result

def pauli_basis(n):
    labels = ['I', 'X', 'Y', 'Z']
    return [''.join(p) for p in product(labels, repeat=n)]

def hamming_weight(pauli_str):
    return sum(1 for c in pauli_str if c != 'I')

def expand_in_pauli_basis(A, n_qubits):
    basis = pauli_basis(n_qubits)
    coeffs = []
    for pstr in basis:
        P = kron_paulis(pstr)
        coeff = np.trace(P.conj().T @ A) / (2 ** n_qubits)
        coeffs.append(coeff)
    return basis, np.array(coeffs)

def pauli_weight_distribution(U, A, n_qubits):
    A_evolved = U @ A @ U.conj().T
    basis, coeffs = expand_in_pauli_basis(A_evolved, n_qubits)
    weights = [hamming_weight(p) for p in basis]
    weight_dict = {}
    for w in range(n_qubits + 1):
        mask = np.array(weights) == w
        weight_dict[w] = np.sum(np.abs(coeffs[mask]) ** 2)
    return weight_dict

# --- Brickwork unitary ---
def create_partial_swap_gate(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, 1j*s, 0],
        [0, 1j*s, c, 0],
        [0, 0, 0, 1]
    ], dtype=complex)

def embed_two_qubit_gate(gate: np.ndarray, q1: int, q2: int, n_qubits: int) -> np.ndarray:
    ops = [np.eye(2, dtype=complex) if i not in (q1, q2) else None for i in range(n_qubits)]
    def kron_with_gate(ops, gate):
        kron_list, i = [], 0
        while i < len(ops):
            if ops[i] is None and i+1 < len(ops) and ops[i+1] is None:
                kron_list.append(gate)
                i += 2
            else:
                kron_list.append(ops[i])
                i += 1
        result = kron_list[0]
        for op in kron_list[1:]:
            result = np.kron(result, op)
        return result
    return kron_with_gate(ops, gate)

def create_brickwork_unitaries(n_qubits: int, theta_even: float, theta_odd: float):
    pswap_even = create_partial_swap_gate(theta_even)
    pswap_odd = create_partial_swap_gate(theta_odd)
    U_even = np.eye(2 ** n_qubits, dtype=complex)
    for i in range(0, n_qubits - 1, 2):
        gate_full = embed_two_qubit_gate(pswap_even, i, i + 1, n_qubits)
        U_even = gate_full @ U_even
    U_odd = np.eye(2 ** n_qubits, dtype=complex)
    for i in range(1, n_qubits - 1, 2):
        gate_full = embed_two_qubit_gate(pswap_odd, i, i + 1, n_qubits)
        U_odd = gate_full @ U_odd
    return [U_even, U_odd]

def compose_layers(n_qubits, theta_fix, layer_sequence, params):
    U = np.eye(2 ** n_qubits, dtype=complex)
    for layer, param in zip(layer_sequence, params):
        theta = param * theta_fix
        if layer == 'even':
            U_even, _ = create_brickwork_unitaries(n_qubits, theta, 0)
            U = U_even @ U
        elif layer == 'odd':
            _, U_odd = create_brickwork_unitaries(n_qubits, 0, theta)
            U = U_odd @ U
        else:
            raise ValueError("layer_sequence must contain only 'even' or 'odd'")
    return U

def operator_distance(U, identity=None):
    if identity is None:
        identity = np.eye(U.shape[0], dtype=complex)
    return np.linalg.norm(U - identity, ord='fro') / np.linalg.norm(identity, ord='fro')

def parametric_distance_grid(n_qubits, theta_fix, layer_sequence, ranges):
    identity = np.eye(2 ** n_qubits, dtype=complex)
    all_params = list(product(*ranges))
    distances = []
    for params in all_params:
        if all(p == 0 for p in params):
            distances.append(np.nan)
            continue
        U = compose_layers(n_qubits, theta_fix, layer_sequence, params)
        dist = operator_distance(U, identity)
        distances.append(dist)
    distances = np.array(distances)
    grids = np.meshgrid(*[np.array(list(r)) for r in ranges], indexing='ij')
    return grids, distances.reshape([len(r) for r in ranges])

# --- Advanced Plotting Functions ---

def plot_parameter_slices(distances, ranges, fixed_param_index=2):
    param_labels = ['a', 'b', 'c']
    for i in range(distances.shape[fixed_param_index]):
        plt.figure(figsize=(7,5))
        if fixed_param_index == 2:
            plt.imshow(distances[:,:,i], origin='lower', aspect='auto', cmap='viridis',
                       extent=[ranges[0][0], ranges[0][-1], ranges[1][0], ranges[1][-1]])
            plt.xlabel(param_labels[0])
            plt.ylabel(param_labels[1])
            plt.title(f'Distance from Identity, {param_labels[2]} = {ranges[2][i]}')
            plt.colorbar(label='Distance')
            plt.show()

def plot_stacked_bar(weight_matrix, hamming_weights=[1,2,3,4]):
    plt.figure(figsize=(12,5))
    bottom = np.zeros(weight_matrix.shape[0])
    for w in hamming_weights:
        plt.bar(range(weight_matrix.shape[0]), weight_matrix[:,w], bottom=bottom, label=f'Hamming {w}')
        bottom += weight_matrix[:,w]
    plt.xlabel('Parameter Index')
    plt.ylabel('Operator Weight')
    plt.title('Stacked Bar: Pauli Weight Contributions')
    plt.legend()
    plt.show()

def plot_violin(weight_matrix, hamming_weights=[1,2,3,4]):
    sns.violinplot(data=weight_matrix[:,hamming_weights])
    plt.xlabel('Hamming Weight')
    plt.ylabel('Distribution')
    plt.title('Violin Plot: Pauli Weight Distribution')
    plt.show()

def plot_clusters(weight_matrix, all_params, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(weight_matrix)
    cluster_labels = kmeans.labels_
    plt.figure(figsize=(8,5))
    plt.scatter(range(len(all_params)), cluster_labels, c=cluster_labels, cmap='tab10')
    plt.xlabel('Parameter Index')
    plt.ylabel('Cluster')
    plt.title('Parameter Region Clustering')
    plt.show()

def plot_contour(distances, ranges):
    plt.figure(figsize=(8,6))
    X, Y = np.meshgrid(ranges[0], ranges[1], indexing='ij')
    plt.contourf(X, Y, distances, levels=15, cmap='plasma')
    plt.xlabel('Theta_even')
    plt.ylabel('Theta_odd')
    plt.title('Contour Plot: Distance from Identity')
    plt.colorbar(label='Distance')
    plt.show()

def plot_pauli_weight_heatmap(weight_matrix, n_qubits, all_params, hamming_range=(1,4), title=""):
    plt.figure(figsize=(10, 6))
    plt.imshow(weight_matrix[:, hamming_range[0]:hamming_range[1]+1], aspect='auto', cmap='viridis',
               extent=[hamming_range[0], hamming_range[1], 0, len(all_params)])
    plt.colorbar(label='Total Operator Weight')
    plt.xlabel('Hamming Weight')
    plt.ylabel('Parameter Index')
    plt.title(title)
    plt.show()

def plot_pauli_weight_contributions(weight_matrix, all_params, n_qubits):
    avg_weights = weight_matrix @ np.arange(n_qubits+1) / np.sum(weight_matrix, axis=1)
    plt.figure(figsize=(8,5))
    plt.plot(range(len(all_params)), avg_weights, marker='o')
    plt.xlabel("Parameter Index")
    plt.ylabel("Average Hamming Weight")
    plt.title("Average Hamming Weight vs Parameter Index")
    plt.show()

def plot_max_weight_type(weight_matrix, all_params, n_qubits):
    max_weight = np.argmax(weight_matrix, axis=1)
    plt.figure(figsize=(8,5))
    plt.plot(range(len(all_params)), max_weight, marker='o')
    plt.xlabel("Parameter Index")
    plt.ylabel("Dominant Hamming Weight")
    plt.title("Dominant Hamming Weight vs Parameter Index")
    plt.show()

def find_high_onebody_regions(weight_matrix, all_params, n_qubits, threshold=0.4):
    one_body = weight_matrix[:,1]
    max_other = np.max(np.hstack([weight_matrix[:,2:4]]), axis=1)
    indices = np.where((one_body > max_other) & (one_body > threshold))[0]
    return indices

def sample_high_probability_params(all_params, weight_matrix, n_qubits, n_samples=10):
    """
    Samples n parameter sets (a, b, c) with high probability according to normalized 1-body Pauli weight.
    Probability for each parameter set is:
        prob_i = W1 / (W1 + W2 + W3 + W4)  if all params > 0, else 0
    Returns a list of chosen parameter sets.
    """
    scores = []
    for idx, params in enumerate(all_params):
        # Exclude trivial points (any zero parameter)
        if any(p == 0 for p in params):
            scores.append(0)
        else:
            W1 = weight_matrix[idx, 1]
            W234 = np.sum(weight_matrix[idx, 2:5])
            score = W1 / (W1 + W234 + 1e-8)  # Normalized 1-body fraction
            scores.append(score)
    scores = np.array(scores)

    # Normalize to probability distribution
    prob_dist = scores / (np.sum(scores) if np.sum(scores) > 0 else 1)

    # Sample n_samples indices according to prob_dist
    sample_inds = np.random.choice(len(all_params), size=n_samples, p=prob_dist)
    chosen_params = [all_params[i] for i in sample_inds]

    return chosen_params
# Generate all 4-letter Pauli strings with even number of X or Y
def even_xy_pauli_strings(n_qubits):
    result = []
    for pstr in product(paulis, repeat=n_qubits):
        xy_count = sum(1 for c in pstr if c in ('X', 'Y'))
        if xy_count % 2 == 0 and xy_count > 0:  # exclude IIII, ZZZZ, etc. if you want only nontrivial
            result.append(''.join(pstr))
    return result


# --- Main analysis loop ---
# python
if __name__ == "__main__":
    n_qubits = 4
    paulis = ['I', 'X', 'Y', 'Z']
    theta_fix = np.pi / 15
    layer_sequence = ['even', 'odd', 'even']
    ranges = [range(0, 10), range(0, 10), range(0, 30)]
    pauli_strings = even_xy_pauli_strings(n_qubits)
    A = kron_paulis("ZIII")
    pauli_str = "ZIII"

    print(f"Number of 4-qubit Pauli strings with even number of X/Y (excluding zero): {len(pauli_strings)}")
    print("Sample pauli_str for plot:", pauli_str[:100] + '...')
    all_params = list(product(*ranges))
    weight_matrix = np.zeros((len(all_params), n_qubits + 1))
    distances = []

    for idx, params in enumerate(all_params):
        U = compose_layers(n_qubits, theta_fix, layer_sequence, params)
        weight_dict = pauli_weight_distribution(U, A, n_qubits)
        for w in range(n_qubits + 1):
            weight_matrix[idx, w] = weight_dict.get(w, 0)
        distances.append(operator_distance(U))

    grids, dist_grid = parametric_distance_grid(n_qubits, theta_fix, layer_sequence, ranges)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.size': 12})

    fig, axes = plt.subplots(3, 2, figsize=(14, 18), constrained_layout=True)
    param_labels = ['a', 'b', 'c']
    c_indices = [1, 14, 20, 27]

    # Find global vmin/vmax for selected c slices
    selected_slices = [dist_grid[:, :, c] for c in c_indices]
    vmin = min(np.nanmin(s) for s in selected_slices)
    vmax = max(np.nanmax(s) for s in selected_slices)

    # Plot heatmaps for selected c values
    ims = []
    for idx, c in enumerate(c_indices):
        row = idx // 2
        col = idx % 2
        im = axes[row, col].imshow(
            dist_grid[:, :, c], origin='lower', aspect='auto', cmap='magma',
            extent=[ranges[0][0], ranges[0][-1], ranges[1][0], ranges[1][-1]],
            vmin=vmin, vmax=vmax
        )
        axes[row, col].set_xlabel('a')
        axes[row, col].set_ylabel('b')
        axes[row, col].set_title(f'Distance from Identity\nc = {ranges[2][c]}', fontsize=15)
        ims.append(im)

    # Add a single horizontal colorbar after the top two rows
    cbar = fig.colorbar(
        ims[0],
        ax=axes[:2, :].ravel().tolist(),
        orientation='horizontal',
        fraction=0.05,
        pad=0.08,
        aspect=40,
        use_gridspec=True
    )
    cbar.set_label('Distance from Identity',fontsize=15)

    # Third row, first: Normalized stacked bar chart for Hamming weights
    hamming_weights = [1, 2, 3, 4]
    norm_weights = weight_matrix[:, hamming_weights] / (
                np.sum(weight_matrix[:, hamming_weights], axis=1, keepdims=True) + 1e-12)
    bottom = np.zeros(norm_weights.shape[0])
    for i, w in enumerate(hamming_weights):
        axes[2, 0].bar(
            range(norm_weights.shape[0]),
            norm_weights[:, i],
            bottom=bottom,
            label=f'Hamming {w}',
            width=1.0
        )
        bottom += norm_weights[:, i]
    axes[2, 0].set_xlabel('Parameter Index', fontsize=12)
    axes[2, 0].set_ylabel('Fraction', fontsize=12)
    axes[2, 0].set_title('Normalized Hamming Weight', fontsize=15)
    axes[2, 0].legend()

    # Third row, second: Violin plot
    sns.violinplot(data=weight_matrix[:, 1:], ax=axes[2, 1])
    axes[2, 1].set_xlabel('Hamming Weight', fontsize=12)
    axes[2, 1].set_ylabel('Distribution', fontsize=12)
    axes[2, 1].set_title('Distribution of Pauli Weight', fontsize=15)
    for ax in axes.ravel():
        ax.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('summary_panel_selected_c.pdf', bbox_inches='tight')
    plt.show()
#     n_qubits = 4
#     theta_fix = np.pi / 15
#     layer_sequence = ['even', 'odd', 'even']
#     ranges = [range(1, 38), range(1, 38), range(1, 38)]
#     # ZIII
#     grids_z, norms_z = parametric_commutator_grid(n_qubits, theta_fix, layer_sequence, ranges, "XYXI")
#     plot_commutator_norm(grids_z, norms_z, layer_sequence, theta_fix, "XYXI")
#     # XXII
#     grids_xx, norms_xx = parametric_commutator_grid(n_qubits, theta_fix, layer_sequence, ranges, "ZZZI")
#     plot_commutator_norm(grids_xx, norms_xx, layer_sequence, theta_fix, "ZZZI")

# def main():
#     n_qubits = 8  # Try 6–8 for best speed/memory
#     N_cycles = 100
#     theta_fix=np.pi/15
#
#     S=S3(1000)
#     Fib=Fibonacci(1000)
#     Frob=Frustrated(1000)
#     Frus=Frustrated(1000)
#     Pal=Palindrome(1000)
#     Mir1=Mirror1(1000)
#     Mir2=Mirror2(1000)
#
#     theta_cycles = parse_cycle_string(S, theta_fix)
#     print("Cycles (theta_even, theta_odd):", theta_cycles)
#     U_pal = build_brickwork_unitary(n_qubits, theta_cycles)
#     angles_pal, spacings_pal = get_eigenvalues_and_spacings(U_pal)
#
#     # -- A. Random CUE Unitary --
#     U_CUE = unitary_group.rvs(2 ** n_qubits)
#     angles_CUE, spacings_CUE = get_eigenvalues_and_spacings(U_CUE)
#
#     # -- B. True Brickwork (fixed θ) --
#     theta_fixed = np.pi / 4
#     theta_cycles_brickwork = [(theta_fixed, theta_fixed)] * N_cycles
#     U_brickwork = build_brickwork_unitary(n_qubits, theta_cycles_brickwork)
#     angles_brick, spacings_brick = get_eigenvalues_and_spacings(U_brickwork)
#
#     # -- C. Example: Alternating/Random θ --
#     np.random.seed(42)
#     theta_cycles_mixed = [(np.random.uniform(0, np.pi), np.random.uniform(0, np.pi)) for _ in range(N_cycles)]
#     U_mixed = build_brickwork_unitary(n_qubits, theta_cycles_mixed)
#     angles_mixed, spacings_mixed = get_eigenvalues_and_spacings(U_mixed)
#
#     # --- Plot/Compare ---
#     results = [
#         ("Palindrome", angles_pal, spacings_pal),
#         ("Random CUE", angles_CUE, spacings_CUE),
#         ("True Brickwork", angles_brick, spacings_brick),
#         ("Varying θ Brickwork", angles_mixed, spacings_mixed)
#     ]
#     plot_all_eigen_stats(results, n_qubits)
#
#     for name, _, spacings in results:
#         wd_dist, pois_dist = compare_spacing_distribution(spacings)
#         print(f"{name}: Wigner-Dyson distance = {wd_dist:.3f}, Poisson distance = {pois_dist:.3f}")
#
# if __name__ == "__main__":
#     main()