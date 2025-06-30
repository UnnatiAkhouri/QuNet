import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy import linalg

def extract_observables(two_qubit_dms, n_qubits):
    """
    Extract single-qubit σz expectation values and two-qubit correlations from density matrices.

    Args:
        two_qubit_dms: Array of shape (n_trials, n_timesteps, n_pairs) containing dictionaries
                    with two-qubit density matrices
        n_qubits: Number of qubits in the system

    Returns:
        sz: Array of shape (n_trials, n_timesteps, n_qubits) with σz expectation values
        corr: Array of shape (n_trials, n_timesteps, n_qubits, n_qubits) with σz-σz correlations
    """
    n_trials = two_qubit_dms.shape[0]
    n_timesteps = two_qubit_dms.shape[1]

    # Pauli Z matrix
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli_z_tensor = np.kron(pauli_z, pauli_z)

    # Initialize arrays as complex to avoid casting warnings
    sz = np.zeros((n_trials, n_timesteps, n_qubits), dtype=complex)
    corr = np.zeros((n_trials, n_timesteps, n_qubits, n_qubits), dtype=complex)

    # For each trial and timestep
    for trial in range(n_trials):
        for t in range(n_timesteps):
            # Count how many times each qubit appears to average properly
            counts = np.zeros(n_qubits)

            # For each qubit pair
            for pair, dm in two_qubit_dms[trial, t].items():
                i, j = pair

                # Get density matrix as numpy array
                rho = dm.data

                # Calculate local observables
                sz_i = np.trace(rho @ np.kron(pauli_z, np.eye(2, dtype=complex)))
                sz_j = np.trace(rho @ np.kron(np.eye(2, dtype=complex), pauli_z))

                # Calculate correlation
                corr_ij = np.trace(rho @ pauli_z_tensor) - sz_i * sz_j

                # Update counts
                counts[i] += 1
                counts[j] += 1

                # Update expectation values (will average later)
                sz[trial, t, i] += sz_i
                sz[trial, t, j] += sz_j

                # Store correlation
                corr[trial, t, i, j] = corr_ij
                corr[trial, t, j, i] = corr_ij

            # Average the expectation values
            for i in range(n_qubits):
                if counts[i] > 0:
                    sz[trial, t, i] /= counts[i]

    # Take real parts at the end - for Hermitian observables like Pauli-Z
    # the expectation values should be real, with any imaginary component
    # coming from numerical imprecision
    return sz, corr


def plot_mean_correlation_sums(corrs_all_datas):
    dataset_labels = ["PC_UNC", "PC_C", "POL","CNOT"]
    markers = ["o", "d", "^","s"]
    rules = ["p0", "p25", "p50", "p75", "p1"]
    colors = [
    '#40E0D0',  # Turquoise
    '#00FA9A',  # Medium spring green
    '#32CD32',  # Lime green
    '#FFD700',  # Gold
    '#FFA500'   # Orange
]
    
    fig, axes = plt.subplots(len(corrs_all_datas), 1, figsize=(8, 2*len(corrs_all_datas)))
    
    # Make axes iterable if there's only one subplot
    if len(corrs_all_datas) == 1:
        axes = [axes]
    
    for i, corr_per_ic in enumerate(corrs_all_datas):
        for j, corr_per_rule_in_ic in enumerate(corr_per_ic):
            # Get the correlation matrices for all trials
            trial_data = corr_per_rule_in_ic[1]
            
            # Number of time steps
            n_timesteps = len(trial_data[0])
            
            # Calculate mean sum for each time step across all trials
            mean_correlation_sums = []
            for t in range(n_timesteps):
                # Get the sum of absolute values for this time step across all trials
                step_sums = [np.sum(np.abs(trial[t])) for trial in trial_data]
                # Calculate the mean across all trials
                mean_correlation_sums.append(np.mean(step_sums))

            n_timesteps=101
            # Create time steps array
            x = np.arange(1, n_timesteps + 1)
            
            # Plot the mean sums
            axes[i].plot(x, mean_correlation_sums[0:200], 
                        marker=markers[i], color=colors[j], 
                        label=rules[j], markersize=2, markevery=10)
        
        # Add a vertical line at x=10 to each subplot
        axes[i].axvline(x=10, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        # Add dataset marker and label to upper right corner
        dataset_handle = plt.Line2D([0], [0], marker=markers[i], color='black', 
                              markerfacecolor='black', markersize=5, linestyle='')
        dataset_legend = axes[i].legend([dataset_handle], [dataset_labels[i]], 
                            loc='upper right', fontsize=8)
        axes[i].add_artist(dataset_legend)
        
        # Add rule legend at the bottom of each subplot
        rule_handles = [plt.Line2D([0], [0], color=colors[k], linewidth=2) 
                     for k in range(len(rules))]
        rule_legend = axes[3].legend(handles=rule_handles, labels=rules, 
                             loc='lower right', fontsize=10)
        axes[3].add_artist(rule_legend)
        
        # Add x-axis label only to the bottom subplot
        if i == len(corrs_all_datas) - 1:
            axes[i].set_xlabel(r'$\ell$', fontsize=18)
        
        # Remove y-axis label from individual subplots
        axes[i].set_ylabel('')
        
        axes[i].grid(True)
    
    # Add a single y-axis label for the entire figure
    fig.text(0.02, 0.5, r'Total correlation $(\langle \sum_{a,b} \frac{|C^{xx}_{ab}| (\ell)}{4}\rangle_{\rm ens}$)', va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.15)

    #plt.savefig("Correlation_growth_12Q_all_rules_all_CS.png",dpi=300,bbox_inches="tight")
    # Adjust layout to make room for the y-axis label
    
    return fig


def von_neumann_entropy(rho):
    """
    Calculate the von Neumann entropy S = -Tr(rho log(rho)) for a density matrix.
    
    Args:
        rho: A numpy array representing a density matrix
        
    Returns:
        The von Neumann entropy as a real number
    """
    # Get the eigenvalues of the density matrix
    eigenvalues = linalg.eigvalsh(rho)
    
    # Remove very small eigenvalues that might cause numerical issues
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # Calculate entropy: -sum(λ_i * log(λ_i))
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    
    return entropy

def calculate_entropy_for_size(dms, trial_idx, step_idx, size):
    """
    Calculate entropies for all subsystems of a specific size in a given trial and step.
    
    Args:
        dms: Dictionary of density matrices (key: qubit_size -> trial_idx -> step_idx -> subsystem_tuple)
        trial_idx: Trial index
        step_idx: Step index
        size: Qubit subsystem size
        
    Returns:
        List of entropy values for all subsystems of the specified size
    """
    if size not in dms:
        return []
    
    try:
        subsystems = dms[size][trial_idx][step_idx]
        entropies = []
        
        for qubits, dm in subsystems.items():
            # Convert DM to numpy array
            dm_array = dm.data.toarray()
            
            # Calculate entropy
            try:
                entropy = von_neumann_entropy(dm_array)
                entropies.append(entropy)
            except Exception as e:
                print(f"Error calculating entropy for size {size}, qubits {qubits} at step {step_idx}: {e}")
        
        return entropies
    except (KeyError, IndexError) as e:
        print(f"Error accessing data for size {size}, trial {trial_idx}, step {step_idx}: {e}")
        return []


def analyze_entropy_across_sizes(dms_dict, num_trials, num_steps=5, max_size=8):
    """
    Calculate entropy statistics across different qubit subsystem sizes.
    
    Args:
        dms_dict: Dictionary of density matrices by size 
                (e.g., {1: onedms, 2: twodms, 3: threedms, ...})
        num_trials: Number of trials
        num_steps: Number of steps to analyze
        max_size: Maximum qubit subsystem size
        
    Returns:
        Dictionary with structure: {step_idx: {size: (mean_entropy, std_entropy)}}
    """
    # Initialize results dictionary
    results = {step: {} for step in range(num_steps)}
    
    # Process each step
    for step_idx in range(num_steps):
        print(f"Processing step {step_idx+1}/{num_steps}")
        
        # Process each qubit subsystem size
        for size in range(1, max_size + 1):
            if size not in dms_dict:
                print(f"No data available for {size}-qubit subsystems")
                continue
                
            # Collect entropies across all trials for this step and size
            all_entropies = []
            
            for trial_idx in range(num_trials):
                entropies = calculate_entropy_for_size(dms_dict, trial_idx, step_idx, size)
                all_entropies.extend(entropies)
            
            # Calculate statistics if we have data
            if all_entropies:
                mean_entropy = np.mean(all_entropies)
                std_entropy = np.std(all_entropies)
                results[step_idx][size] = (mean_entropy, std_entropy)
            else:
                print(f"No entropy data for {size}-qubit subsystems at step {step_idx}")
    
    return results


def plot_entropy_by_size_and_step(entropy_results, num_steps=5, max_size=8,col="random"):
    """
    Plot entropy vs. subsystem size with different lines for each step.
    
    Args:
        entropy_results: Results from analyze_entropy_across_sizes
        num_steps: Number of steps that were analyzed
        max_size: Maximum qubit subsystem size
    """
    plt.figure(figsize=(12, 8))
    
    # Create a color map with different shades of blue
    if col=="random":
        blues = cm.Blues(np.linspace(0.3, 1.0, num_steps))
    if col=="mimic":
        blues = cm.Purples(np.linspace(0.3, 1.0, num_steps))
    if col =="random":
        blues = cm.Reds(np.linspace(0.3, 1.0, num_steps))
    # Plot each step with a different shade of blue
    for step_idx in range(num_steps):
        if step_idx not in entropy_results:
            continue
            
        sizes = sorted(entropy_results[step_idx].keys())
        means = [entropy_results[step_idx][size][0] for size in sizes]
        stds = [entropy_results[step_idx][size][1] for size in sizes]
        
        plt.errorbar(sizes, means, yerr=stds, 
                    marker='o', linestyle='-', capsize=5,
                    color=blues[step_idx], 
                    label=f'Step {step_idx}')
    
    plt.xlabel('Qubit Subsystem Size')
    plt.ylabel('von Neumann Entropy (bits)')
    plt.title('Entropy Scaling with Subsystem Size at Different Time Steps')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, max_size + 1))
    plt.savefig('entropy_scaling.png')
    plt.show()
    
    # Print the results in a table
    print("Entropy Results by Size and Step:")
    print("Size | " + " | ".join([f"Step {i} Mean±Std" for i in range(num_steps)]))
    print("-" * (7 + 19 * num_steps))
    
    for size in range(1, max_size + 1):
        row = [f"{size:4d}"]
        
        for step in range(num_steps):
            if step in entropy_results and size in entropy_results[step]:
                mean, std = entropy_results[step][size]
                row.append(f"{mean:.4f}±{std:.4f}")
            else:
                row.append("N/A")
        
        print(" | ".join(row))


# Main function to run all analyses
def analyze_entropy_scaling(dms_dict, num_trials, num_steps=5, max_size=8,col="random"):
    """
    Analyze entropy scaling with subsystem size across different time steps.
    
    Args:
        dms_dict: Dictionary of density matrices by size
        num_trials: Number of trials
        num_steps: Number of steps
        max_size: Maximum qubit subsystem size
    """
    # Calculate entropy statistics for each size and step
    results = analyze_entropy_across_sizes(dms_dict, num_trials, num_steps, max_size)
    
    # Plot and print the results
    plot_entropy_by_size_and_step(results, num_steps, max_size,col)
    
    return results


def average_mutual_information_vs_subsystem(
    dms_by_size,
    full_dms,
    sizes_to_compare=[(2,6), (3,5), (4,4), (5,3), (6,2)],
    entropy_func=None
):
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import linalg

    def von_neumann_entropy(rho):
        if hasattr(rho, "data"):
            arr = rho.data.toarray() if hasattr(rho.data, "toarray") else np.array(rho.data)
        else:
            arr = np.array(rho)
        arr = np.asarray(arr, dtype=np.complex128)
        eigenvalues = linalg.eigvalsh(arr)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        return entropy

    n_trials = len(full_dms)
    n_steps = len(full_dms[0])
    n_total = 8
    sizes = sorted(set([a for ab in sizes_to_compare for a in ab]))
    results = {size: [np.nan]*n_steps for size in sizes}
    done_pairs = set()

    for (a, b) in sizes_to_compare:
        key = tuple(sorted((a, b)))
        if key in done_pairs:
            continue
        done_pairs.add(key)
        # Find the minimum number of steps available for both sizes
        steps_a = min([len(dms_by_size[a][trial]) for trial in range(n_trials) if dms_by_size[a][trial] is not None])
        steps_b = min([len(dms_by_size[b][trial]) for trial in range(n_trials) if dms_by_size[b][trial] is not None])
        steps = min(n_steps, steps_a, steps_b)
        for t in range(n_steps - steps, n_steps):
            mi_list = []
            for trial in range(n_trials):
                if (a not in dms_by_size or b not in dms_by_size or
                    dms_by_size[a][trial] is None or dms_by_size[b][trial] is None):
                    continue
                t_a = t - (n_steps - len(dms_by_size[a][trial]))
                t_b = t - (n_steps - len(dms_by_size[b][trial]))
                if t_a < 0 or t_b < 0:
                    continue
                A_dict = dms_by_size[a][trial][t_a]
                B_dict = dms_by_size[b][trial][t_b]
                full_dm = full_dms[trial][t][tuple(range(n_total))]
                for A_indices in itertools.combinations(range(n_total), a):
                    B_indices = tuple(sorted(set(range(n_total)) - set(A_indices)))
                    if A_dict is None or B_dict is None:
                        continue
                    A_dm = A_dict.get(A_indices)
                    B_dm = B_dict.get(B_indices)
                    if A_dm is None or B_dm is None:
                        continue
                    A_dm = A_dm.data.toarray()
                    B_dm = B_dm.data.toarray()
                    S_A = von_neumann_entropy(A_dm)
                    S_B = von_neumann_entropy(B_dm)
                    S_AB = von_neumann_entropy(full_dm)
                    mi = S_A + S_B - S_AB
                    mi_list.append(mi)
            if mi_list:
                results[a][t] = np.mean(mi_list)
                results[b][t] = np.mean(mi_list)  # Symmetric assignment

    plt.figure(figsize=(8, 5))
    for t in range(n_steps):
        y = [results[size][t] for size in sizes]
        plt.plot(sizes, y, marker='o', label=f'Time step {t}')
    plt.xlabel('Subsystem Size')
    plt.ylabel('Average Mutual Information')
    plt.title('Average MI vs Subsystem Size (per time step)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results