import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import random
from collections import defaultdict, deque
from scipy.linalg import cholesky
from math import floor, sqrt, log, exp
from typing import List, Dict, Tuple, Optional
import heapq
from collections import defaultdict, deque

#This function draws a quantum circuit diagram based on the number of qubits, timesteps, and a sequence of gates.

def draw_quantum_circuit(num_qubit, timesteps, gate_sequence):
    fig, ax = plt.subplots(figsize=(2 + timesteps, 1 + num_qubit * 0.7))
    ax.axis('off')

    x_gap = 1
    dot_radius = 0.1
    pastel_colors = {
        1: '#b3c6ff',  # blue
        2: '#C1E1C1',  # green
        3: '#fff7b3',  # yellow
        4: '#ffb3d9',  # pink
    }

    # Draw qubit lines and dots
    for q in range(num_qubit):
        y = num_qubit - q
        ax.add_patch(plt.Circle((0, y), dot_radius, color='black', zorder=5))
        ax.plot([0, timesteps * x_gap], [y, y], color='black', lw=1, zorder=1)
        ax.text(-0.4, y, f'q{q}', ha='right', va='center', fontsize=12)

    # Draw timestep labels
    for t in range(timesteps):
        x = (t + 1) * x_gap
        ax.text(x, num_qubit + 0.7, f't{t+1}', ha='center', va='bottom', fontsize=12, color='gray')

    for t in range(timesteps):
        gates = gate_sequence[t]
        used_qubits = set()

        # Handle multi-qubit gates (tuples)
        for gate in gates:
            if isinstance(gate, tuple):
                label = gate[0]
                qubits = list(gate[1:])
                n = len(qubits)
                if n == 0 or n > 4:
                    continue
                color = pastel_colors.get(n, '#e0e0e0')
                x = (t + 1) * x_gap
                qubits_sorted = sorted(qubits)
                y_positions = [num_qubit - q for q in qubits_sorted]
                if n == 1:
                    q = qubits[0]
                    y = num_qubit - q
                    ax.add_patch(plt.Rectangle((x - 0.2, y - 0.2), 0.4, 0.4,
                                               facecolor=color, ec='black', zorder=3))
                    ax.text(x, y, str(label), ha='center', va='center', fontsize=12, zorder=4)
                    used_qubits.add(q)
                elif all(qubits_sorted[i] + 1 == qubits_sorted[i+1] for i in range(n-1)):
                    y_min = min(y_positions)
                    y_max = max(y_positions)
                    height = y_max - y_min + 0.4
                    ax.add_patch(plt.Rectangle((x - 0.25, y_min - 0.2), 0.5, height,
                                               facecolor=color, ec='black', zorder=3))
                    ax.text(x, (y_min + y_max) / 2, str(label), ha='center', va='center', fontsize=12, zorder=4)
                    used_qubits.update(qubits)
                else:
                    for y in y_positions:
                        ax.add_patch(plt.Circle((x, y), 0.18, facecolor=color, ec='black', zorder=3))
                        ax.text(x, y, str(label), ha='center', va='center', fontsize=10, zorder=4)
                    ax.plot([x]*len(y_positions), y_positions, 'k:', lw=1, zorder=2)
                    used_qubits.update(qubits)

        # Handle single-qubit gates (strings)
        if isinstance(gates, list):
            for q in range(num_qubit):
                if q in used_qubits:
                    continue
                gate = gates[q]
                if isinstance(gate, str) and gate:
                    y = num_qubit - q
                    x = (t + 1) * x_gap
                    if gate == "Measure":
                        # Draw white box
                        ax.add_patch(plt.Rectangle((x - 0.2, y - 0.2), 0.4, 0.4,
                                                   facecolor='white', ec='black', zorder=3))
                        # Draw diagonal arrow (↗)
                        ax.annotate(
                            '', xy=(x + 0.12, y + 0.12), xytext=(x - 0.12, y - 0.12),
                            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1.5),
                            zorder=4
                        )
                    else:
                        color = pastel_colors[1]
                        ax.add_patch(plt.Rectangle((x - 0.2, y - 0.2), 0.4, 0.4,
                                                   facecolor=color, ec='black', zorder=3))
                        ax.text(x, y, str(gate), ha='center', va='center', fontsize=12, zorder=4)

    ax.set_xlim(-0.5, timesteps * x_gap + 0.5)
    ax.set_ylim(0.5, num_qubit + 0.5)
    plt.tight_layout()
    plt.show()

#Visualize some famous quantum algorithms using the draw_quantum_circuit function.
def Shors_algorithm():
    """
    Draws a quantum circuit for Shor's algorithm.

    Args:
        num_qubit (int): Number of qubits.
        timesteps (int): Number of timesteps in the circuit.
        gate_sequence (list): List of gates at each timestep.
    """
    # Shor's Algorithm: H on all, modular exponentiation, QFT, Measure
    num_qubit = 4
    timesteps = 5
    gate_sequence = [
        ['H', 'H', 'H', 'H'],  # t=0: Hadamard on all
        [('',), ('ModExp', 0, 1, 2, 3)],  # t=1: Modular exponentiation (4-qubit)
        [('',), ('QFT', 0, 1, 2, 3)],  # t=2: QFT (4-qubit)
        ['', '', '', ''],  # t=3: (idle)
        ['Measure', 'Measure', 'Measure', 'Measure']  # t=4: Measure all
    ]
    return draw_quantum_circuit(num_qubit, timesteps, gate_sequence)


def Grovers_algorithm():
    """
    Draws a quantum circuit for Grover's algorithm.

    Args:
        num_qubit (int): Number of qubits.
        timesteps (int): Number of timesteps in the circuit.
        gate_sequence (list): List of gates at each timestep.
    """
    num_qubit = 3
    timesteps = 5
    gate_sequence = [
        ['H', 'H', 'H'],  # t=0: Hadamard on all
        [('',), ('Oracle', 0, 1, 2)],  # t=1: Oracle (3-qubit gate)
        [('',), ('Diff', 0, 1, 2)],    # t=2: Diffusion (3-qubit gate)
        ['', '', ''],                  # t=3: (idle)
        ['Measure', 'Measure', 'Measure']  # t=4: Measure all
    ]
    return draw_quantum_circuit(num_qubit, timesteps, gate_sequence)


def Teleportation():
    """
    Draws a quantum circuit for quantum teleportation.

    Args:
        num_qubit (int): Number of qubits.
        timesteps (int): Number of timesteps in the circuit.
        gate_sequence (list): List of gates at each timestep.
    """
# Teleportation: H, CNOT, Bell, X, Z, Measure
    num_qubit = 3
    timesteps = 6
    gate_sequence = [
        ['H', '', ''],                # t=0: H on qubit 0
        [('',), ('CNOT', 0, 1), '', ''],  # t=1: CNOT 0->1
        [('',), ('CNOT', 1, 2), '', ''],  # t=2: CNOT 1->2
        [('',), ('H', 1), '', ''],        # t=3: H on qubit 1
        ['Measure', 'Measure', ''],       # t=4: Measure 0,1
        ['', '', 'X']                     # t=5: X on qubit 2 (classically controlled, shown as X)
    ]
    circuit =draw_quantum_circuit(num_qubit, timesteps, gate_sequence)
    return circuit

#Draw a random two-qubit gate sequence for a given number of qubits and timesteps.

def random_two_qubit_gate_sequence(num_qubit, timesteps, gate_name='CNOT'):
    """
    Generates a random two-qubit gate sequence for a given number of qubits and timesteps.
    Each gate couples a qubit to its right or left neighbor.
    Returns a list of lists (length num_qubit) for each timestep.
    """
    gate_sequence = []
    for t in range(timesteps):
        used = set()
        gates = [''] * num_qubit
        qubit_indices = list(range(num_qubit))
        random.shuffle(qubit_indices)
        for q in qubit_indices:
            if q in used:
                continue
            direction = random.choice(['left', 'right'])
            if direction == 'right' and q < num_qubit - 1 and (q + 1) not in used:
                gates[q] = (gate_name, q, q + 1)
                used.add(q)
                used.add(q + 1)
            elif direction == 'left' and q > 0 and (q - 1) not in used:
                gates[q] = (gate_name, q, q - 1)
                used.add(q)
                used.add(q - 1)
        gate_sequence.append(gates)
    return gate_sequence

#Now add measurement gates to the random two-qubit gate sequence.
def random_two_qubit_gate_sequence_with_measure(num_qubit, timesteps, gate_name='CNOT', p=0.1):
    """
    Generates a random two-qubit gate sequence for a given number of qubits and timesteps.
    With probability p, applies a Measure gate to each qubit between two-qubit gate layers.
    Returns a list of lists (length num_qubit) for each timestep.
    """
    gate_sequence = []
    for t in range(timesteps):
        # Two-qubit gate layer
        used = set()
        gates = [''] * num_qubit
        qubit_indices = list(range(num_qubit))
        random.shuffle(qubit_indices)
        for q in qubit_indices:
            if q in used:
                continue
            direction = random.choice(['left', 'right'])
            if direction == 'right' and q < num_qubit - 1 and (q + 1) not in used:
                gates[q] = (gate_name, q, q + 1)
                used.add(q)
                used.add(q + 1)
            elif direction == 'left' and q > 0 and (q - 1) not in used:
                gates[q] = (gate_name, q, q - 1)
                used.add(q)
                used.add(q - 1)
        gate_sequence.append(gates)
        # Measurement layer (not after last two-qubit layer)
        if t < timesteps - 1:
            measure_layer = [
                "Measure" if random.random() < p else '' for _ in range(num_qubit)
            ]
            gate_sequence.append(measure_layer)
    return gate_sequence

# This function finds the minimum cut trajectory between two qubits in a circuit sequence.

def find_min_cut_trajectory_between_with_measure_robust(
    a_circuit_sequence, start_index_1, start_index_2, num_trajectories=1000
):
    timesteps = len(a_circuit_sequence)
    num_qubits = len(a_circuit_sequence[0])
    min_cuts = float('inf')
    min_cut_path = []

    for traj in range(num_trajectories):
        idx1 = min(start_index_1, start_index_2)
        idx2 = max(start_index_1, start_index_2)
        cuts = 0
        path = []
        stopped = False

        for t in reversed(range(timesteps)):
            if stopped:
                path.append('EXIT')
                continue

            gates = a_circuit_sequence[t]
            measure1 = (idx1 < num_qubits and isinstance(gates[idx1], str) and gates[idx1] == "Measure")
            measure2 = (idx2 < num_qubits and isinstance(gates[idx2], str) and gates[idx2] == "Measure")
            coupling = any(isinstance(g, tuple) and set(g[1:]) == {idx1, idx2} for g in gates)

            # Edge exit via measure
            if (idx1 == 0 and idx2 == 1 and measure1) or (idx1 == num_qubits - 2 and idx2 == num_qubits - 1 and measure2):
                path.append('EXIT')
                stopped = True
                continue

            if coupling:
                # At left edge: must move L (right)
                if idx1 == 0 and idx2 == 1:
                    idx1 += 1
                    idx2 += 1
                    path.append('L')
                    cuts += 1
                    # After cut, traverse consecutive measures to the right
                    while idx2 < num_qubits and isinstance(gates[idx2], str) and gates[idx2] == "Measure":
                        if random.random() < 0.5:
                            idx1 += 1
                            idx2 += 1
                            path.append('L-MEASURE')
                        else:
                            break
                # At right edge: must move R (left)
                elif idx1 == num_qubits - 2 and idx2 == num_qubits - 1:
                    idx1 -= 1
                    idx2 -= 1
                    path.append('R')
                    cuts += 1
                    # After cut, traverse consecutive measures to the left
                    while idx1 >= 0 and isinstance(gates[idx1], str) and gates[idx1] == "Measure":
                        if random.random() < 0.5:
                            idx1 -= 1
                            idx2 -= 1
                            path.append('R-MEASURE')
                        else:
                            break
                else:
                    rand_val = random.random()
                    if rand_val < 0.5:
                        if idx1 > 0:
                            idx1 -= 1
                            idx2 -= 1
                            path.append('R')
                            cuts += 1
                            # After cut, traverse consecutive measures to the left
                            while idx1 >= 0 and isinstance(gates[idx1], str) and gates[idx1] == "Measure":
                                if random.random() < 0.5:
                                    idx1 -= 1
                                    idx2 -= 1
                                    path.append('R-MEASURE')
                                else:
                                    break
                        else:
                            path.append('')
                    else:
                        if idx2 < num_qubits - 1:
                            idx1 += 1
                            idx2 += 1
                            path.append('L')
                            cuts += 1
                            # After cut, traverse consecutive measures to the right
                            while idx2 < num_qubits and isinstance(gates[idx2], str) and gates[idx2] == "Measure":
                                if random.random() < 0.5:
                                    idx1 += 1
                                    idx2 += 1
                                    path.append('L-MEASURE')
                                else:
                                    break
                        else:
                            path.append('')
            elif measure1 or measure2:
                # Traverse consecutive measures to the left
                if measure1 and idx1 > 0:
                    while idx1 > 0 and isinstance(gates[idx1], str) and gates[idx1] == "Measure":
                        if random.random() < 0.5:
                            idx1 -= 1
                            idx2 -= 1
                            path.append('R-MEASURE')
                        else:
                            break
                # Traverse consecutive measures to the right
                elif measure2 and idx2 < num_qubits - 1:
                    while idx2 < num_qubits - 1 and isinstance(gates[idx2], str) and gates[idx2] == "Measure":
                        if random.random() < 0.5:
                            idx1 += 1
                            idx2 += 1
                            path.append('L-MEASURE')
                        else:
                            break
                else:
                    path.append('')
            else:
                path.append('')

        if cuts < min_cuts:
            min_cuts = cuts
            min_cut_path = path

    return min_cuts, min_cut_path

#seq=random_two_qubit_gate_sequence_with_measure(5,5, gate_name='CNOT', p=0.2)
#min_cuts,min_cut_path=find_min_cut_trajectory_between_with_measure_robust(seq, 0, 1, num_trajectories=100)
#print(seq)
#draw_quantum_circuit(5,len(seq), seq)


# Your random version
#random_cuts, random_path = find_min_cut_trajectory_between_with_measure_robust(
#    seq, 2, 3, num_trajectories=100000
#)
#print("Minimum cuts:", min_cuts)
#print("Cut path:", min_cut_path)
def plot_minimal_cut_system_size_vs_measurement_probability(p_values,qubit_sizes,num_trials=100):
    """
    num_trials = 100
    p_values = np.arange(0, 1.01, 0.2)
    qubit_sizes = np.arange(5, 101, 5)
    """
    results = []
    for p in p_values:
        for nq in qubit_sizes:
            min_cuts_list = []
            for _ in range(num_trials):
                seq = random_two_qubit_gate_sequence_with_measure(nq, nq, gate_name='CNOT', p=p)
                min_cuts, _ = find_min_cut_trajectory_between_with_measure_robust(seq, nq//2-1, nq//2, num_trajectories=10000)
                min_cuts_list.append(min_cuts)
            avg_cuts = np.mean(min_cuts_list)
            results.append((nq, avg_cuts, p))
            print(f"nq={nq}, p={p:.2f}, avg min cuts={avg_cuts:.2f}")

    # Prepare data for plotting
    sizes = np.array([r[0] for r in results])
    avg_cuts = np.array([r[1] for r in results])
    ps = np.array([r[2] for r in results])

    # Red gradient: low p = light, high p = dark
    colors = plt.cm.Reds(ps)

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(sizes, avg_cuts, c=ps, cmap='Reds', s=40)
    cbar = plt.colorbar(sc)
    cbar.set_label('Measurement Probability p')
    plt.xlabel('System size (number of qubits)')
    plt.ylabel('Average Minimum Cuts')
    plt.title('Average Minimum Cuts vs System Size\nColored by Measurement Probability p')
    plt.tight_layout()
    plt.show()

def plot_minimal_cut_subsystem_size_separate():
    num_trials = 5
    p_values = np.arange(0, 1.01, 0.25)
    system_sizes = np.arange(10, 101, 25)
    cmap = plt.get_cmap('tab10')  # 10 distinct colors

    for idx, n in enumerate(system_sizes):
        plt.figure(figsize=(10, 6))
        base_color = cmap(idx % 10)
        for p_idx, p in enumerate(p_values):
            print(f"System size n={n}, p={p:.2f}")
            cuts_vs_pos = []
            for cut_pos in range(n - 1):
                min_cuts_list = []
                for trial in range(num_trials):
                    seq = random_two_qubit_gate_sequence_with_measure(n, n, gate_name='CNOT', p=p)
                    min_cuts, _ = find_min_cut_trajectory_between_with_measure_robust(
                        seq, cut_pos, cut_pos + 1, num_trajectories=100
                    )
                    min_cuts_list.append(min_cuts)
                avg_cuts = np.mean(min_cuts_list)
                cuts_vs_pos.append(avg_cuts)
            # Gradient for p within the base color
            color = list(base_color)
            color[-1] = 0.3 + 0.7 * (p_idx / (len(p_values) - 1))  # alpha for p gradient
            x_vals = np.arange(1, n) / n
            plt.scatter(x_vals, cuts_vs_pos, color=color, label=f'p={p:.2f}', s=30)
        plt.xlabel('End index / System size')
        plt.ylabel('Average Minimal Cuts')
        plt.title(f'Minimal Cuts vs Position (System size n={n})')
        plt.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()

