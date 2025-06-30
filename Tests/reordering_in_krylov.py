import numpy as np

#import numpy as np
import matplotlib.pyplot as plt

def create_partial_swap_gate(theta):
    return np.array([
        [1, 0,           0,          0],
        [0, np.cos(theta), 1j*np.sin(theta), 0],
        [0, 1j*np.sin(theta), np.cos(theta), 0],
        [0, 0,           0,          1]
    ], dtype=complex)

def permute_basis(order, n):
    dim = 2 ** n
    perm = np.zeros((dim, dim), dtype=complex)
    for k in range(dim):
        bits = [(k >> l) & 1 for l in range(n)]
        permuted = [bits[order[m]] for m in range(n)]
        idx = sum([b << l for l, b in enumerate(permuted)])
        perm[idx, k] = 1
    return perm

def embed_two_qubit_gate(gate, i, j, n):
    qubit_order = [i, j] + [q for q in range(n) if q != i and q != j]
    inv_order = np.argsort(qubit_order)
    P = permute_basis(qubit_order, n)
    P_inv = permute_basis(inv_order, n)
    op = np.kron(gate, np.eye(2 ** (n - 2), dtype=complex))
    return P_inv @ op @ P

def construct_circuit_with_embedding(n_qubits, circuit_structure, theta):
    pswap = create_partial_swap_gate(theta)
    unitaries = []
    for pairs in circuit_structure:
        U = np.eye(2 ** n_qubits, dtype=complex)
        for pair in pairs:
            gate_full = embed_two_qubit_gate(pswap, pair[0], pair[1], n_qubits)
            U = gate_full @ U
        unitaries.append(U)
    return unitaries

def plot_unitaries_per_time_step(unitaries):
    for t, U in enumerate(unitaries):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(np.real(U), aspect='auto', cmap='bwr')
        axes[0].set_title(f'Time {t+1}: Real part')
        plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(np.imag(U), aspect='auto', cmap='bwr')
        axes[1].set_title(f'Time {t+1}: Imag part')
        plt.colorbar(im1, ax=axes[1])
        plt.suptitle(f'Unitary at Time Step {t+1}')
        plt.tight_layout()
        plt.show()

def permute_state(state, order, n):
    permuted = np.zeros_like(state)
    for k in range(len(state)):
        bits = [(k >> l) & 1 for l in range(n)]
        new_bits = [bits[order[m]] for m in range(n)]
        idx = sum([b << l for l, b in enumerate(new_bits)])
        permuted[idx] = state[k]
    return permuted

# Example usage and check
if __name__ == "__main__":
    n = 6
    theta = np.pi / 4
    pswap = create_partial_swap_gate(theta)

    # Initial state |010101>
    state = np.zeros(2**n, dtype=complex)
    state[int('010101', 2)] = 1.0

    # Apply embedded PSWAP to (0, 4)
    U = embed_two_qubit_gate(pswap, 0, 4, n)
    result1 = U @ state

    # Manual permutation: bring 0,4 to front, apply PSWAP, permute back
    order = [0, 4] + [i for i in range(n) if i not in [0, 4]]
    inv_order = np.argsort(order)
    state_perm = permute_state(state, order, n)
    op = np.kron(pswap, np.eye(2**(n-2), dtype=complex))
    result2 = permute_state(op @ state_perm, inv_order, n)

    # Check if results match
    assert np.allclose(result1, result2), "PSWAP embedding failed for non-nearest neighbors"
    print("Check passed: PSWAP acts correctly on non-nearest neighbor qubits.")