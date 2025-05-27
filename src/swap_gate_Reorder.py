import numpy as np
import matplotlib.pyplot as plt


def swap_gate(n, a, b):
    """
    Create a SWAP gate for an N-qubit system that swaps qubits at positions a and b.

    Parameters:
    n (int): Total number of qubits in the system.
    a (int): Index of the first qubit to be swapped (0-based index).
    b (int): Index of the second qubit to be swapped (0-based index).

    Returns:
    np.ndarray: A 2^N x 2^N matrix representing the SWAP gate.
    """
    # Total size of the system
    size = 2 ** n
    # Start with an identity matrix
    swap = np.eye(size)

    # Iterate through all basis states
    for i in range(size):
        # Convert the state index to binary (as a string)
        binary = format(i, f'0{n}b')
        # Swap the bits at positions a and b
        swapped_binary = list(binary)
        swapped_binary[a], swapped_binary[b] = swapped_binary[b], swapped_binary[a]
        # Convert the swapped binary string back to an integer
        swapped_index = int("".join(swapped_binary), 2)
        # Modify the matrix to reflect the swap
        swap[i, i] = 0
        swap[i, swapped_index] = 1

    return swap


n = 4  # Number of qubits
a = 0  # First qubit to swap
b = 2  # Second qubit to swap

swap_matrix = swap_gate(n, a, b)
print("SWAP Gate:")
print(swap_matrix)
plt.figure(figsize=(8, 8))  # Adjust the figure size
plt.imshow(swap_matrix, cmap="gray", interpolation="nearest")  # Use a grayscale heatmap
plt.colorbar(label="Matrix Value")  # Add a legend for clarity
plt.title(f"SWAP Gate Matrix (Qubits: {n}, Swap: {a} ↔ {b})")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.grid(False)  # Optional: Turn grid off for cleaner visualization
plt.show()



