import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import cos, sin, I, Rational, simplify, zeros, eye, Matrix

"""
This is just a helper code to see the transition matrix of a PSAWP operator. By transition matrix we mean: consider all the 2 letter Pauli words (16 in all)
then the transition matrix is a 16 x 16 matrix that describes how a Pauli word evolves into a linear combination of Pauli words under the action of PSAWP.

This is helpful in breaking up larger computation as PSWAP only acts on neighbours or pairs at a time. 
"""

def compute_pswap_pauli_action_sympy(theta):
    """Symbolic version - returns 16x16 transition matrix in terms of cos/sin theta."""
    c, s = cos(theta), sin(theta)
    U = Matrix([[1,0,0,0], [0,c,I*s,0], [0,I*s,c,0], [0,0,0,1]])
    paulis = [eye(2), Matrix([[0,1],[1,0]]), Matrix([[0,-I],[I,0]]), Matrix([[1,0],[0,-1]])]
    M = zeros(16, 16)
    for p0 in range(4):
        for p1 in range(4):
            P_evolved = U.H @ sp.kronecker_product(paulis[p0], paulis[p1]) @ U
            for q0 in range(4):
                for q1 in range(4):
                    Q = sp.kronecker_product(paulis[q0], paulis[q1])
                    M[p0*4+p1, q0*4+q1] = simplify((Q.H @ P_evolved).trace() / 4)
    return M

def compute_pswap_pauli_action(theta: float):
    """
    Compute how PSWAP(θ) transforms each two-qubit Pauli string.

    Returns a dictionary mapping (p0, p1) -> [(new_p0, new_p1, coeff), ...]
    where p0, p1 are Pauli types (0=I, 1=X, 2=Y, 3=Z)
    """
    c = np.cos(theta)
    s = np.sin(theta)

    # Build the gate matrix
    U = np.array([
        [1, 0, 0, 0],
        [0, c, 1j*s, 0],
        [0, 1j*s, c, 0],
        [0, 0, 0, 1]
    ], dtype=complex)

    # Pauli matrices
    paulis = [
        np.array([[1, 0], [0, 1]], dtype=complex),  # I
        np.array([[0, 1], [1, 0]], dtype=complex),  # X
        np.array([[0, -1j], [1j, 0]], dtype=complex),  # Y
        np.array([[1, 0], [0, -1]], dtype=complex)  # Z
    ]

    action = {}

    for p0 in range(4):
        for p1 in range(4):
            # Two-qubit Pauli
            P = np.kron(paulis[p0], paulis[p1])

            # Conjugate by U: U† P U
            P_evolved = U.conj().T @ P @ U

            # Decompose back into Pauli basis
            coeffs = []
            for q0 in range(4):
                for q1 in range(4):
                    Q = np.kron(paulis[q0], paulis[q1])
                    # Trace(Q† P_evolved) / 4
                    coeff = np.trace(Q.conj().T @ P_evolved) / 4
                    threshold_overlap =1e-12
                    if np.abs(coeff) > threshold_overlap:
                        coeffs.append(((q0, q1), coeff))

            action[(p0, p1)] = coeffs
    print(action)

    return action


def build_transition_matrix(action):
    """Build 16x16 transition matrix from action dictionary. Rows/cols: II,IX,IY,IZ,XI,..."""
    M = np.zeros((16, 16), dtype=complex)
    for (p0, p1), coeffs in action.items():
        row = p0 * 4 + p1
        for (q0, q1), coeff in coeffs:
            col = q0 * 4 + q1
            M[row, col] = coeff
    return M


def plot_transition_matrix(M, title="Pauli Transition Matrix"):
    """Plot magnitude and phase of transition matrix."""
    labels = [f"{a}{b}" for a in "IXYZ" for b in "IXYZ"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(np.abs(M), cmap='viridis')
    ax1.set_xticks(range(16));
    ax1.set_yticks(range(16))
    ax1.set_xticklabels(labels);
    ax1.set_yticklabels(labels)
    ax1.set_title("Magnitude");
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(np.angle(M), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax2.set_xticks(range(16));
    ax2.set_yticks(range(16))
    ax2.set_xticklabels(labels);
    ax2.set_yticklabels(labels)
    ax2.set_title("Phase");
    plt.colorbar(im2, ax=ax2)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    return fig


def plot_symbolic_matrix(M, title="Symbolic Pauli Transition Matrix"):
    """Plot symbolic matrix with LaTeX-rendered entries."""
    labels = [f"{a}{b}" for a in "IXYZ" for b in "IXYZ"]
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_xlim(-0.5, 15.5); ax.set_ylim(15.5, -0.5)
    ax.set_xticks(range(16)); ax.set_yticks(range(16))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Output Pauli"); ax.set_ylabel("Input Pauli")
    for i in range(17): ax.axhline(i-0.5, color='gray', lw=0.5); ax.axvline(i-0.5, color='gray', lw=0.5)
    for i in range(16):
        for j in range(16):
            expr = sp.nsimplify(M[i,j])
            if expr != 0:
                latex_str = sp.latex(expr)
                ax.text(j, i, f"${latex_str}$", ha='center', va='center', fontsize=6)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return fig


actionan = compute_pswap_pauli_action(2*np.pi)
M=build_transition_matrix(actionan)

print(M)
plot_transition_matrix(M)

theta = sp.Symbol('theta', real=True)
M_sym = compute_pswap_pauli_action_sympy(theta)
print("\nSymbolic transition matrix:")
sp.pprint(M_sym)
plot_symbolic_matrix(M_sym)