# For spin systems, operators can be represented as:
# - Dense matrices (small systems)
# - Sparse matrices (medium systems)
# - Matrix Product Operators (large systems)
import numpy as np
from scipy.linalg import expm


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
    
    def modified_gram_schmidt_krylov(self, max_iterations):
        """
        Modified Gram-Schmidt for Krylov operators
        """
        # Initialize
        krylov_ops = [self.O0 / self.norm(self.O0)]  # |𝒪₀⟩
        current_op = self.O0

        lanczos_a = []  # diagonal coefficients
        lanczos_b = []  # off-diagonal coefficients

        for n in range(1, max_iterations):
            # Generate next operator
            next_op = self.apply_unitary(current_op)  # U†OₙU

            # Modified Gram-Schmidt: remove projections sequentially
            for i in range(len(krylov_ops)):
                proj = self.inner_product(krylov_ops[i], next_op)
                next_op = next_op - proj * krylov_ops[i]

                # Store coefficients
                if i == len(krylov_ops) - 1:  # Last projection
                    if len(krylov_ops) == 1:
                        lanczos_a.append(proj)  # a₀
                    else:
                        lanczos_a.append(proj)  # aₙ₋₁

            # Calculate norm and normalize
            norm_val = self.norm(next_op)
            lanczos_b.append(norm_val)

            if norm_val < 1e-12:  # Krylov space exhausted
                break

            krylov_ops.append(next_op / norm_val)
            current_op = self.apply_unitary(krylov_ops[-2])  # For next iteration

        return krylov_ops, lanczos_a, lanczos_b


    def compute_krylov_complexity(self, t_max, krylov_ops):
        complexity = []
        all_coefficients = []
        current_op = self.O0 

        for t in range(t_max + 1):
            coefficients = []
            for n, krylov_op in enumerate(krylov_ops):
                coeff = self.inner_product(krylov_op, current_op)
                coefficients.append(coeff)
            all_coefficients.append(coefficients)  # Collect for each t

            K_t = sum(n * abs(coeff) ** 2 for n, coeff in enumerate(coefficients))
            complexity.append(K_t)

            if t < t_max:
                current_op = self.apply_unitary(current_op)

        return complexity, all_coefficients
    
    def autocorrelation(self, t_max):
        """
        Compute autocorrelation function ⟨O₀|O(t)⟩
        """
        autocorr = []
        current_op = self.O0.copy()

        for t in range(t_max + 1):
            overlap = self.inner_product(self.O0, current_op)
            autocorr.append(overlap)
            if t < t_max:
                current_op = self.apply_unitary(current_op)

        return np.array(autocorr)
    

    def analyze(self, t_max):
        """
        Complete Krylov complexity analysis for quantum circuit
        """

        # Perform Gram-Schmidt orthonormalization
        krylov_ops, a_coeffs, b_coeffs = self.modified_gram_schmidt_krylov(
            t_max + 1
        )

        # Compute Krylov complexity
        complexity, coeffs = self.compute_krylov_complexity(
            t_max, krylov_ops
        )

        return {
            'complexity': complexity,
            'krylov_operators': krylov_ops,
            'lanczos_a': a_coeffs,
            'lanczos_b': b_coeffs,
            'expansion_coefficients': coeffs
        }