"""
Upload to github
Here we define the core module for operator spreading simulation using sparse Pauli dynamics.
The goal is to use knowledge of what conjugation of Pauli operators result in in terns of letters rather than doing actual matrix multiplications.
The code below implements Pauli string encoding, PSWAP gate action, and sparse operator evolution.
"""
import numpy as np
from numba import njit
from collections import defaultdict
from typing import Dict, Tuple, List


# =============================================================================
# Pauli String Encoding
# =============================================================================
# We encode a Pauli string on L qubits using 2L bits:
# - Lower L bits: z part (1 if Z or Y present)
# - Upper L bits: x part (1 if X or Y present)
# tiny x and z represent the 2 bit strings whereas the capital X,Y,Z and I represent the Pauli matrices
# x and z can both take value 0 or 1 and the mappings are:
# I = (0,0), X = (1,0), Y = (1,1), Z = (0,1)
# Example: XYZ on 3 qubits = X_0 Y_1 Z_2
#   x part: 110 (binary) = 6
#   z part: 011 (binary) = 3
#   Full encoding: (6 << 3) | 3 = 51

@njit
def pauli_at_site(pauli_int: int, site: int, L: int) -> int:
    """Get Pauli type at a specific site: 0=I, 1=X, 2=Y, 3=Z"""
    # we pull out the int at the site. We do this by right shifting the pauli in at site and site +L. Then we compare it with 1 using the &1 operation which will either return the pauli int if its 1 or give 0 if Pauli int is 0
    # for a system of L qubits, the integer pauli_int has:
    #Bits 0 to L-1 → the z-bits for each site
    #Bits L to 2L-1 → the x-bits for each site
    x_bit = (pauli_int >> (L + site)) & 1
    z_bit = (pauli_int >> site) & 1
    # Mapping: (x=0,z=0)->I=0, (x=1,z=0)->X=1, (x=1,z=1)->Y=2, (x=0,z=1)->Z=3
    #this maps the pauli_int encoding back to the Pauli matrices.
    if x_bit == 0 and z_bit == 0:
        return 0  # I
    elif x_bit == 1 and z_bit == 0:
        return 1  # X
    elif x_bit == 1 and z_bit == 1:
        return 2  # Y
    else:  # x_bit == 0 and z_bit == 1
        return 3  # Z


@njit
def set_pauli_at_site(pauli_int: int, site: int, L: int, pauli_type: int) -> int:
    """Set Pauli type at a specific site (0=I, 1=X, 2=Y, 3=Z)"""
    #We ar eusing the symplectic represention for qubits
    # Clear existing bits
    # x-bit and z-bit for that "site" get zeroed out
    x_mask = ~(1 << (L + site))
    z_mask = ~(1 << site)
    pauli_int = pauli_int & x_mask & z_mask

    # Set new bits based on pauli_type
    # I=0: (0,0), X=1: (1,0), Y=2: (1,1), Z=3: (0,1)
    if pauli_type == 1:  # X
        pauli_int |= (1 << (L + site))
    elif pauli_type == 2:  # Y
        pauli_int |= (1 << (L + site))
        pauli_int |= (1 << site)
    elif pauli_type == 3:  # Z
        pauli_int |= (1 << site)
    # pauli_type == 0 (I): both bits stay 0
    return pauli_int


@njit
def hamming_weight(pauli_int: int, L: int) -> int:
    """Count number of non-identity Paulis"""
    #this counts the number of non-identity term in the stirng
    x_part = pauli_int >> L
    z_part = pauli_int & ((1 << L) - 1)
    # A site is non-trivial if either x or z bit is set
    non_trivial = x_part | z_part
    count = 0
    while non_trivial:
        count += non_trivial & 1
        non_trivial >>= 1
    return count


@njit
def site_has_nontrivial(pauli_int: int, site: int, L: int) -> bool:
    #checks if a particular site has non-identity term
    """Check if site has non-identity Pauli"""
    x_bit = (pauli_int >> (L + site)) & 1
    z_bit = (pauli_int >> site) & 1
    return (x_bit | z_bit) == 1


#Now we define the gate. We use the Pswap gate. The gate we study is parametrized below, however a general 2Q U(1) may also be defined here
#Define the general U(1) later TODO
# =============================================================================
# PSWAP Gate Action in Pauli Basis
# =============================================================================
# The gate U(θ) = exp(-iθ/4 (XX + YY)) acts on computational basis as:
# |00⟩ → |00⟩
# |01⟩ → cos(θ/2)|01⟩ + i·sin(θ/2)|10⟩
# |10⟩ → i·sin(θ/2)|01⟩ + cos(θ/2)|10⟩
# |11⟩ → |11⟩

def compute_pswap_pauli_action(theta: float) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], complex]]]:
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
                    threshold_overlap =1e-12 #SET THRESHOLD ACCORDING TO YOUR CHOICE/STANDARDS!!! TODO adaptive threshold
                    if np.abs(coeff) > threshold_overlap:
                        coeffs.append(((q0, q1), coeff))
            action[(p0, p1)] = coeffs
    return action


def apply_two_qubit_gate(operator: Dict[int, complex],
                         site0: int, site1: int,
                         L: int,
                         gate_action: Dict) -> Dict[int, complex]:
    """
    Apply a two-qubit gate to the sparse operator.

    Args:
        operator: Dictionary mapping pauli_int -> coefficient
        site0, site1: Sites where gate acts
        L: System size
        gate_action: Precomputed gate action from compute_pswap_pauli_action

    Returns:
        New operator dictionary
    """
    #here we use the above computation of how pauli matrices evolve under pswap and use that to define new set of Pauli ints
    new_operator = defaultdict(complex)

    for pauli_int, coeff in operator.items():
        # Get Paulis at the two sites
        p0 = pauli_at_site(pauli_int, site0, L)
        p1 = pauli_at_site(pauli_int, site1, L)

        # Get the transformation
        for (q0, q1), gate_coeff in gate_action[(p0, p1)]:
            # Build new Pauli string
            new_pauli = pauli_int
            new_pauli = set_pauli_at_site(new_pauli, site0, L, q0)
            new_pauli = set_pauli_at_site(new_pauli, site1, L, q1)

            new_operator[new_pauli] += coeff * gate_coeff

    return dict(new_operator)


def truncate_operator(operator: Dict[int, complex],
                      threshold: float) -> Tuple[Dict[int, complex], float, float]:
    """
    Truncate small coefficients from operator.

    Returns:
        (truncated_operator, kept_norm, total_norm)
    """
    total_norm = sum(np.abs(c) ** 2 for c in operator.values())

    truncated = {k: v for k, v in operator.items() if np.abs(v) ** 2 >= threshold ** 2}
    kept_norm = sum(np.abs(c) ** 2 for c in truncated.values())

    return truncated, kept_norm, total_norm


# =============================================================================
# Brickwork Circuit Evolution
# =============================================================================

def apply_even_layer(operator: Dict[int, complex], L: int, theta: float) -> Dict[int, complex]:
    """Apply gates on pairs (0,1), (2,3), ..."""
    gate_action = compute_pswap_pauli_action(theta)

    for i in range(0, L, 2):
        j = (i + 1) % L
        operator = apply_two_qubit_gate(operator, i, j, L, gate_action)

    return operator


def apply_odd_layer(operator: Dict[int, complex], L: int, theta: float) -> Dict[int, complex]:
    """Apply gates on pairs (1,2), (3,4), ..., (L-1, 0) for PBC"""
    gate_action = compute_pswap_pauli_action(theta)

    for i in range(1, L, 2):
        j = (i + 1) % L
        operator = apply_two_qubit_gate(operator, i, j, L, gate_action)

    return operator


def evolve_one_step(operator: Dict[int, complex],
                    L: int,
                    theta_even: float,
                    theta_odd: float,
                    threshold: float) -> Tuple[Dict[int, complex], float, float]:
    """
    Apply one full time step (even + odd layer) with truncation.

    Returns:
        (evolved_operator, kept_norm, total_norm)
    """
    # Even layer
    operator = apply_even_layer(operator, L, theta_even)

    # Odd layer
    operator = apply_odd_layer(operator, L, theta_odd)

    # Truncate
    operator, kept_norm, total_norm = truncate_operator(operator, threshold)

    return operator, kept_norm, total_norm


# =============================================================================
# Observable Computation
# =============================================================================

def compute_observables(operator: Dict[int, complex], L: int) -> dict:
    """
    Here we compute all observables from the current operator state; this list can be increased to include other observables--GS orthogonality etc for Krylov.

    Returns dictionary with:
        - avg_weight: average Hamming weight
        - std_weight: std of Hamming weight
        - num_strings: number of Pauli strings
        - weight_dist: distribution over weights 0 to L...ie how much probability is in strings of hamming weight 0,1,2,...up to L
        - site_density: probability of non-trivial Pauli at each site
    """
    weights = []
    probs = []
    site_density = np.zeros(L)
    weight_dist = np.zeros(L + 1)

    for pauli_int, coeff in operator.items():
        prob = np.abs(coeff) ** 2
        w = hamming_weight(pauli_int, L)

        weights.append(w)
        probs.append(prob)
        weight_dist[w] += prob

        # Site density
        for site in range(L):
            if site_has_nontrivial(pauli_int, site, L):
                site_density[site] += prob

    weights = np.array(weights)
    probs = np.array(probs)

    avg_weight = np.sum(weights * probs) #computes the expectation value of the hamming weight operator ie hamming weight of a string P_i times the c_i^2 ie coeff squared of P_i.
    var_weight = np.sum((weights - avg_weight) ** 2 * probs)
    std_weight = np.sqrt(var_weight)

    return {
        'avg_weight': avg_weight,
        'std_weight': std_weight,
        'num_strings': len(operator),
        'weight_dist': weight_dist,
        'site_density': site_density
    }


def create_initial_operator(L: int, site: int) -> Dict[int, complex]:
    """Create initial Z operator at specified site."""
    # Z at site: X part = 0, Z part = 1 at that site
    pauli_int = 1 << site
    return {pauli_int: 1.0 + 0j}


# =============================================================================
# Main Simulation Function
# =============================================================================

def run_simulation(L: int,
                   theta_sequence: List[Tuple[float, float]],
                   threshold: float = 1e-8,
                   initial_site: int = None,
                   stop_at_boundary: bool = True,
                   arrival_threshold: float = 1e-8) -> dict:
    """
    Run the full operator spreading simulation. Note you can mke this into a density matrix run if you pre define the initial opertor to be linear combo of Pauli strings with appropriate weights
    TODO: make a function that generates initial operator as linear conbo of Pauli strings.
    Args:
        L: System size
        theta_sequence: List of (theta_even, theta_odd) for each time step
        threshold: Truncation threshold for coefficients
        initial_site: Site for initial Z operator (default: center)
        stop_at_boundary: If True, stop when front reaches boundary
        arrival_threshold: Threshold for defining "arrival" at a site

    Returns:
        Dictionary with all time series data
    """
    if initial_site is None:
        initial_site = L // 2

    # Initialize
    operator = create_initial_operator(L, initial_site)

    # Storage for results. UPDATE THIS IF YOU UPDATE THE COMPUTE OBSERVABLES TO INCLUDE KRYLOV
    results = {
        'L': L,
        'threshold': threshold,
        'initial_site': initial_site,
        'num_steps': len(theta_sequence),
        'time_steps': [],
        'avg_weight': [],
        'std_weight': [],
        'num_strings': [],
        'kept_norm': [],
        'total_norm': [],
        'weight_dist': [],
        'site_density': [],
        'arrival_times': np.full(L, -1, dtype=int)  # -1 means not yet arrived
    }

    # Record initial state
    obs = compute_observables(operator, L)
    results['time_steps'].append(0)
    results['avg_weight'].append(obs['avg_weight'])
    results['std_weight'].append(obs['std_weight'])
    results['num_strings'].append(obs['num_strings'])
    results['kept_norm'].append(1.0)
    results['total_norm'].append(1.0)
    results['weight_dist'].append(obs['weight_dist'].copy())
    results['site_density'].append(obs['site_density'].copy())

    # Mark initial site as arrived at t=0
    results['arrival_times'][initial_site] = 0

    # Evolution
    for t, (theta_even, theta_odd) in enumerate(theta_sequence, 1):
        operator, kept_norm, total_norm = evolve_one_step(
            operator, L, theta_even, theta_odd, threshold
        )

        obs = compute_observables(operator, L)

        results['time_steps'].append(t)
        results['avg_weight'].append(obs['avg_weight'])
        results['std_weight'].append(obs['std_weight'])
        results['num_strings'].append(obs['num_strings'])
        results['kept_norm'].append(kept_norm)
        results['total_norm'].append(total_norm)
        results['weight_dist'].append(obs['weight_dist'].copy())
        results['site_density'].append(obs['site_density'].copy())

        # Update arrival times
        for site in range(L):
            if results['arrival_times'][site] == -1:
                if obs['site_density'][site] >= arrival_threshold:
                    results['arrival_times'][site] = t

        # Check stopping condition
        if stop_at_boundary:
            # Check if front has reached the furthest sites from center
            max_dist = L // 2  # For PBC, max distance is L/2
            all_arrived = True
            for site in range(L):
                dist = min(abs(site - initial_site), L - abs(site - initial_site))
                if dist == max_dist and results['arrival_times'][site] == -1:
                    all_arrived = False
                    break

            if all_arrived:
                break

    # Convert lists to arrays
    results['avg_weight'] = np.array(results['avg_weight'])
    results['std_weight'] = np.array(results['std_weight'])
    results['num_strings'] = np.array(results['num_strings'])
    results['kept_norm'] = np.array(results['kept_norm'])
    results['total_norm'] = np.array(results['total_norm'])
    results['weight_dist'] = np.array(results['weight_dist'])
    results['site_density'] = np.array(results['site_density'])

    return results