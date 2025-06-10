import numpy as np

import src.measurements as measure
from src.setup import xp

import os

import src.density_matrix as DM
from src.random_unitary import random_energy_preserving_unitary
from src.channels import phase_covariant_kraus_operators, create_uncorrelated_2qubit_kraus,create_perfectly_correlated_2qubit_kraus,embed_edge_channel_full,apply_composite_edge_channel,two_qubit_depolarizing_kraus
from src.channels import CNOT_kraus

def run(dm: DM.DensityMatrix, num_iterations: int, order_rule, first_10_order, sub_unitary, connectivity,
        channel_prob: int,channels, Unitaries=None,return_all_dms=False, verbose=False):
    """
    Args:
        dm: the density matrix to evolve
        num_iterations: An integer representing the number of iterations the system will go through
        order_rule: a function that takes (past_order, prev_pops, pops, two_qubit_dms_previous, two_qubit_dms_current, connectivity, sub_unitary)
        first_10_order: order to use for the first 10 iterations
        sub_unitary: parameter for the order rule
        connectivity: parameter for the order rule
        Unitaries: either a list of DMs, a single unitary, or None for random unitaries
        return_all_dms: whether to return all intermediate density matrices
        verbose: a float or false. Progress reporting frequency

    Returns:
        tuple of (measurement_results, final_dm) if return_all_dms=False,
        or (measurement_results, list_of_all_dms) if return_all_dms=True
    """
    # Initialize storage for all density matrices if requested
    all_dms = []
    if return_all_dms:
        all_dms.append(dm.copy())  # Store initial dm

    # Initialize dictionaries for measurements
    pops_values = {}
    two_qubit_dms = {}
    three_qubit_dms = {}
    #four_qubit_dms = {}
    #five_qubit_dms = {}
    #six_qubit_dms = {}
    #seven_qubit_dms = {}
    orders_list = []  # To store orders used at each step

    # Compute initial measurements
    pops_values[0] = {index: pop for index, pop in enumerate(measure.pops(dm))}
    two_qubit_dms[0] = measure.two_qbit_dm_of_every_pair(dm)
    three_qubit_dms[0] = measure.three_qbit_dm_of_every_triplet(dm)
    #four_qubit_dms[0] = measure.four_qbit_dm_of_every_quartet(dm)

    # Set up unitaries
    generate_random_unitary = False
    if type(Unitaries) == list:
        assert len(Unitaries) == num_iterations, "There must be a unitary for each trial"
        num_unitaries = len(Unitaries)
    elif type(Unitaries) == DM.DensityMatrix:
        Unitaries = [Unitaries]
        num_unitaries = 1
    else:
        generate_random_unitary = True
        print("using random unitaries")

    # Initialize order to make sure it's defined
    if 1 in range(10):
        order = first_10_order[1]  # Get the first order
    else:
        # In case num_iterations <= 1, provide a default order
        # You may need to adjust this based on your specific requirements
        order = first_10_order[0] if first_10_order else []

    # Main evolution loop
    for i in range(1, num_iterations + 1):
        # Determine the order for partitioning
        if i < 10:
            order = first_10_order[i] if i < len(first_10_order) else order

        # Store the current order
        orders_list.append(order)

        chunk_sizes = [len(chunk) for chunk in order]
        leftovers = dm.number_of_qbits % np.sum(chunk_sizes)
        if leftovers:
            leftover_identity = DM.Identity(DM.energy_basis(leftovers))

        # Progress reporting
        if verbose and i / num_iterations * 100 % verbose == 0:
            percent = str(int(i / num_iterations * 100)).zfill(2)
            print(f"{percent}%")

        # Generate or select unitary
        if generate_random_unitary:
            U = DM.tensor([random_energy_preserving_unitary(chunk_size) for chunk_size in chunk_sizes])
            if leftovers:
                U = U.tensor(leftover_identity)
        else:
            U = Unitaries[i % num_unitaries]

        # Evolve the system
        dm = step(dm, order, U, channels, channel_prob, not generate_random_unitary)

        # Store the evolved dm if requested
        if return_all_dms:
            all_dms.append(dm.copy())

        # Always measure populations and two-qubit density matrices
        # This ensures they're available for the order rule
        pops_values[i] = {index: pop for index, pop in enumerate(measure.pops(dm))}
        two_qubit_dms[i] = measure.two_qbit_dm_of_every_pair(dm)

        # Only compute additional measurements for the last 5 steps
        if i > num_iterations - 5:
            # You can add additional measurements here
            two_qubit_dms[i] = measure.two_qbit_dm_of_every_pair(dm)
            three_qubit_dms[i] = measure.three_qbit_dm_of_every_triplet(dm)
            #four_qubit_dms[i] = measure.four_qbit_dm_of_every_quartet(dm)
            #five_qubit_dms[i] = measure.five_qbit_dm_of_every_quintet(dm)
            #six_qubit_dms[i] = measure.six_qbit_dm_of_every_sextet(dm)
            #seven_qubit_dms[i] = measure.seven_qbit_dm_of_every_seventet(dm)
            pass

        # Calculate the next order for iterations after 10
        previous_order = order
        if i >= 9 and i < num_iterations:  # Up to but not including the last iteration
            next_i = i + 1
            if next_i >= 10:  # Only use order_rule for iterations >= 10
                order = order_rule(previous_order, pops_values[i - 1], pops_values[i],
                                   two_qubit_dms[i - 1], two_qubit_dms[i],
                                   connectivity, sub_unitary, dm)

    # Return results based on whether all DMs were requested
    #measurement_results = (pops_values, two_qubit_dms,three_qubit_dms,four_qubit_dms, five_qubit_dms,six_qubit_dms,seven_qubit_dms, orders_list)
    measurement_results = (pops_values, two_qubit_dms, three_qubit_dms,orders_list)

    if return_all_dms:
        return measurement_results, all_dms
    else:
        return measurement_results, dm


def step(dm: DM.DensityMatrix, order: list[np.ndarray], Unitary: DM.DensityMatrix,channels,channel_prob:int=0.2,
         unitary_reused=False) -> DM.DensityMatrix:
    """
    Args:
        dm: the density matrix to evolve
        order: the qbit order to be used e.g. [0,2,1,3]
        Unitary: A Unitary that will be used to evolve the system
        unitary_reused: if the unitary will be reused make sure to undo the reordering

    Returns: A density matrix that has been evolved by the given hamiltonians for the given step sizes.

    """
    # make sure each qbit is assigned to a group and that there are no extras or duplicates.
    # flatten order using a list comprehension
    order = [qbit for chunk in order for qbit in chunk]

    assert set(list(order)) == set(range(dm.number_of_qbits)), f"{set(order)} vs {set(range(dm.number_of_qbits))}"
    #print(f"Unitary before relabeling:\n{Unitary}")
    Unitary.relabel_basis(order)
    #print(f"Unitary after relabeling:\n{Unitary}")

    Unitary.change_to_energy_basis()
    dm.change_to_energy_basis()
    dm = Unitary * dm * Unitary.H

    if np.random.rand() < channel_prob:
        dm = apply_composite_edge_channel(dm, channels, dm.number_of_qbits)

    if unitary_reused:
        inverse_order = list(range(len(order)))
        for i, value in enumerate(order):
            inverse_order[value] = i
        Unitary.relabel_basis(inverse_order)

    return dm



def save_data(data: np.ndarray, num_qbits: str, measurement: str, num_chunks: str, connectivity_type: str,
              run_index: str, sim_index=int, extra=""):
    if extra != "":
        path = f"../data/num_qbits={num_qbits}_num_chunks={num_chunks}_connectivity_type={connectivity_type}_other={extra}/index={sim_index}"
    else:
        path = f"../data/num_qbits={num_qbits}_num_chunks={num_chunks}_connectivity_type={connectivity_type}/index={sim_index}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    file_name = path + f"/{measurement}_{run_index}.dat"
    np.savetxt(file_name, data,
               header=f"{measurement} for {num_qbits} qbits with connectivity {connectivity_type} in chunks {num_chunks}")
