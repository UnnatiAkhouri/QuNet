import numpy as np

import src.measurements as measure
from src.setup import xp

import os

import src.density_matrix as DM
from src.random_unitary import random_energy_preserving_unitary


def run(dm: DM.DensityMatrix, num_iterations: int, order_rule, first_10_order, NM_orders_list, sub_unitary, connectivity,
        Unitaries=None, return_all_dms=False, verbose=False):
    all_dms = []
    if return_all_dms:
        all_dms.append(dm.copy())

    pops_values = {}
    two_qubit_dms = {}
    three_qubit_dms = {}
    four_qubit_dms = {}
    five_qubit_dms = {}
    six_qubit_dms = {}
    seven_qubit_dms = {}
    eight_qubit_dms = {}
    orders_list = []

    pops_values[0] = {index: pop for index, pop in enumerate(measure.pops(dm))}
    print
    two_qubit_dms[0] = measure.two_qbit_dm_of_every_pair(dm)
    num_qubits = len(measure.pops(dm))

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

    if NM_orders_list is not None and hasattr(order_rule, "__name__") and order_rule.__name__ == "NonMarkovian":
        for i in range(1, num_iterations + 1):
            pattern_dict = {
                'g': [[q, q + 1] for q in range(0, num_qubits - 1, 2)],
                'j': [[0, num_qubits - 1]] + [[j, j + 1] for j in range(1, num_qubits - 2, 2)]
            }
            letter = NM_orders_list[i]
            print(letter)
            order = pattern_dict[letter]
            print(order)
            orders_list.append(order)

            chunk_sizes = [len(chunk) for chunk in order]
            leftovers = dm.number_of_qbits % np.sum(chunk_sizes)
            if leftovers:
                leftover_identity = DM.Identity(DM.energy_basis(leftovers))

            if verbose and i / num_iterations * 100 % verbose == 0:
                percent = str(int(i / num_iterations * 100)).zfill(2)
                print(f"{percent}%")

            if generate_random_unitary:
                U = DM.tensor([random_energy_preserving_unitary(chunk_size) for chunk_size in chunk_sizes])
                if leftovers:
                    U = U.tensor(leftover_identity)
            else:
                U = Unitaries[i % num_unitaries]

            dm = step(dm, order, U, not generate_random_unitary)

            if return_all_dms:
                all_dms.append(dm.copy())

            pops_values[i] = {index: pop for index, pop in enumerate(measure.pops(dm))}
            two_qubit_dms[i] = measure.two_qbit_dm_of_every_pair(dm)

            if i > num_iterations - 5:
                two_qubit_dms[i] = measure.two_qbit_dm_of_every_pair(dm)
                three_qubit_dms[i] = measure.three_qbit_dm_of_every_triplet(dm)
                four_qubit_dms[i] = measure.four_qbit_dm_of_every_quartet(dm)
                five_qubit_dms[i] = measure.five_qbit_dm_of_every_quintet(dm)
                six_qubit_dms[i] = measure.six_qbit_dm_of_every_sextet(dm)
                seven_qubit_dms[i] = measure.seven_qbit_dm_of_every_seventet(dm)
                eight_qubit_dms[i] = measure.eight_qbit_dm_of_every_octet(dm)


    else:

        # Pre-calculate the order for the first step (i=1)

        if num_iterations >= 1:
            first_order = first_10_order[1] if 1 < len(first_10_order) else first_10_order[0]

            orders_list.append(first_order)

        for i in range(1, num_iterations + 1):

            # Use the order that was calculated in the previous iteration

            # (or pre-calculated for the first step)

            if i <= len(orders_list):

                order = orders_list[i - 1]  # orders_list is 0-indexed, steps are 1-indexed

            else:

                # Fallback (shouldn't happen if logic is correct)

                order = first_10_order[0]

            chunk_sizes = [len(chunk) for chunk in order]

            leftovers = dm.number_of_qbits % np.sum(chunk_sizes)

            if leftovers:
                leftover_identity = DM.Identity(DM.energy_basis(leftovers))

            if verbose and i / num_iterations * 100 % verbose == 0:
                percent = str(int(i / num_iterations * 100)).zfill(2)

                print(f"{percent}%")

            if generate_random_unitary:

                U = DM.tensor([random_energy_preserving_unitary(chunk_size) for chunk_size in chunk_sizes])

                if leftovers:
                    U = U.tensor(leftover_identity)

            else:

                U = Unitaries[i % num_unitaries]

            # Apply the quantum evolution step

            dm = step(dm, order, U, not generate_random_unitary)

            if return_all_dms:
                all_dms.append(dm.copy())

            # NOW we have the new pops values for step i

            pops_values[i] = {index: pop for index, pop in enumerate(measure.pops(dm))}

            two_qubit_dms[i] = measure.two_qbit_dm_of_every_pair(dm)

            if i > num_iterations - 5:
                two_qubit_dms[i] = measure.two_qbit_dm_of_every_pair(dm)

                three_qubit_dms[i] = measure.three_qbit_dm_of_every_triplet(dm)

                four_qubit_dms[i] = measure.four_qbit_dm_of_every_quartet(dm)

                five_qubit_dms[i] = measure.five_qbit_dm_of_every_quintet(dm)

                six_qubit_dms[i] = measure.six_qbit_dm_of_every_sextet(dm)

                seven_qubit_dms[i] = measure.seven_qbit_dm_of_every_seventet(dm)

                eight_qubit_dms[i] = measure.eight_qbit_dm_of_every_octet(dm)

            # Calculate the order for the NEXT step (i+1) using current and previous pops

            if i < num_iterations:  # Only if there's a next step

                next_step = i + 1

                if next_step < 10:

                    # For steps 2-9, use predefined orders

                    next_order = first_10_order[next_step] if next_step < len(first_10_order) else first_10_order[0]

                else:

                    # For step 10 and beyond, use order_rule with current (i) and previous (i-1) pops

                    next_order = order_rule(

                        order,  # Current order that was just used

                        pops_values[i - 1],  # Previous step pops

                        pops_values[i],  # Current step pops (just calculated)

                        two_qubit_dms[i - 1],  # Previous step two-qubit DMs

                        two_qubit_dms[i],  # Current step two-qubit DMs (just calculated)

                        connectivity,

                        sub_unitary,

                        dm

                    )

                orders_list.append(next_order)

    measurement_results = (
        pops_values, two_qubit_dms, three_qubit_dms, four_qubit_dms,
        five_qubit_dms, six_qubit_dms, seven_qubit_dms, eight_qubit_dms, orders_list
    )

    if return_all_dms:
        return measurement_results, all_dms
    else:
        return measurement_results, dm


def step(dm: DM.DensityMatrix, order: list[np.ndarray], Unitary: DM.DensityMatrix,
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
