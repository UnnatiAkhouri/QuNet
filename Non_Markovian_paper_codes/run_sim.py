"""
This code is the main runner for operator spreading simulations. This code relies on the functions described in core
Any code changes made in code eg. adding KRYLOV COMPLEXITY CODE, should be updated here.
The inputs/outputs will be CSV input/output. This is so you could use pandas later on.
You can tweak the number of runs you want to do. CURRENTLY IT USES THE NUMBER OF ROWS FROM NM LIST.
"""

import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path
from typing import List, Tuple
import time

from core import run_simulation
def parse_theta_string(theta_str: str) -> Tuple[float, float]:
    """
    <THIS IS EXTREMELY IMPORTANT>
    THE INPUT FOR THIS CODE NEEDS COLLAPSED SEQUENCES THAT INDICATE THE TYPE OF LAYER IT IS ie g or j
    AND THE NUMBER OF CONSECUTIVE LAYERS
    Parse theta string into (theta_even, theta_odd).

    Supported formats:
    1. "(26, 'j')" or "(26, 'g')" - coefficient × π/15 or whatever
    2. "(26, 'j'), (13, 'g')" - both in one cell

    - j = even layer
    - g = odd layer

    Example: "(26, 'j')" -> angle = 26 * angle for even layer
    """
    import re

    theta_even = 0.0
    theta_odd = 0.0

#Pattern matching
    pattern_new = r"\((\d+)\s*,\s*['\"]([jg])['\"]\)"
    matches_new = re.findall(pattern_new, theta_str)

    if matches_new:
        # New format found
        for match in matches_new:
            coeff, gate_type = match
            angle = float(coeff) * np.pi/300.0

            if gate_type == 'j':
                theta_even = angle
            elif gate_type == 'g':
                theta_odd = angle
        return theta_even, theta_odd

#other formats can also work
    pattern_legacy = r'([jg])\((\d+)/(\d+)\)'
    matches_legacy = re.findall(pattern_legacy, theta_str)

    if matches_legacy:
        for match in matches_legacy:
            gate_type, num, denom = match
            angle = float(num) / float(denom) * np.pi

            if gate_type == 'j':
                theta_even = angle
            elif gate_type == 'g':
                theta_odd = angle
        return theta_even, theta_odd

    # BREAK
    return theta_even, theta_odd


def load_theta_sequences(csv_path: str) -> List[List[Tuple[float, float]]]:
    """
    Load theta sequences from CSV file.

    Expected format:
    - Row 1: Gate types (j,g,j,g,j,g,...)
    - Row 2+: Coefficients for each run (26,1,1,1,3,1,...)

    Each pair of columns (j,g) forms one time step.
    Angle = coefficient × π/15

    Returns list of sequences, where each sequence is [(theta_even, theta_odd), ...]
    """
    import ast

    try:
        with open(csv_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return []

        # Check if first row contains j/g pattern
        first_row = lines[0].split(',')
        first_row = [x.strip().lower() for x in first_row]

        if first_row[0] in ['j', 'g']:
            # New format: first row is gate types
            gate_types = first_row

            sequences = []
            for line in lines[1:]:  # Skip header row
                coeffs = line.split(',')
                coeffs = [x.strip() for x in coeffs]

                seq = []
                theta_even = 0.0
                theta_odd = 0.0

                for i, coeff_str in enumerate(coeffs):
                    if not coeff_str:
                        continue

                    try:
                        coeff = float(coeff_str)
                    except ValueError:
                        continue

                    angle = coeff * np.pi/300.0
                    gate_type = gate_types[i] if i < len(gate_types) else 'j'

                    if gate_type == 'j':
                        theta_even = angle
                    elif gate_type == 'g':
                        theta_odd = angle
                        # After seeing 'g', we have a complete time step
                        seq.append((theta_even, theta_odd))
                        theta_even = 0.0
                        theta_odd = 0.0

                # If there's a trailing j without g, add it with g=0
                if theta_even != 0.0:
                    seq.append((theta_even, 0.0))

                if seq:
                    sequences.append(seq)

            return sequences
#CHECKS ARE INBUILT SO IF YOU CODE DOES NOT HAVE THE CORRECT COLLAPSED SEQUENCES, IT WILL FAIL
    except Exception as e:
        print(f"Note: Could not parse as new format, trying other formats: {e}")

    # Try list format: [(26, 'j'), (1, 'g'), ...]
    sequences = []

    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                # Try to parse as a Python list
                entries = ast.literal_eval(line)

                if not isinstance(entries, list):
                    entries = [entries]

                seq = []
                theta_even = 0.0
                theta_odd = 0.0

                for coeff, gate_type in entries:
                    angle = float(coeff) * np.pi/300.0

                    if gate_type == 'j':
                        theta_even = angle
                    elif gate_type == 'g':
                        theta_odd = angle
                        seq.append((theta_even, theta_odd))
                        theta_even = 0.0
                        theta_odd = 0.0

                if theta_even != 0.0:
                    seq.append((theta_even, 0.0))

                if seq:
                    sequences.append(seq)

            except (ValueError, SyntaxError):
                # Try old cell-based CSV format
                pass

    # Fall back to old CSV parsing if nothing worked
    if not sequences:
        df = pd.read_csv(csv_path, header=None)
        for _, row in df.iterrows():
            seq = []
            for cell in row:
                if pd.isna(cell):
                    continue
                theta_even, theta_odd = parse_theta_string(str(cell))
                seq.append((theta_even, theta_odd))
            if seq:
                sequences.append(seq)
    print(sequences)
    return sequences


def save_results(results: dict, output_dir: str, run_id: int):
    """Save results from a single run to CSV files. CURRENT THIS IS DONE PER RULE Per run i.e. all the stats per rule per run in one csv"""
    os.makedirs(output_dir, exist_ok=True)

    # Hamming weight time series
    df_weight = pd.DataFrame({
        'time_step': results['time_steps'],
        'avg_weight': results['avg_weight'],
        'std_weight': results['std_weight'],
        'num_strings': results['num_strings'],
        'kept_norm': results['kept_norm'],
        'total_norm': results['total_norm']
    })
    df_weight.to_csv(f"{output_dir}/run_{run_id:03d}_hamming_weight.csv", index=False)

    # Site density time series
    L = results['L']
    site_cols = {f'site_{i}': results['site_density'][:, i] for i in range(L)}
    site_cols['time_step'] = results['time_steps']
    df_site = pd.DataFrame(site_cols)
    # Reorder columns
    cols = ['time_step'] + [f'site_{i}' for i in range(L)]
    df_site = df_site[cols]
    df_site.to_csv(f"{output_dir}/run_{run_id:03d}_site_density.csv", index=False)

    # Weight distribution time series
    weight_cols = {f'weight_{i}': results['weight_dist'][:, i] for i in range(L + 1)}
    weight_cols['time_step'] = results['time_steps']
    df_wdist = pd.DataFrame(weight_cols)
    cols = ['time_step'] + [f'weight_{i}' for i in range(L + 1)]
    df_wdist = df_wdist[cols]
    df_wdist.to_csv(f"{output_dir}/run_{run_id:03d}_weight_dist.csv", index=False)

    # Arrival times
    df_arrival = pd.DataFrame({
        'site': list(range(L)),
        'arrival_time': results['arrival_times'],
        'distance_from_center': [min(abs(i - results['initial_site']),
                                     L - abs(i - results['initial_site']))
                                 for i in range(L)]
    })
    df_arrival.to_csv(f"{output_dir}/run_{run_id:03d}_arrival_times.csv", index=False)


def compute_aggregate_statistics(all_results: List[dict], output_dir: str):
    """Compute and save aggregate statistics across all runs. This is average stats of all runs per rule"""
    os.makedirs(output_dir, exist_ok=True)

    # Find common time range
    min_steps = min(len(r['time_steps']) for r in all_results)

    # Stack arrays
    avg_weights = np.array([r['avg_weight'][:min_steps] for r in all_results])
    std_weights = np.array([r['std_weight'][:min_steps] for r in all_results])
    num_strings = np.array([r['num_strings'][:min_steps] for r in all_results])
    kept_norms = np.array([r['kept_norm'][:min_steps] for r in all_results])

    L = all_results[0]['L']
    site_densities = np.array([r['site_density'][:min_steps] for r in all_results])

    # Compute statistics
    df_stats = pd.DataFrame({
        'time_step': list(range(min_steps)),
        'avg_weight_mean': np.mean(avg_weights, axis=0),
        'avg_weight_std': np.std(avg_weights, axis=0),
        'avg_weight_min': np.min(avg_weights, axis=0),
        'avg_weight_max': np.max(avg_weights, axis=0),
        'num_strings_mean': np.mean(num_strings, axis=0),
        'num_strings_std': np.std(num_strings, axis=0),
        'kept_norm_mean': np.mean(kept_norms, axis=0),
        'kept_norm_std': np.std(kept_norms, axis=0),
    })
    df_stats.to_csv(f"{output_dir}/aggregate_hamming_weight.csv", index=False)

    # Aggregate site density
    site_mean = np.mean(site_densities, axis=0)
    site_std = np.std(site_densities, axis=0)

    # Save mean
    site_cols = {f'site_{i}': site_mean[:, i] for i in range(L)}
    site_cols['time_step'] = list(range(min_steps))
    df_site_mean = pd.DataFrame(site_cols)
    cols = ['time_step'] + [f'site_{i}' for i in range(L)]
    df_site_mean = df_site_mean[cols]
    df_site_mean.to_csv(f"{output_dir}/aggregate_site_density_mean.csv", index=False)

    # Save std
    site_cols = {f'site_{i}': site_std[:, i] for i in range(L)}
    site_cols['time_step'] = list(range(min_steps))
    df_site_std = pd.DataFrame(site_cols)
    df_site_std = df_site_std[cols]
    df_site_std.to_csv(f"{output_dir}/aggregate_site_density_std.csv", index=False)

    # Arrival time statistics
    arrival_times = np.array([r['arrival_times'] for r in all_results])

    df_arrival_stats = pd.DataFrame({
        'site': list(range(L)),
        'arrival_time_mean': np.mean(arrival_times, axis=0),
        'arrival_time_std': np.std(arrival_times, axis=0),
        'arrival_time_min': np.min(arrival_times, axis=0),
        'arrival_time_max': np.max(arrival_times, axis=0),
        'distance_from_center': [min(abs(i - all_results[0]['initial_site']),
                                     L - abs(i - all_results[0]['initial_site']))
                                 for i in range(L)]
    })
    df_arrival_stats.to_csv(f"{output_dir}/aggregate_arrival_times.csv", index=False)

    print(f"Aggregate statistics saved to {output_dir}/")





def main():
    parser = argparse.ArgumentParser(description='Operator Spreading Simulation')
    parser.add_argument('--theta_csv', type=str, required=True,
                        help='CSV file with theta sequences')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--L', type=int, default=6,
                        help='System size (number of qubits)')
    parser.add_argument('--threshold', type=float, default=1e-8,
                        help='Truncation threshold')
    parser.add_argument('--stop_at_boundary', action='store_true',
                        help='Stop when front reaches boundary')
    parser.add_argument('--arrival_threshold', type=float, default=1e-8,
                        help='Threshold for site arrival')
    parser.add_argument('--max_runs', type=int, default=None,
                        help='Maximum number of runs to process')

    args = parser.parse_args()

    #The print statements will help debug where the code is failing

    print(f"Loading theta sequences from {args.theta_csv}...")
    sequences = load_theta_sequences(args.theta_csv)
    print(f"Loaded {len(sequences)} sequences")

    if args.max_runs:
        sequences = sequences[:args.max_runs]
        print(f"Using first {args.max_runs} sequences")

    # Run all simulations
    print(f"\nRunning {len(sequences)} simulations with L={args.L}...")
    all_results = []

    for i, seq in enumerate(sequences):
        start_time = time.time()

        results = run_simulation(
            L=args.L,
            theta_sequence=seq,
            threshold=args.threshold,
            stop_at_boundary=args.stop_at_boundary,
            arrival_threshold=args.arrival_threshold
        )

        elapsed = time.time() - start_time

        # Save individual results
        save_results(results, f"{args.output_dir}/individual", i)
        all_results.append(results)

        print(f"  Run {i+1}/{len(sequences)}: {len(results['time_steps'])} steps, "
              f"{results['num_strings'][-1]} strings, {elapsed:.2f}s")

    # Compute aggregate statistics
    print("\nComputing aggregate statistics...")
    compute_aggregate_statistics(all_results, f"{args.output_dir}/aggregate")

    # Save metadata
    metadata = {
        'parameter': ['L', 'threshold', 'arrival_threshold', 'num_runs',
                      'stop_at_boundary', 'theta_csv'],
        'value': [args.L, args.threshold, args.arrival_threshold, len(sequences),
                  args.stop_at_boundary, args.theta_csv]
    }
    pd.DataFrame(metadata).to_csv(f"{args.output_dir}/metadata.csv", index=False)

    print(f"\nAll results saved to {args.output_dir}/")
    print("Done!")

#because the data is stored in df format, we can easily use pandas and sci-kit learn.

if __name__ == '__main__':
    main()