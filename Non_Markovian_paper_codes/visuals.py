"""
Visualization script for operator spreading results.
THIS IS A HELPER CODE AND PLOTS PER RULE TYPE. USE COMPARE_RUNS_ENHANCED TO GENERATE PLOTS WITH MULTIPLE FIGS ON SAME
Creates plots for Hamming weight, site density, and random walk analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from pathlib import Path


def plot_hamming_weight(input_dir: str, output_dir: str):
    """Plot average Hamming weight vs time with individual runs and aggregate."""

    # Load aggregate data
    agg_df = pd.read_csv(f"{input_dir}/aggregate/aggregate_hamming_weight.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Aggregate with error bands
    ax = axes[0]
    t = agg_df['time_step']
    mean = agg_df['avg_weight_mean']
    std = agg_df['avg_weight_std']

    ax.plot(t, mean, 'b-', linewidth=2, label='Mean')
    ax.fill_between(t, mean - std, mean + std, alpha=0.3, color='blue', label='±1 std')
    ax.fill_between(t, agg_df['avg_weight_min'], agg_df['avg_weight_max'],
                    alpha=0.1, color='blue', label='Min-Max')

    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Average Hamming weight', fontsize=12)
    ax.set_title('Hamming Weight Growth (Aggregate)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Individual runs
    ax = axes[1]
    individual_dir = f"{input_dir}/individual"

    # Find all individual run files
    run_files = sorted(Path(individual_dir).glob("run_*_hamming_weight.csv"))

    colors = plt.cm.viridis(np.linspace(0, 1, len(run_files)))

    for i, f in enumerate(run_files):
        df = pd.read_csv(f)
        ax.plot(df['time_step'], df['avg_weight'], color=colors[i],
                alpha=0.7, linewidth=1, label=f'Run {i}' if i < 5 else None)

    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Average Hamming weight', fontsize=12)
    ax.set_title('Hamming Weight Growth (Individual Runs)', fontsize=14)
    if len(run_files) <= 10:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/hamming_weight.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/hamming_weight.png")


def plot_site_density_heatmap(input_dir: str, output_dir: str, L: int):
    """Plot site density as a heatmap over time."""

    # Load aggregate mean
    df = pd.read_csv(f"{input_dir}/aggregate/aggregate_site_density_mean.csv")

    # Extract density matrix
    times = df['time_step'].values

    # Limit to time steps up to 200
    max_time_idx = np.where(times <= 200)[0]
    if len(max_time_idx) > 0:
        max_idx = max_time_idx[-1] + 1
    else:
        max_idx = len(times)

    times = times[:max_idx]
    density = np.array([[df[f'site_{i}'].iloc[t] for i in range(L)] for t in range(len(times))])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Heatmap
    ax = axes[0]
    im = ax.imshow(density.T, aspect='auto', origin='lower',
                   extent=[times[0], times[-1], -0.5, L - 0.5],
                   cmap='YlGnBu', norm=mcolors.LogNorm(vmin=1e-6, vmax=1))

    plt.colorbar(im, ax=ax, label='Site density')
    ax.set_xlabel('Circuit Layer ($\ell$)', fontsize=25)
    ax.set_ylabel('Site', fontsize=25)
    ax.tick_params(labelsize=15)
    # ax.set_title('Operator Spreading (Site Density)', fontsize=14)

    # Mark the center
    center = L // 2
    ax.axhline(y=center, color='white', linestyle='--', linewidth=0.5, alpha=0.5)

    # Plot 2: Selected time slices
    ax = axes[1]

    # Select a few time points
    time_indices = [0, len(times) // 4, len(times) // 2, 3 * len(times) // 4, len(times) - 1]
    time_indices = [t for t in time_indices if t < len(times)]

    sites = np.arange(L)
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

    for i, t_idx in enumerate(time_indices):
        ax.plot(sites, density[t_idx], 'o-', color=colors[i],
                label=f't={times[t_idx]}', markersize=4)

    ax.set_xlabel('Site', fontsize=12)
    ax.set_ylabel('Site density', fontsize=12)
    ax.set_title('Site Density at Different Times', fontsize=14)
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/site_density.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/site_density.png")


def plot_arrival_times(input_dir: str, output_dir: str, L: int):
    """Plot arrival times vs distance from center."""

    df = pd.read_csv(f"{input_dir}/aggregate/aggregate_arrival_times.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Arrival time vs distance
    ax = axes[0]

    # Filter valid arrivals
    valid = df[df['arrival_time_mean'] >= 0]

    ax.errorbar(valid['distance_from_center'], valid['arrival_time_mean'],
                yerr=valid['arrival_time_std'], fmt='o', capsize=3,
                markersize=8, label='Data')

    # Linear fit
    from scipy import stats
    mask = valid['arrival_time_mean'] > 0
    if mask.sum() >= 2:
        x = valid.loc[mask, 'arrival_time_mean'].values
        y = valid.loc[mask, 'distance_from_center'].values
        slope, intercept, r, _, _ = stats.linregress(x, y)

        t_fit = np.linspace(0, valid['arrival_time_mean'].max(), 100)
        d_fit = slope * t_fit + intercept
        ax.plot(d_fit, t_fit, 'r--', label=f'Fit: v={slope:.3f} sites/step')

    ax.set_xlabel('Distance from center', fontsize=12)
    ax.set_ylabel('Arrival time', fontsize=12)
    ax.set_title('Light Cone: Arrival Time vs Distance', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Arrival time map (spatial)
    ax = axes[1]

    sites = df['site'].values
    times = df['arrival_time_mean'].values
    times[times < 0] = np.nan  # Mark never-arrived as NaN

    colors = plt.cm.viridis(times / np.nanmax(times))
    colors[np.isnan(times)] = [0.5, 0.5, 0.5, 1]  # Gray for never arrived

    ax.bar(sites, np.ones(L), color=colors, edgecolor='black', linewidth=0.5)

    for i, t in enumerate(times):
        if not np.isnan(t):
            ax.text(i, 0.5, f'{int(t)}', ha='center', va='center', fontsize=10)

    ax.set_xlabel('Site', fontsize=12)
    ax.set_ylabel('')
    ax.set_title('Arrival Time Map', fontsize=14)
    ax.set_yticks([])

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(0, np.nanmax(times)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Arrival time')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/arrival_times.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/arrival_times.png")


def plot_weight_distribution(input_dir: str, output_dir: str, L: int):
    """Plot weight distribution evolution."""

    # Load from first individual run (or aggregate if available)
    individual_dir = f"{input_dir}/individual"
    run_files = sorted(Path(individual_dir).glob("run_*_weight_dist.csv"))

    if not run_files:
        print("No weight distribution files found")
        return

    df = pd.read_csv(run_files[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Stacked area plot
    ax = axes[0]

    times = df['time_step'].values
    weights = np.array([[df[f'weight_{w}'].iloc[t] for w in range(L + 1)]
                        for t in range(len(times))])

    colors = plt.cm.viridis(np.linspace(0, 1, L + 1))

    ax.stackplot(times, weights.T, labels=[f'w={w}' for w in range(L + 1)],
                 colors=colors, alpha=0.8)

    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Weight Distribution Over Time', fontsize=14)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, 1)

    # Legend outside
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

    # Plot 2: Weight distribution at different times
    ax = axes[1]

    time_indices = [0, len(times) // 3, 2 * len(times) // 3, len(times) - 1]
    time_indices = [t for t in time_indices if t < len(times)]

    width = 0.8 / len(time_indices)
    w_values = np.arange(L + 1)

    for i, t_idx in enumerate(time_indices):
        offset = (i - len(time_indices) / 2 + 0.5) * width
        ax.bar(w_values + offset, weights[t_idx], width,
               label=f't={times[t_idx]}', alpha=0.8)

    ax.set_xlabel('Hamming weight', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Weight Distribution Snapshots', fontsize=14)
    ax.legend()
    ax.set_xticks(w_values)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/weight_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/weight_distribution.png")


def plot_pauli_string_evolution(input_dir: str, output_dir: str, L: int):
    """Plot when different Pauli weights appear over time."""
    import sys
    sys.path.insert(0, '.')

    # Load weight distribution from first run
    df = pd.read_csv(f"{input_dir}/individual/run_000_weight_dist.csv")

    times = df['time_step'].values
    weights_matrix = np.array([[df[f'weight_{w}'].iloc[t] for w in range(L + 1)]
                               for t in range(len(times))])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Heatmap: time vs weight
    im = ax.imshow(weights_matrix.T, aspect='auto', origin='lower',
                   extent=[times[0], times[-1], -0.5, L + 0.5],
                   cmap='hot', norm=plt.Normalize(0, 1))

    plt.colorbar(im, ax=ax, label='Probability')
    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Hamming weight', fontsize=12)
    ax.set_title('Pauli Weight Evolution', fontsize=14)
    ax.set_yticks(range(L + 1))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/pauli_weight_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/pauli_weight_heatmap.png")

def plot_num_strings(input_dir: str, output_dir: str):
    """Plot number of Pauli strings vs time."""

    agg_df = pd.read_csv(f"{input_dir}/aggregate/aggregate_hamming_weight.csv")

    fig, ax = plt.subplots(figsize=(8, 5))

    t = agg_df['time_step']
    mean = agg_df['num_strings_mean']
    std = agg_df['num_strings_std']

    ax.plot(t, mean, 'b-', linewidth=2, label='Mean')
    ax.fill_between(t, mean - std, mean + std, alpha=0.3, color='blue')

    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Number of Pauli strings', fontsize=12)
    ax.set_title('Operator Complexity Growth', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Check if it looks exponential
    if len(t) > 3:
        ax2 = ax.twinx()
        ax2.semilogy(t, mean, 'r--', alpha=0.5)
        ax2.set_ylabel('Log scale', fontsize=10, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/num_strings.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/num_strings.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize operator spreading results')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory with simulation output')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for plots (default: input_dir/plots)')
    parser.add_argument('--L', type=int, default=6,
                        help='System size')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"{args.input_dir}/plots"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Creating visualizations in {args.output_dir}/\n")

    # Generate all plots
    plot_hamming_weight(args.input_dir, args.output_dir)
    plot_site_density_heatmap(args.input_dir, args.output_dir, args.L)
    plot_arrival_times(args.input_dir, args.output_dir, args.L)
    plot_weight_distribution(args.input_dir, args.output_dir, args.L)
    plot_num_strings(args.input_dir, args.output_dir)
    plot_pauli_string_evolution(args.input_dir, args.output_dir, args.L)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()