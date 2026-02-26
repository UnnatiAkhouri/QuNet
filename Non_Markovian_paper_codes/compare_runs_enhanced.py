import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#set your plotting configs here. This includes the font type, color, line styles, labels for rules etc.

plt.style.use('default')
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
})

# Rule mapping configuration
RULE_MAPPING = {
    'results_rand': '$R_0$',
    'rule_rand': '$R_0$',
    'results_0300': '$R_1$',
    'results_0300': '$R_1$',
    'results_3032': '$R_2$',
    'results_3032': '$R_2$',
    'results_2000': '$R_3$',
    'rule_2000': '$R_3$',
    'results_1302': '$R_4$',
    'rule_1302': '$R_4$',
    'results_3311': '$R_5$',
    'rule_3311': '$R_5$',
}

# Pastel color scheme for rules (distinct but soft colors)
RULE_COLORS = {
    '$R_1$': '#d11141',  # Pastel red
    '$R_2$': '#00b159',  # Pastel green
    '$R_3$': '#00aedb',  # Pastel blue
    '$R_4$': '#f37735',  # Pastel yellow
    '$R_5$': '#ffc425',  # Pastel purple
}

# Line styles for rules
RULE_LINESTYLES = {
    '$R_1$': '-',  # solid
    '$R_2$': '--',  # dashed
    '$R_3$': '-.',  # dash-dot
    '$R_4$': ':',  # dotted
    '$R_5$': (0, (3, 1, 1, 1)),  # densely dashdotted
}

# Random walk styling (bold black)
RANDOM_STYLE = {
    'color': 'black',
    'linewidth': 3,
    'linestyle': '-',
    'label_suffix': ' (Random)'
}


def get_run_style(run_name: str) -> Dict:
    """Get the style configuration for a run based on its name."""
    # Check if this is a random walk run
    if 'rand' in run_name.lower():
        return {
            'color': RANDOM_STYLE['color'],
            'linewidth': RANDOM_STYLE['linewidth'],
            'linestyle': RANDOM_STYLE['linestyle'],
            'label': '$R_0$',
            'is_random': True
        }

    # Check if this matches a known rule
    for key, rule_label in RULE_MAPPING.items():
        if key in run_name.lower():
            return {
                'color': RULE_COLORS.get(rule_label, '#808080'),
                'linewidth': 2,
                'linestyle': RULE_LINESTYLES.get(rule_label, '-'),
                'label': rule_label,
                'is_random': False,
                'rule_full_name': key
            }

    # Default styling for unknown runs
    return {
        'color': '#808080',
        'linewidth': 2,
        'linestyle': '-',
        'label': run_name,
        'is_random': False
    }


def create_rule_mapping_text() -> str:
    """Create text showing the rule mapping for the legend."""
    unique_rules = {}
    for key, rule_label in RULE_MAPPING.items():
        if rule_label not in unique_rules:
            # Extract just the rule number
            rule_num = key.split('_')[-1] if '_' in key else key
            unique_rules[rule_label] = rule_num

    lines = ['Rule Mapping:']
    for rule_label in sorted(unique_rules.keys()):
        lines.append(f'{rule_label} = Rule {unique_rules[rule_label]}')

    return '\n'.join(lines)


def load_run_data(run_dir: str) -> dict:
    """Load aggregate data from a single run directory."""
    data = {'name': Path(run_dir).name, 'dir': run_dir}

    # Load aggregate hamming weight
    hw_file = f"{run_dir}/aggregate/aggregate_hamming_weight.csv"
    if Path(hw_file).exists():
        df = pd.read_csv(hw_file)
        data['time'] = df['time_step'].values
        data['avg_weight_mean'] = df['avg_weight_mean'].values
        data['avg_weight_std'] = df['avg_weight_std'].values
        data['num_strings_mean'] = df['num_strings_mean'].values
        data['num_strings_std'] = df['num_strings_std'].values

    # Load aggregate arrival times
    arr_file = f"{run_dir}/aggregate/aggregate_arrival_times.csv"
    if Path(arr_file).exists():
        df = pd.read_csv(arr_file)
        data['arrival_times'] = df['arrival_time_mean'].values
        data['arrival_time_std'] = df['arrival_time_std'].values
        data['arrival_distances'] = df['distance_from_center'].values



    # Load analysis results
    analysis_file = f"{run_dir}/analysis_results.csv"
    if Path(analysis_file).exists():
        df = pd.read_csv(analysis_file)
        if len(df) > 0:
            data['velocity'] = df['velocity_velocity'].iloc[0]
            data['diffusion_const'] = df['walk_diffusion_const'].iloc[0]

    return data


def compare_hamming_weight(runs: List[dict], output_file: str):
    """Plot average Hamming weight comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: Mean with error bands
    ax = axes[0]
    for run in runs:
        if 'time' not in run:
            continue

        style = get_run_style(run['name'])
        t = run['time']
        mean = run['avg_weight_mean']
        std = run['avg_weight_std']

        ax.plot(t, mean, color=style['color'], linewidth=style['linewidth'],
                linestyle=style['linestyle'], label=style['label'])

        # Add error bands with matching color
        if not style['is_random']:  # Lighter bands for non-random
            ax.fill_between(t, mean - std, mean + std,
                            color=style['color'], alpha=0.2)
        else:  # Even lighter for random
            ax.fill_between(t, mean - std, mean + std,
                            color=style['color'], alpha=0.1)

    ax.set_xlabel('$\ell$', fontsize=30)
    plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern (LaTeX-like)

    ax.set_ylabel(r'$\langle w_{\rm ave}(\ell)\rangle_{\rm ens}$', fontsize=30)
    #ax.set_ylabel(r'$w_{\rm ave}(\ell)$', fontsize=25)
    ax.tick_params(labelsize=25)
    #ax.set_title('Hamming Weight Growth Comparison', fontsize=15)
    ax.legend(fontsize=25, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Add rule mapping text box below the plot
    #mapping_text = create_rule_mapping_text()
    #ax.text(0.02, -0.25, mapping_text, transform=ax.transAxes,
    #        fontsize=9, verticalalignment='top',
    #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
    #        family='monospace')

    # Plot 2: Growth rate comparison
    ax = axes[1]

    growth_rates = []
    names = []
    colors = []
    for run in runs:
        if 'time' not in run or len(run['time']) < 2:
            continue

        style = get_run_style(run['name'])
        t = run['time']
        w = run['avg_weight_mean']

        # Linear fit
        from scipy import stats
        slope, _, _, _, _ = stats.linregress(t, w)
        growth_rates.append(slope)
        names.append(style['label'])
        colors.append(style['color'])

    if growth_rates:
        x = np.arange(len(names))
        bars = ax.bar(x, growth_rates, color=colors, edgecolor='black', linewidth=1)

        # Bold edge for random walk
        for i, name in enumerate(names):
            if 'Random' in name:
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(3)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Growth rate (weight/step)', fontsize=12)
        #ax.set_title('Weight Growth Rate Comparison', fontsize=15)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def compare_velocity(runs: List[dict], output_file: str):
    """Plot butterfly velocity comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: Arrival time vs distance for all runs (mean of left and right)
    ax = axes[0]
    for run in runs:
        if 'arrival_times' not in run:
            continue

        style = get_run_style(run['name'])
        times = run['arrival_times']
        dists = run['arrival_distances']
        times_std = run['arrival_time_std']

        # Find center position (where distance == 0)
        center_idx = np.where(dists == 0)[0]
        if len(center_idx) == 0:
            continue
        center_idx = center_idx[0]

        # Split into left (before center) and right (after center)
        times_left = times[:center_idx]
        dists_left = dists[:center_idx]

        times_right = times[center_idx + 1:]
        dists_right = dists[center_idx + 1:]

        # Get absolute distances (assuming symmetric)
        abs_dists_left = np.abs(dists_left)
        abs_dists_right = np.abs(dists_right)

        # Calculate mean arrival times for matching distances
        # Reverse left arrays so distances align (left goes from far to center)
        times_left_rev = times_left[::-1]
        abs_dists_left_rev = abs_dists_left[::-1]

        # Find common length (in case of asymmetry)
        min_len = min(len(times_left_rev), len(times_right))

        if min_len > 0:
            times_left_matched = times_left_rev[:min_len]
            times_right_matched = times_right[:min_len]
            dists_matched = abs_dists_right[:min_len]

            # Calculate mean of left and right arrival times
            times_mean = (times_left_matched + times_right_matched) / 2.0

            # Handle std for marker sizes
            if times_std is not None:
                std_left = times_std[:center_idx][::-1][:min_len]
                std_right = times_std[center_idx + 1:][:min_len]
                std_mean = np.sqrt(std_left ** 2 + std_right ** 2) / 2.0
                marker_sizes = 20 + std_mean * 100
            else:
                marker_sizes = np.full(len(times_mean), 50)

            # Handle unpaired points (if left side is longer)
            if len(times_left_rev) > len(times_right):
                extra_times = times_left_rev[min_len:]
                extra_dists = abs_dists_left_rev[min_len:]
                times_mean = np.concatenate([times_mean, extra_times])
                dists_matched = np.concatenate([dists_matched, extra_dists])
                if times_std is not None:
                    extra_std = times_std[:center_idx][::-1][min_len:]
                    extra_sizes = 20 + extra_std * 100
                    marker_sizes = np.concatenate([marker_sizes, extra_sizes])
                else:
                    marker_sizes = np.concatenate([marker_sizes, np.full(len(extra_times), 50)])

            # Handle unpaired points (if right side is longer)
            elif len(times_right) > len(times_left_rev):
                extra_times = times_right[min_len:]
                extra_dists = abs_dists_right[min_len:]
                times_mean = np.concatenate([times_mean, extra_times])
                dists_matched = np.concatenate([dists_matched, extra_dists])
                if times_std is not None:
                    extra_std = times_std[center_idx + 1:][min_len:]
                    extra_sizes = 20 + extra_std * 100
                    marker_sizes = np.concatenate([marker_sizes, extra_sizes])
                else:
                    marker_sizes = np.concatenate([marker_sizes, np.full(len(extra_times), 50)])

            # Filter valid times (>= 0)
            valid = times_mean >= 0
            if np.any(valid):
                ax.scatter(times_mean[valid], dists_matched[valid],
                           s=marker_sizes[valid] if isinstance(marker_sizes, np.ndarray) else marker_sizes,
                           color=style['color'], alpha=0.8,
                           edgecolors='black', linewidth=0.5,
                           label=style['label'])
                # Connect with line
                ax.plot(times_mean[valid], dists_matched[valid], '-',
                        color=style['color'], linewidth=style['linewidth'] * 0.5,
                        linestyle=style['linestyle'], alpha=0.5)

    ax.set_xlabel(r'$\langle t_{\rm far-side}\rangle_{\rm ens}$', fontsize=30)
    ax.set_ylabel(r'$z$', fontsize=30)
    ax.tick_params(labelsize=25)
    #ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Plot 2: Velocity bar chart
    ax = axes[1]

    velocities = []
    names = []
    colors = []
    for run in runs:
        if 'velocity' in run and not np.isnan(run['velocity']):
            style = get_run_style(run['name'])
            velocities.append(run['velocity'])
            names.append(style['label'])
            colors.append(style['color'])

    if velocities:
        x = np.arange(len(names))
        bars = ax.bar(x, velocities, color=colors, edgecolor='black', linewidth=1)

        # Bold edge for random walk
        for i, name in enumerate(names):
            if 'Random' in name:
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(3)

        ax.set_xticks(x)
        ax.tick_params(labelsize=15)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Butterfly velocity (sites/step)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")





def compare_complexity(runs: List[dict], output_file: str,
                       t_min: int = 2, t_max: int = 100,
                       inset_t_min: int = 75, inset_t_max: int = 110):
    """Plot number of Pauli strings comparison with time range and inset."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for run in runs:
        if 'time' not in run:
            continue

        style = get_run_style(run['name'])
        t = run['time']
        num_strings = run['num_strings_mean']
        num_strings_std = run.get('num_strings_std', None)

        # Filter to specified time range
        mask = (t >= t_min) & (t <= t_max)
        t_filtered = t[mask]
        num_strings_filtered = num_strings[mask]

        if len(t_filtered) > 0:
            # Add shaded region for standard deviation
            if num_strings_std is not None:
                std_filtered = num_strings_std[mask]
                ax.fill_between(t_filtered,
                                num_strings_filtered - std_filtered,
                                num_strings_filtered + std_filtered,
                                color=style['color'], alpha=0.2)

            ax.plot(t_filtered, num_strings_filtered, 'o-',
                    color=style['color'], linewidth=style['linewidth'],
                    linestyle=style['linestyle'], markersize=.1,
                    label=style['label'])

    ax.set_xlabel(f'$\ell$ ', fontsize=30)
    ax.set_ylabel(r'$\langle C(\ell) \rangle_{\rm ens}$', fontsize=30)
    ax.tick_params(labelsize=25)
    ax.grid(True, alpha=0.3)

    # Add inset for zoomed view
    axins = inset_axes(ax, width="40%", height="40%", loc='lower right',
                       borderpad=5)

    for run in runs:
        if 'time' not in run:
            continue

        style = get_run_style(run['name'])
        t = run['time']
        num_strings = run['num_strings_mean']
        num_strings_std = run.get( None)

        # Filter to inset time range
        mask = (t >= inset_t_min) & (t <= inset_t_max)
        t_inset = t[mask]
        num_strings_inset = num_strings[mask]

        if len(t_inset) > 0:
            # Add shaded region for standard deviation in inset
            if num_strings_std is not None:
                std_inset = num_strings_std[mask]
                axins.fill_between(t_inset,
                                   num_strings_inset - std_inset,
                                   num_strings_inset + std_inset,
                                   color=style['color'], alpha=0.2)

            axins.plot(t_inset, num_strings_inset, 'o-',
                       color=style['color'], linewidth=style['linewidth'],
                       linestyle=style['linestyle'], markersize=.1)

    axins.grid(True, alpha=0.3)
    axins.tick_params(labelsize=18)

    # Add rectangle on main plot to show inset region
    if inset_t_min >= t_min and inset_t_max <= t_max:
        from matplotlib.patches import Rectangle
        from matplotlib.patches import ConnectionPatch

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def create_summary_table(runs: List[dict], output_file: str):
    """Create a summary table comparing all runs."""
    from scipy import stats

    summary = []

    for run in runs:
        style = get_run_style(run['name'])
        row = {'name': style['label']}

        # Final weight
        if 'avg_weight_mean' in run:
            row['final_weight'] = run['avg_weight_mean'][-1]

        # Growth rate
        if 'time' in run and len(run['time']) > 1:
            slope, _, _, _, _ = stats.linregress(run['time'], run['avg_weight_mean'])
            row['growth_rate'] = slope

        # Velocity
        if 'velocity' in run:
            row['velocity'] = run['velocity']

        # Diffusion constant
        if 'diffusion_const' in run:
            row['diffusion_const'] = run['diffusion_const']

        # Final complexity
        if 'num_strings_mean' in run:
            row['final_strings'] = run['num_strings_mean'][-1]

        summary.append(row)

    df = pd.DataFrame(summary)
    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

    # Also print to console
    print("\nSummary Table:")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Compare multiple simulation runs')
    parser.add_argument('--run_dirs', type=str, nargs='+',
                        help='Directories containing simulation results')
    parser.add_argument('--output_dir', type=str, default='comparison',
                        help='Output directory for comparison plots')
    parser.add_argument('--pattern', type=str, default=None,
                        help='Glob pattern to find run directories (e.g., "results_rule*")')
    parser.add_argument('--complexity_t_min', type=int, default=2,
                        help='Minimum time step for complexity plot (default: 2)')
    parser.add_argument('--complexity_t_max', type=int, default=100,
                        help='Maximum time step for complexity plot (default: 100)')
    parser.add_argument('--inset_t_min', type=int, default=75,
                        help='Minimum time step for inset zoom (default: 75)')
    parser.add_argument('--inset_t_max', type=int, default=110,
                        help='Maximum time step for inset zoom (default: 110)')

    args = parser.parse_args()

    # Find run directories
    if args.pattern:
        import glob
        run_dirs = sorted(glob.glob(args.pattern))
    elif args.run_dirs:
        run_dirs = args.run_dirs
    else:
        print("Error: Must provide either --pattern or --run_dirs")
        return

    if not run_dirs:
        print("No run directories found!")
        return

    print(f"Comparing {len(run_dirs)} runs:")
    for d in run_dirs:
        print(f"  - {d}")

    # Load all data
    runs = []
    for run_dir in run_dirs:
        try:
            data = load_run_data(run_dir)
            runs.append(data)
            style = get_run_style(data['name'])
            print(f"  Loaded: {data['name']} -> {style['label']}")
        except Exception as e:
            print(f"  Error loading {run_dir}: {e}")

    if not runs:
        print("No valid runs loaded!")
        return

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating comparison plots in {args.output_dir}/")

    # Generate all comparison plots
    compare_hamming_weight(runs, f"{args.output_dir}/compare_hamming_weight.png")
    compare_velocity(runs, f"{args.output_dir}/compare_velocity.png")
    compare_complexity(runs, f"{args.output_dir}/compare_complexity.png",
                       t_min=args.complexity_t_min,
                       t_max=args.complexity_t_max,
                       inset_t_min=args.inset_t_min,
                       inset_t_max=args.inset_t_max)
    create_summary_table(runs, f"{args.output_dir}/summary_table.csv")

    print(f"\nAll comparisons saved to {args.output_dir}/")
    print(f"\nComplexity plot settings:")
    print(f"  Main plot: t={args.complexity_t_min} to {args.complexity_t_max}")
    print(f"  Inset zoom: t={args.inset_t_min} to {args.inset_t_max}")


if __name__ == '__main__':
    main()