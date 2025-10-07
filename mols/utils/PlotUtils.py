#!/usr/bin/env python3
"""Plotting utilities for active learning results."""

import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Any
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent  # Adjust based on your structure
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgb
# Local imports
from utils.DataUtils import get_training_data, get_all_training_repeats
font = {'size' : 16}
mpl.rc('font', **font)
mpl.rc('lines', linewidth=1.5)
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2


def register_tintmap(name: str, hex_color: str):
    c = to_rgb('#'+hex_color) if not hex_color.startswith('#') else to_rgb(hex_color)
    cmap = LinearSegmentedColormap.from_list(name, [(1,1,1), c, (0.05,0.05,0.05)], N=256)
    mpl.colormaps.register(cmap, name=name)
    return name
try:
    OI_BLUE    = register_tintmap("OI_BLUE",    "#0072B2")
    OI_ORANGE  = register_tintmap("OI_ORANGE",  "#E69F00")
    OI_MAGENTA = register_tintmap("OI_MAGENTA", "#CC79A7")
    OI_Vermillion = register_tintmap("OI_Vermillion", "#D55E00")
    OI_Bluish_Green = register_tintmap("OI_Bluish_Green", "#009E73")
except:
    pass
def get_selector_color_map() -> Dict[str, str]:
    """Create a deterministic color mapping for selector names.
    
    Returns:
        Dictionary mapping selector names to colors
    """

    additional_colors = [
        '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
        '#8b4513', '#17becf', '#a6cee3', '#fb9a99', '#fdbf6f', 
        '#cab2d6', '#800080', '#b15928', '#fbb4ae', '#b3cde3', 
        '#ccebc5', '#decbe4'
    ]
    
    # Start with required mappings
    color_map = {
        'ExploitSelector': '#2ca02c',     # green
        'ThompsonSelector': '#ff7f0e', # orange
        'RandomSelector': '#1f77b4'       # blue
    }
    
    # Add other common selectors
    other_selectors = [
        'LLMFeaturiseQwenSelector', 'LLMAgentNoSMILES', 'LLMFeaturiseLlamaSelector',
        'LLMFatSelector', 'LLMFeaturiseGPTSelector', 'LLMAgentSelector',
    ]
    
    for i, selector in enumerate(other_selectors):
        if i < len(additional_colors):
            color_map[selector] = additional_colors[i]
    
    return color_map

def extract_benchmark_info(filename: str) -> Tuple[str, str, str, str, str]:
    """Extract benchmark information from filename.
    
    Args:
        filename: Benchmark filename (e.g., 'benchmark_D2R_GPRegOracle_RandomSelector_60_60.csv')
    
    Returns:
        Tuple of (target, oracle, selector, initial_size, batch_size)
    """
    # Remove .csv extension and split by underscore
    parts = filename.replace('.csv', '').split('_')
    
    if len(parts) < 6 or parts[0] != 'benchmark':
        raise ValueError(f"Invalid benchmark filename format: {filename}")
    
    # Extract components
    target = parts[1]
    oracle = parts[2]
    selector = parts[3]
    initial_size = parts[4]
    batch_size = parts[5]
     
    return target, oracle, selector, initial_size, batch_size

def get_top_20_affinities(results_file: str) -> Dict[int, List[float]]:
    """Extract top 20 highest affinity compounds for each cycle from training results.
    
    Args:
        results_file: Path to training results JSON file
        
    Returns:
        Dictionary mapping total_labeled to list of top 20 affinities
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Get protein name from data_path
    data_path_str = results["experiment_config"]["data_path"]
    protein = Path(data_path_str).stem  # Extract filename without extension
    data_path = project_root / "data" / f"{protein}.csv"
    
    # Load all data
    data = pd.read_csv(data_path)
    smiles_to_affinity = dict(zip(data['SMILES'], data['affinity']))
    
    # Extract top 20 affinities for each cycle
    top_20_affinities = {}
    
    for cycle in results["al_cycles"]:
        samples_selected = cycle["samples_selected"]
        
        # Get affinities for selected samples
        affinities = []
        for smiles in samples_selected:
            affinities.append(smiles_to_affinity[smiles])
        
        # Get top 20 highest affinities
        if len(affinities) >= 20:
            top_20 = sorted(affinities, reverse=True)[:20]
        else:
            # If less than 20 samples, use all available
            top_20 = sorted(affinities, reverse=True)
        
        top_20_affinities[cycle['cycle']] = top_20
    
    return top_20_affinities

def plot_UMAP(results_file: str, output_dir: str) -> None:
    """Plot UMAP visualization of AL exploration in chemical space. 
    Shows the dataset as shaded density and selected samples 
    colored by AL cycle using a custom OI color scheme.
    
    Args:
        results_file: Path to training results JSON file
        output_dir: Directory to save the plot
    """
    from scipy.stats import gaussian_kde
    from umap import UMAP
    from utils.SMILESUtils import smiles_to_fingerprint

    # Reproducibility
    np.random.seed(42)
    random.seed(42)

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Load dataset
    data_path = project_root / results["experiment_config"]["data_path"]
    data = pd.read_csv(data_path)
    all_smiles = data.iloc[:, 0]
    
    # Extract selected samples for each cycle
    selected_samples, cycle_numbers = [], []
    for cycle in results["al_cycles"]:
        selected_samples.extend(cycle["samples_selected"])
        cycle_numbers.extend([cycle["cycle"]] * len(cycle["samples_selected"]))
    selected_samples = results["al_cycles"][0]["samples_selected"]
    # Fingerprints
    all_fps = np.array([smiles_to_fingerprint(s, radius=4, nBits=4096) for s in all_smiles])
    sel_fps = np.array([smiles_to_fingerprint(s, radius=4, nBits=4096) for s in selected_samples])
    
    # UMAP embedding
    reducer = UMAP(n_components=2, random_state=42, transform_seed=42, n_jobs=1)
    all_emb = reducer.fit_transform(all_fps)
    sel_emb = reducer.transform(sel_fps)
    
    # Prepare density background
    x, y = all_emb[:, 0], all_emb[:, 1]
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    truncated_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "trunc_greys", plt.cm.Greys(np.linspace(0.5, 1, 256))
    )
    # Background density
    ax.scatter(x, y, c=z, s=10, alpha=0.4, cmap=truncated_cmap, rasterized=True)
    
    # Selected samples DataFrame
    plot_df = pd.DataFrame({
        'x': sel_emb[:, 0],
        'y': sel_emb[:, 1],
    })
    
    # Cycle gradient scatter
    scatter = ax.scatter(
        plot_df['x'], plot_df['y'],
        c='#CC79A7',
        alpha=0.9,
        s=40,
        edgecolor='k',
        linewidth=0
    )
    
    # Labels and title
    ax.set_title(
        f"UMAP of BO Initialization\n"
        f"{results['experiment_config']['data_path'].split('/')[-1][:-4]} {'Bad' if results['experiment_config']['bad_start'] else 'Random'} Start",
        fontsize=18
    )
    ax.set_xlabel("UMAP Dimension 1", fontsize=16)
    ax.set_ylabel("UMAP Dimension 2", fontsize=16)
    
    # Save
    output_path = Path(output_dir) / "umap_exploration.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"UMAP visualization saved to {output_path}")


def plot_training_results(results_file: str, output_dir: str = None) -> None:
    """Plot active learning training results.
    
    Args:
        results_file: Path to the training results JSON file
        output_dir: Directory to save the plots
    """

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get metrics DataFrame
    metrics_df = get_training_data(results_file)

    # Plot performance metrics with top affinity boxplot
    title = f'{results["experiment_config"]["model"]} {results["experiment_config"]["selector"]} {results["experiment_config"]["batch_size"]}'
    plot_performance_metrics_with_top_affinity(
        metrics_dfs=[metrics_df],
        title=title,
        output_path=output_path / "training_metrics.png",
        show_individual_runs=False,
        color_map=None,
        results_files=[results_file]
    )
    
    # Generate UMAP visualization
    #plot_UMAP(results_file, output_dir)

def plot_benchmark(
    target: str,
    output_dir: str = project_root / "output" / "plots" / "benchmarks",
    oracle_name: str = '*',
    selector_name: str = '*',
    initial_size: int = None,
    batch_size: int = None,
    bad_start: bool = False,
    selector_kwargs: Optional[Dict[str, Any]] = None,
    max_rand_idx = None
) -> None:
    """Benchmark across combinations of oracle_name, selector_name, initial_size, batch_size for a given target.
    
    Args:
        target: Name of the target
        output_dir: Directory to save benchmark plots
        oracle_name: Name of the oracle (None for all)
        selector_name: Name of the selector (* for all)
        initial_size: Initial size (* for all)
        batch_size: Batch size (* for all)
        bad_start: Whether to use bad start initialization
        selector_kwargs: Selector-specific parameters for filtering
    """
    def interpolate_clean_data(metrics_df: pd.DataFrame, initial_size_value: int, batch_size_value: int, total_budget: int) -> pd.DataFrame:
        """Interpolates missing data points where (total_labeled-initial_size) % batch_size == 0.
        """
        def linear_interpolate(x, x_l, x_h, y_l, y_h):
            res = y_l + (y_h - y_l) * (x - x_l) / (x_h - x_l)
            return res
        
        # Create a new dataframe with the same columns
        for nlabeled in range(initial_size_value, total_budget+1, batch_size_value):
            if nlabeled not in metrics_df["total_labeled"].values:
                # Linear interpolation between the two closest points
                lower_point = metrics_df[metrics_df["total_labeled"] < nlabeled].iloc[-1]
                upper_point = metrics_df[metrics_df["total_labeled"] > nlabeled].iloc[0]
                nl, nh = lower_point["total_labeled"], upper_point["total_labeled"]
                new_row = pd.Series({
                    "total_labeled": nlabeled,
                    "recall_2pc": linear_interpolate(nlabeled, nl, nh, lower_point["recall_2pc"], upper_point["recall_2pc"]),
                    "recall_5pc": linear_interpolate(nlabeled, nl, nh, lower_point["recall_5pc"], upper_point["recall_5pc"]),
                    "rmse": linear_interpolate(nlabeled, nl, nh, lower_point["rmse"], upper_point["rmse"]),
                    "spearman_rho": linear_interpolate(nlabeled, nl, nh, lower_point["spearman_rho"], upper_point["spearman_rho"]),
                    "overall_max": linear_interpolate(nlabeled, nl, nh, lower_point["overall_max"], upper_point["overall_max"])
                })
                # Insert new row and ensure data is sorted by total_labeled from smallest to largest
                metrics_df = pd.concat([metrics_df, new_row.to_frame().T], ignore_index=True)

        metrics_df = metrics_df[(metrics_df["total_labeled"] - initial_size_value) % batch_size_value == 0]     
        metrics_df = metrics_df.sort_values(by="total_labeled", ascending=True)
        return metrics_df
        
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    benchmark_dir = project_root / "output" / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    initial_size = str(initial_size) if initial_size is not None else '*'
    batch_size = str(batch_size) if batch_size is not None else '*'
    bad_start_str = 'Bad' if bad_start else 'Random'
    # Get all training repeats. The utility function uses glob, so * can be used directly.
    training_repeats = get_all_training_repeats(
        oracle_name=oracle_name,
        selector_name=selector_name,
        initial_size=initial_size,
        batch_size=batch_size,
        target=target,
        bad_start=bad_start,
        selector_kwargs=selector_kwargs
    )
    print(f'Found {len(training_repeats)} training repeats')
    if not training_repeats:
        print(f"No training repeats found for target {target}")
        return
    
    # Load metrics for each repeat
    metrics_dfs = []
    results_files = []
    for repeat in training_repeats:
        results_file = repeat / "Training_results.json"
        if not results_file.exists():
            print(f"Warning: No results file found in {repeat}")
            continue
        
        file = json.load(open(results_file))
        if max_rand_idx is not None and file["experiment_config"]['random_seed'] >= max_rand_idx:
            continue
        initial_size_value = file["experiment_config"]["initial_size"]
        batch_size_value = file["experiment_config"]["batch_size"]
        total_budget_value = file["experiment_config"]["total_budget"]
        #If not in the file, set to False
        bad_start_value = file["experiment_config"].get("bad_start", False)

        if bad_start_value == bad_start:
            training_data = get_training_data(str(results_file))
            # Allow 5 % error to be rounnded up
            if training_data.iloc[-1]['total_labeled'] < total_budget_value * 0.95:
                print(f"Total labeled is less than {total_budget_value * 0.95}")
                continue
            if training_data.iloc[-1]['total_labeled'] > total_budget_value * 0.95 and training_data.iloc[-1]['total_labeled'] < total_budget_value:
                training_data.loc[training_data.index[-1], 'total_labeled'] = total_budget_value
                print(f"Setting total labeled to {total_budget_value}")
            metrics_df = interpolate_clean_data(training_data, int(initial_size_value), int(batch_size_value), int(total_budget_value))
            metrics_dfs.append(metrics_df)
            results_files.append(str(results_file))

    print(f"Loaded {len(metrics_dfs)} training repeats")
    
    if not metrics_dfs:
        print("No valid training data found")
        return
    
    # Create title with wildcards
    parts = [oracle_name, selector_name, initial_size, batch_size, bad_start_str]
    if max_rand_idx is not None:
        parts.append(f"maxseed_{max_rand_idx}")
    title = f'Oracle: {oracle_name}, Selector: {selector_name}, Initial Size: {initial_size}, Batch Size: {batch_size}, {bad_start_str}'
    if selector_kwargs and 'beta' in selector_kwargs:
        title += f", Selector Kwargs: {selector_kwargs['beta']}"
        parts.append(f"beta_{selector_kwargs['beta']}")
    elif selector_kwargs:
        title += f", Selector Kwargs: {selector_kwargs}"
        parts.append(f"{'_'.join([k for k in selector_kwargs.keys() if selector_kwargs[k] == True])}")
    
    # Plot performance metrics with top affinity boxplot
    aggregated_data = plot_performance_metrics_with_top_affinity(
        metrics_dfs=metrics_dfs,
        title=title,
        output_path=output_path / f"benchmark_{target}_{'_'.join(parts)}.png",
        show_individual_runs=True,
        color_map=None,
        results_files=results_files
    )
    
    # Save aggregated data
    benchmark_file = benchmark_dir / f"benchmark_{target}_{'_'.join(parts)}.csv"
    aggregated_data.to_csv(benchmark_file, index=False)
    print(f"Saved benchmark data to {benchmark_file}")

def plot_performance_metrics(
    metrics_dfs: List[pd.DataFrame],
    title: str,
    output_path: str,
    show_individual_runs: bool = False,
    color_map: Optional[Dict[str, str]] = None,
    line_labels: Optional[List[str]] = None,
    std: bool = True
) -> pd.DataFrame:
    """Plot performance metrics for one or more training runs.
    
    Args:
        metrics_dfs: List of DataFrames containing metrics
        title: Title for the plot
        output_path: Where to save the plot
        show_individual_runs: Whether to show individual runs as faint lines
        color_map: Dictionary mapping labels to colors (for multiple datasets)
        line_labels: Labels for each dataset in metrics_dfs
        std: Whether to display standard deviation as shadows around the lines (default: True)
    
    Returns:
        DataFrame containing aggregated data
    """
    # Set up the plot style
    plt.style.use('ggplot')
    
    # Create subplots with shared y-axis only between ax1 and ax2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    ax2.sharey(ax1)
    fig.suptitle(title, fontsize=14)
    
    # Define metrics and their corresponding axes
    metrics_config = [
        ('recall_2pc', ax1, 'Recall@2%', 'darkblue'),
        ('recall_5pc', ax2, 'Recall@5%', 'red'),
        ('rmse', ax3, 'RMSE', 'green'),
        ('overall_max', ax4, "Overall Max", 'orange')
    ]
    
    # If we have multiple datasets with labels, use color_map
    if color_map and line_labels and len(metrics_dfs) == len(line_labels):
        # Track which labels we've already seen to avoid duplicate legend entries
        seen_labels = set()
        
        # Plot each dataset with its assigned color
        for i, (df, label) in enumerate(zip(metrics_dfs, line_labels)):
            color = color_map.get(label, f'C{i}')
            alpha = 0.3 if show_individual_runs else 0.8
            
            for metric, ax, title_text, _ in metrics_config:
                # Only add label to legend if we haven't seen this label before
                legend_label = label if label not in seen_labels else None
                
                # Plot the mean line
                ax.plot(df["total_labeled"], df[metric], 
                       color=color, alpha=alpha, linewidth=2, label=legend_label)
                
                # Plot confidence interval if std column exists and std is True
                std_col = f"{metric}_std"
                if std and std_col in df.columns:
                    ax.fill_between(df["total_labeled"],
                                  df[metric] - df[std_col],
                                  df[metric] + df[std_col],
                                  alpha=0.2, color=color)
            
            # Mark this label as seen after plotting all metrics for this dataset
            seen_labels.add(label)
    else:
        # Original behavior for single dataset or multiple runs
        # Plot individual runs if requested
        for df in metrics_dfs:
            if show_individual_runs:
                for metric, ax, title_text, default_color in metrics_config:
                    ax.plot(df["total_labeled"], df[metric], 
                           alpha=0.2, color=default_color)
        
        # Calculate mean and std for each metric
        all_data = pd.concat(metrics_dfs)
        mean_data = all_data.groupby('total_labeled').mean()
        std_data = all_data.groupby('total_labeled').std()
        
        # Plot mean and std for each metric
        for metric, ax, title_text, default_color in metrics_config:
            ax.plot(mean_data.index, mean_data[metric], 
                   label='Mean', color=default_color, linewidth=2)
            if std:
                ax.fill_between(mean_data.index,
                              mean_data[metric] - std_data[metric],
                              mean_data[metric] + std_data[metric],
                              alpha=0.2, color=default_color)
    
    # Configure each subplot
    for metric, ax, title_text, default_color in metrics_config:
        ax.set_title(title_text)
        ax.set_xlabel('Total Labeled')
        ax.set_ylabel(title_text)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary if we have aggregated data
    if not color_map or not line_labels:
        all_data = pd.concat(metrics_dfs)
        mean_data = all_data.groupby('total_labeled').mean()
        std_data = all_data.groupby('total_labeled').std()
        #Add ending _std to the column names
        std_data.columns = [col + '_std' for col in std_data.columns]
        #Merge mean_data and std_data
        mean_data = pd.merge(mean_data, std_data, on='total_labeled')
        
        print("\nTraining Results Summary:")
        print(f"Final Recall@2%: {mean_data['recall_2pc'].iloc[-1]:.3f} ± {mean_data['recall_2pc_std'].iloc[-1]:.3f}")
        print(f"Final Recall@5%: {mean_data['recall_5pc'].iloc[-1]:.3f} ± {mean_data['recall_5pc_std'].iloc[-1]:.3f}")
        print(f"Final RMSE: {mean_data['rmse'].iloc[-1]:.3f} ± {mean_data['rmse_std'].iloc[-1]:.3f}")
        print(f"Final Spearman's rho: {mean_data['spearman_rho'].iloc[-1]:.3f} ± {mean_data['spearman_rho_std'].iloc[-1]:.3f}")
        print(f"Final Overall Max: {mean_data['overall_max'].iloc[-1]:.3f} ± {mean_data['overall_max_std'].iloc[-1]:.3f}")
        # Return aggregated data
        return mean_data.reset_index()
    else:
        # Return empty DataFrame for multiple datasets case
        return pd.DataFrame()

def plot_performance_metrics_with_top_affinity(
    metrics_dfs: List[pd.DataFrame],
    title: str,
    output_path: str,
    show_individual_runs: bool = False,
    color_map: Optional[Dict[str, str]] = None,
    line_labels: Optional[List[str]] = None,
    results_files: Optional[List[str]] = None,
    std: bool = True
) -> pd.DataFrame:
    """Plot performance metrics for one or more training runs with top 20 affinity boxplot.
    
    Args:
        metrics_dfs: List of DataFrames containing metrics
        title: Title for the plot
        output_path: Where to save the plot
        show_individual_runs: Whether to show individual runs as faint lines
        color_map: Dictionary mapping labels to colors (for multiple datasets)
        line_labels: Labels for each dataset in metrics_dfs
        results_files: List of paths to training results JSON files (required for top affinity plot)
        std: Whether to display standard deviation as shadows around the lines (default: True)
    
    Returns:
        DataFrame containing aggregated data
    """
    plt.style.use('ggplot')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    ax2.sharey(ax1)
    fig.suptitle(title, fontsize=14)
    metrics_config = [
        ('recall_2pc', ax1, 'Recall@2%', 'darkblue'),
        ('recall_5pc', ax2, 'Recall@5%', 'red'),
        ('rmse', ax3, 'RMSE', 'green'),
        ('overall_max', ax4, 'Overall Max', 'purple')
    ]
    xticks = None
    xticklabels = None
    initial_size = None
    batch_size = None
    n_cycles = None
    if results_files and len(results_files) > 0:
        with open(results_files[0], 'r') as f:
            results = json.load(f)
        initial_size = results['experiment_config'].get('initial_size', 0)
        batch_size = results['experiment_config'].get('batch_size', 1)
        n_cycles = len(results['al_cycles'])
        xticks = [initial_size + batch_size * i for i in range(n_cycles)]
        xticklabels = [str(x) for x in xticks]
    if color_map and line_labels and len(metrics_dfs) == len(line_labels):
        seen_labels = set()
        n_datasets = len(line_labels)
        # Each dataset is offset by 5 units, centered around the canonical x-tick
        for i, (df, label) in enumerate(zip(metrics_dfs, line_labels)):
            color = color_map.get(label, f'C{i}')
            alpha = 0.3 if show_individual_runs else 0.8
            for metric, ax, title_text, _ in metrics_config:
                legend_label = label if label not in seen_labels else None
                ax.plot(df["total_labeled"], df[metric], color=color, alpha=alpha, linewidth=2, label=legend_label)
                std_col = f"{metric}_std"
                if std and std_col in df.columns:
                    ax.fill_between(df["total_labeled"], df[metric] - df[std_col], df[metric] + df[std_col], alpha=0.2, color=color)
            seen_labels.add(label)
    else:
        for df in metrics_dfs:
            if show_individual_runs:
                for metric, ax, title_text, default_color in metrics_config:
                    if metric != 'top_affinity':
                        ax.plot(df["total_labeled"], df[metric], alpha=0.2, color=default_color)
        all_data = pd.concat(metrics_dfs)
        mean_data = all_data.groupby('total_labeled').mean()
        std_data = all_data.groupby('total_labeled').std()
        for metric, ax, title_text, default_color in metrics_config:
            if metric != 'top_affinity':
                ax.plot(mean_data.index, mean_data[metric], label='Mean', color=default_color, linewidth=2)
                if std:
                    ax.fill_between(mean_data.index, mean_data[metric] - std_data[metric], mean_data[metric] + std_data[metric], alpha=0.2, color=default_color)
        if results_files and xticks is not None:
            all_top_20_data = {}
            for results_file in results_files:
                top_20_data = get_top_20_affinities(results_file)
                for x in xticks:
                    if x not in all_top_20_data:
                        all_top_20_data[x] = []
                    all_top_20_data[x].extend(top_20_data.get(x, []))
            boxplot_data = []
            boxplot_positions = []
            for x in xticks:
                vals = all_top_20_data.get(x, [])
                if len(vals) > 0:
                    boxplot_data.append(vals)
                    boxplot_positions.append(x)
            if boxplot_data:
                ax4.boxplot(
                    boxplot_data,
                    positions=boxplot_positions,
                    widths=2.4,  # 2x larger boxes
                    patch_artist=True,
                    boxprops=dict(facecolor='purple', alpha=0.7),
                    medianprops=dict(color='black', linewidth=3.0),  # 2x thicker median line
                    showfliers=False  # Don't show outliers
                )
                # Add single circle for maximum value
                for j, (data, pos) in enumerate(zip(boxplot_data, boxplot_positions)):
                    if len(data) > 0:
                        max_val = max(data)
                        ax4.scatter(pos, max_val, color='purple', s=36, zorder=5)  # s=36 for markersize=6 equivalent
    if xticks is not None:
        ax4.set_xticks(xticks)
        ax4.set_xticklabels(xticklabels)
    for metric, ax, title_text, default_color in metrics_config:
        ax.set_title(title_text)
        ax.set_xlabel('Total Labeled')
        ax.set_ylabel(title_text)
        ax.grid(True, alpha=0.3)
        if metric != 'top_affinity':
            ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    if not color_map or not line_labels:
        all_data = pd.concat(metrics_dfs)
        mean_data = all_data.groupby('total_labeled').mean()
        std_data = all_data.groupby('total_labeled').std()
        std_data.columns = [col + '_std' for col in std_data.columns]
        mean_data = pd.merge(mean_data, std_data, on='total_labeled')
        print("\nTraining Results Summary:")
        print(f"Final Recall@2%: {mean_data['recall_2pc'].iloc[-1]:.3f} ± {mean_data['recall_2pc_std'].iloc[-1]:.3f}")
        print(f"Final Recall@5%: {mean_data['recall_5pc'].iloc[-1]:.3f} ± {mean_data['recall_5pc_std'].iloc[-1]:.3f}")
        print(f"Final RMSE: {mean_data['rmse'].iloc[-1]:.3f} ± {mean_data['rmse_std'].iloc[-1]:.3f}")
        print(f"Final Overall Max: {mean_data['overall_max'].iloc[-1]:.3f} ± {mean_data['overall_max_std'].iloc[-1]:.3f}")
        return mean_data.reset_index()
    else:
        return pd.DataFrame()

def plot_concat_benchmark(pattern: str, name: str, output_dir: str = project_root / "output" / "plots" / "benchmarks", std: bool = True) -> None:
    """Plot all benchmarks in a single plot. Same format as plot_benchmark. Does plot all individual lines, only mean and std.
    
    The glob pattern should be a pattern that matches all benchmark files.
    Args:
        pattern: Regex pattern to match benchmark CSV files
        name: Name for the combined plot
        output_dir: Directory to save the plot
        std: Whether to display standard deviation as shadows around the lines (default: True)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all benchmark files
    benchmark_files = [p for p in Path('output/benchmarks/').iterdir() 
         if re.match(pattern, p.name)]
    print(f"Found {len(benchmark_files)} benchmark files")
    
    if not benchmark_files:
        print(f"No benchmark files found matching pattern: {pattern}")
        return
    
    # Load metrics for each benchmark file and extract selector names
    metrics_dfs = []
    line_labels = []
    color_map = get_selector_color_map()
    
    for benchmark_file in benchmark_files:
        try:
            # Extract selector name from filename
            filename = Path(benchmark_file).name
            target, oracle, selector, initial_size, batch_size = extract_benchmark_info(filename)
            
            # Load the benchmark data
            metrics_df = pd.read_csv(benchmark_file)
            metrics_dfs.append(metrics_df)
            line_labels.append(selector)
            
            print(f"Loaded {selector} from {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to load {benchmark_file}: {str(e)}")
            continue
    
    if not metrics_dfs:
        print("No valid benchmark data found")
        return
    
    # Create title
    title = f'Benchmark Comparison: {name}'
    
    # Plot performance metrics
    plot_performance_metrics(
        metrics_dfs=metrics_dfs,
        title=title,
        output_path=output_path / f"concat_benchmark_{name}.png",
        show_individual_runs=False,
        color_map=color_map,
        line_labels=line_labels,
        std=std
    )
    
    print(f"Combined benchmark plot saved to {output_path / f'concat_benchmark_{name}.png'}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot active learning training results")
    parser.add_argument("results_file", type=str, help="Path to training results JSON file")
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    args = parser.parse_args()
    
    plot_training_results(args.results_file, args.output_dir)
