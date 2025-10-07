import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def monte_carlo_hypothesis_test(baseline_values, target_values, baseline_stds, target_stds, n_samples=5, n_simulations=10000):
    """
    Perform Monte Carlo hypothesis test comparing two groups of means.
    
    H0: mean(baseline) <= mean(target)
    H1: mean(baseline) > mean(target)
    
    Parameters:
    - baseline_values: list of observed means for baseline methods
    - target_values: list of observed means for target methods (LC, BALD, EKM, PW)
    - baseline_stds: list of standard deviations for baseline methods
    - target_stds: list of standard deviations for target methods
    - n_samples: number of repeats for each observed mean (default: 5)
    - n_simulations: number of Monte Carlo simulations
    
    Returns:
    - p_value: probability of observing the difference under H0
    - test_statistic: observed difference in means
    - simulated_diffs: array of simulated differences
    """
    
    # Calculate observed test statistic (difference in means)
    observed_baseline_mean = np.mean(baseline_values)
    observed_target_mean = np.mean(target_values)
    observed_diff = observed_baseline_mean - observed_target_mean
    
    # Monte Carlo simulation
    simulated_diffs = []
    
    for _ in range(n_simulations):
        # Generate samples for baseline methods
        baseline_samples = []
        for mean_val, std_val in zip(baseline_values, baseline_stds):
            if pd.isna(mean_val) or pd.isna(std_val):
                continue
            samples = np.random.normal(mean_val, std_val, n_samples)
            baseline_samples.extend(samples)
        
        # Generate samples for target methods
        target_samples = []
        for mean_val, std_val in zip(target_values, target_stds):
            if pd.isna(mean_val) or pd.isna(std_val):
                continue
            samples = np.random.normal(mean_val, std_val, n_samples)
            target_samples.extend(samples)
        
        # Calculate difference in sample means
        if len(baseline_samples) > 0 and len(target_samples) > 0:
            sim_baseline_mean = np.mean(baseline_samples)
            sim_target_mean = np.mean(target_samples)
            sim_diff = sim_baseline_mean - sim_target_mean
            simulated_diffs.append(sim_diff)
    
    simulated_diffs = np.array(simulated_diffs)
    
    # Calculate p-value (one-tailed test)
    # H0: baseline_mean <= target_mean, so we test if observed_diff > 0
    # p-value = P(simulated_diff >= observed_diff | H0)
    p_value = np.mean(simulated_diffs >= observed_diff)
    
    return p_value, observed_diff, simulated_diffs

# Load the data
df = pd.read_csv('results.csv')
df_std = pd.read_csv('std.csv')

# Verify that both DataFrames have the same structure
if not df.columns.equals(df_std.columns):
    print("Warning: Column names in results.csv and std.csv don't match!")
    print(f"Results columns: {df.columns.tolist()}")
    print(f"Std columns: {df_std.columns.tolist()}")

if not df['model'].equals(df_std['model']):
    print("Warning: Model names in results.csv and std.csv don't match!")
    print("Attempting to align by model name...")
    df_std = df_std.set_index('model').reindex(df['model']).reset_index()

# Define the target methods
target_methods = ['LC', 'BALD', 'EKM', 'PW']

# Get all other methods (baseline)
baseline_methods = [method for method in df['model'] if method not in target_methods]

print("Hypothesis Test Results")
print("=" * 50)
print("H0: mean(baseline methods) <= mean(target methods)")
print("H1: mean(baseline methods) > mean(target methods)")
print(f"Target methods: {target_methods}")
print(f"Baseline methods: {baseline_methods}")
print("=" * 50)

# Perform hypothesis test for each column
results = {}
columns_to_test = ['QNLI', 'QQP', 'RTE', 'SST2', 'WNLI', 'MNLI', 'MRPC', 'COLA']

for col in columns_to_test:
    print(f"\n{col}:")
    print("-" * 20)
    
    # Get baseline values and their standard deviations
    baseline_mask = df['model'].isin(baseline_methods)
    baseline_values = df[baseline_mask][col].values
    baseline_stds = df_std[baseline_mask][col].values
    
    # Get target values and their standard deviations
    target_mask = df['model'].isin(target_methods)
    target_values = df[target_mask][col].values
    target_stds = df_std[target_mask][col].values
    
    # Convert string 'n.a.' to NaN and ensure numeric
    baseline_values = pd.to_numeric(baseline_values, errors='coerce')
    baseline_stds = pd.to_numeric(baseline_stds, errors='coerce')
    target_values = pd.to_numeric(target_values, errors='coerce')
    target_stds = pd.to_numeric(target_stds, errors='coerce')
    
    # Remove rows where either mean or std is NaN
    baseline_valid = ~(pd.isna(baseline_values) | pd.isna(baseline_stds))
    target_valid = ~(pd.isna(target_values) | pd.isna(target_stds))
    
    baseline_values = baseline_values[baseline_valid]
    baseline_stds = baseline_stds[baseline_valid]
    target_values = target_values[target_valid]
    target_stds = target_stds[target_valid]
    
    if len(baseline_values) == 0 or len(target_values) == 0:
        print(f"Skipping {col} due to insufficient valid data")
        continue
    
    # Perform Monte Carlo test
    p_value, observed_diff, simulated_diffs = monte_carlo_hypothesis_test(
        baseline_values, target_values, baseline_stds, target_stds
    )
    
    # Store results
    results[col] = {
        'p_value': p_value,
        'observed_diff': observed_diff,
        'baseline_mean': np.mean(baseline_values),
        'target_mean': np.mean(target_values),
        'baseline_std': np.mean(baseline_stds),
        'target_std': np.mean(target_stds),
        'baseline_count': len(baseline_values),
        'target_count': len(target_values)
    }
    
    # Print results
    print(f"Baseline mean: {np.mean(baseline_values):.3f} ± {np.mean(baseline_stds):.3f} (n={len(baseline_values)})")
    print(f"Target mean: {np.mean(target_values):.3f} ± {np.mean(target_stds):.3f} (n={len(target_values)})")
    print(f"Observed difference: {observed_diff:.3f}")
    print(f"P-value: {p_value:.4f}")
    
    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: REJECT H0 (p < {alpha})")
        print("Conclusion: Baseline methods perform significantly better than target methods")
    else:
        print(f"Result: FAIL TO REJECT H0 (p >= {alpha})")
        print("Conclusion: No significant evidence that baseline methods perform better")

# Summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Column':<8} {'Baseline':<12} {'Target':<12} {'Diff':<8} {'P-value':<10} {'Significant':<12}")
print("-" * 85)

for col, result in results.items():
    significant = "YES" if result['p_value'] < 0.05 else "NO"
    baseline_display = f"{result['baseline_mean']:.2f}±{result['baseline_std']:.2f}"
    target_display = f"{result['target_mean']:.2f}±{result['target_std']:.2f}"
    print(f"{col:<8} {baseline_display:<12} {target_display:<12} "
          f"{result['observed_diff']:<8.3f} {result['p_value']:<10.4f} {significant:<12}")

print("\n" + "=" * 85)
print("INTERPRETATION:")
print("- Uses actual standard deviations from std.csv for Monte Carlo simulation")
print("- Positive difference: Baseline methods perform better")
print("- Negative difference: Target methods perform better")
print("- Significant (p < 0.05): Strong evidence against H0")
print("=" * 85)

# Optional: Create comprehensive visualization
fig, axes = plt.subplots(2, len(results), figsize=(4*len(results), 10))
if len(results) == 1:
    axes = axes.reshape(2, 1)

col_idx = 0
for col, result in results.items():
    # Get the data for this column
    baseline_mask = df['model'].isin(baseline_methods)
    target_mask = df['model'].isin(target_methods)
    
    baseline_values = pd.to_numeric(df[baseline_mask][col], errors='coerce')
    target_values = pd.to_numeric(df[target_mask][col], errors='coerce')
    baseline_stds = pd.to_numeric(df_std[baseline_mask][col], errors='coerce')
    target_stds = pd.to_numeric(df_std[target_mask][col], errors='coerce')
    
    # Remove NaN values
    baseline_valid = ~(pd.isna(baseline_values) | pd.isna(baseline_stds))
    target_valid = ~(pd.isna(target_values) | pd.isna(target_stds))
    
    baseline_values = baseline_values[baseline_valid]
    target_values = target_values[target_valid]
    baseline_stds = baseline_stds[baseline_valid]
    target_stds = target_stds[target_valid]
    
    # Create H0 distribution (assuming baseline methods <= target methods)
    # Under H0, we simulate what the difference distribution would look like
    h0_simulations = []
    for _ in range(10000):
        # Sample from baseline
        baseline_samples = []
        for mean_val, std_val in zip(baseline_values, baseline_stds):
            samples = np.random.normal(mean_val, std_val, 5)
            baseline_samples.extend(samples)
        
        # Sample from target  
        target_samples = []
        for mean_val, std_val in zip(target_values, target_stds):
            samples = np.random.normal(mean_val, std_val, 5)
            target_samples.extend(samples)
        
        if len(baseline_samples) > 0 and len(target_samples) > 0:
            h0_simulations.append(np.mean(baseline_samples) - np.mean(target_samples))
    
    h0_simulations = np.array(h0_simulations)
    h0_mean = np.mean(h0_simulations)
    h0_std = np.std(h0_simulations)
    
    # Plot 1: Histogram of H0 distribution with observed difference
    axes[0, col_idx].hist(h0_simulations, bins=50, alpha=0.7, density=True, 
                         color='lightblue', edgecolor='black', label='H0 distribution')
    
    # Overlay theoretical normal distribution
    x_range = np.linspace(h0_simulations.min(), h0_simulations.max(), 100)
    h0_pdf = stats.norm.pdf(x_range, h0_mean, h0_std)
    axes[0, col_idx].plot(x_range, h0_pdf, 'b-', linewidth=2, 
                         label=f'H0: N({h0_mean:.2f}, {h0_std:.2f})')
    
    # Mark observed difference
    observed_diff = result['observed_diff']
    axes[0, col_idx].axvline(observed_diff, color='red', linestyle='-', linewidth=3,
                            label=f'Observed: {observed_diff:.3f}')
    
    # Mark individual baseline and target means
    for val in baseline_values:
        axes[0, col_idx].axvline(val, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    for val in target_values:
        axes[0, col_idx].axvline(val, color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add p-value and significance info
    p_val = result['p_value']
    significant = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    
    axes[0, col_idx].set_title(f'{col}\nBaseline: {result["baseline_mean"]:.2f} vs Target: {result["target_mean"]:.2f}\n'
                              f'Diff: {observed_diff:.3f}, p = {p_val:.4f} {significant}')
    axes[0, col_idx].set_xlabel('Mean Difference (Baseline - Target)')
    axes[0, col_idx].set_ylabel('Density')
    axes[0, col_idx].legend(fontsize=8)
    axes[0, col_idx].grid(True, alpha=0.3)
    
    # Add shaded region for p-value
    if observed_diff > h0_mean:
        x_shade = x_range[x_range >= observed_diff]
        y_shade = stats.norm.pdf(x_shade, h0_mean, h0_std)
        axes[0, col_idx].fill_between(x_shade, y_shade, alpha=0.3, color='red', 
                                     label=f'p-value region')
    
    # Plot 2: Individual data points with error bars
    ax2 = axes[1, col_idx]
    
    # Plot baseline methods
    baseline_x = np.arange(len(baseline_values))
    ax2.errorbar(baseline_x, baseline_values, yerr=baseline_stds, 
                fmt='o', color='orange', capsize=5, capthick=2, 
                label='Baseline methods', markersize=8)
    
    # Plot target methods  
    target_x = np.arange(len(baseline_values), len(baseline_values) + len(target_values))
    ax2.errorbar(target_x, target_values, yerr=target_stds,
                fmt='s', color='green', capsize=5, capthick=2,
                label='Target methods', markersize=8)
    
    # Add horizontal lines for group means
    ax2.axhline(result['baseline_mean'], color='orange', linestyle='-', alpha=0.7,
               xmin=0, xmax=len(baseline_values)/(len(baseline_values)+len(target_values)),
               linewidth=3, label=f'Baseline mean: {result["baseline_mean"]:.2f}')
    ax2.axhline(result['target_mean'], color='green', linestyle='-', alpha=0.7,
               xmin=len(baseline_values)/(len(baseline_values)+len(target_values)), xmax=1,
               linewidth=3, label=f'Target mean: {result["target_mean"]:.2f}')
    
    # Customize plot
    all_methods = [method for method in df['model'] if method in baseline_methods][:len(baseline_values)] + \
                  [method for method in df['model'] if method in target_methods][:len(target_values)]
    
    ax2.set_xticks(range(len(all_methods)))
    ax2.set_xticklabels(all_methods, rotation=45, ha='right')
    ax2.set_ylabel(f'{col} Performance')
    ax2.set_title(f'{col} - Individual Method Performance')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Add separator line between groups
    if len(baseline_values) > 0 and len(target_values) > 0:
        separator_x = len(baseline_values) - 0.5
        ax2.axvline(separator_x, color='black', linestyle=':', alpha=0.5)
    
    col_idx += 1

plt.tight_layout()
plt.savefig('hypothesis_test_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# Create summary comparison plot
plt.figure(figsize=(12, 8))
columns = list(results.keys())
baseline_means = [results[col]['baseline_mean'] for col in columns]
target_means = [results[col]['target_mean'] for col in columns]
baseline_stds = [results[col]['baseline_std'] for col in columns]
target_stds = [results[col]['target_std'] for col in columns]
p_values = [results[col]['p_value'] for col in columns]

x = np.arange(len(columns))
width = 0.35

# Create bar plot with error bars
bars1 = plt.bar(x - width/2, baseline_means, width, yerr=baseline_stds, 
                label='Baseline Methods', alpha=0.7, capsize=5, color='orange')
bars2 = plt.bar(x + width/2, target_means, width, yerr=target_stds,
                label='Target Methods (LC, BALD, EKM, PW)', alpha=0.7, capsize=5, color='green')

# Add significance markers
for i, (col, p_val) in enumerate(zip(columns, p_values)):
    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    if significance:
        max_height = max(baseline_means[i] + baseline_stds[i], target_means[i] + target_stds[i])
        plt.text(i, max_height + 1, significance, ha='center', va='bottom', 
                fontsize=16, fontweight='bold', color='red')

plt.xlabel('Metrics')
plt.ylabel('Mean Performance')
plt.title('Performance Comparison: Baseline vs Target Methods\n'
          '(Error bars show standard deviations, *** p<0.001, ** p<0.01, * p<0.05)')
plt.xticks(x, columns, rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('performance_comparison_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional test: Consistent superiority across all columns
print("\n" + "=" * 100)
print("CONSISTENT SUPERIORITY TEST")
print("=" * 100)
print("Testing if any method consistently outperforms Baseline across ALL columns")
print("For each method, we test: P(method > Baseline for ALL columns simultaneously)")

def consistent_superiority_test(df_means, df_stds, n_samples=5, n_iterations=10000):
    """
    Test if any method consistently outperforms baseline across all columns.
    
    Returns:
    - superiority_probs: dict of P(all greater) for each method
    - counter_results: dict of raw counters for each method
    """
    
    # Get all methods except Baseline
    all_methods = df_means['model'].tolist()
    test_methods = [method for method in all_methods if method != 'Baseline']
    
    # Get baseline row
    baseline_idx = df_means[df_means['model'] == 'Baseline'].index[0]
    
    # Get columns to test (exclude 'model' column)
    test_columns = [col for col in df_means.columns if col != 'model']
    
    # Initialize counters
    counters = {method: 0 for method in test_methods}
    
    print(f"Testing {len(test_methods)} methods across {len(test_columns)} columns")
    print(f"Running {n_iterations} Monte Carlo iterations...")
    
    for iteration in range(n_iterations):
        if iteration % 2000 == 0:
            print(f"  Iteration {iteration}/{n_iterations}")
        
        # Resample all values in the dataframe
        resampled_data = {}
        
        for _, row in df_means.iterrows():
            method = row['model']
            resampled_data[method] = {}
            
            for col in test_columns:
                mean_val = pd.to_numeric(row[col], errors='coerce')
                std_val = pd.to_numeric(df_stds[df_stds['model'] == method][col].iloc[0], errors='coerce')
                
                if pd.isna(mean_val) or pd.isna(std_val):
                    resampled_data[method][col] = np.nan
                else:
                    # Sample n_samples and take the mean
                    samples = np.random.normal(mean_val, std_val, n_samples)
                    resampled_data[method][col] = np.mean(samples)
        
        # Check each test method against baseline
        for method in test_methods:
            all_greater = True
            
            for col in test_columns:
                method_val = resampled_data[method].get(col, np.nan)
                baseline_val = resampled_data['Baseline'].get(col, np.nan)
                
                # Skip if either value is NaN
                if pd.isna(method_val) or pd.isna(baseline_val):
                    continue
                
                if method_val <= baseline_val:
                    all_greater = False
                    break
            
            if all_greater:
                counters[method] += 1
    
    # Calculate probabilities
    superiority_probs = {method: count / n_iterations for method, count in counters.items()}
    
    return superiority_probs, counters

# Run the test
superiority_probs, counter_results = consistent_superiority_test(df, df_std)

print("\nRESULTS - P(method > Baseline for ALL columns):")
print("-" * 60)
for method, prob in superiority_probs.items():
    print(f"{method:<15}: {prob:.6f} ({counter_results[method]:,}/{10000:,} iterations)")

# Find methods with P(all greater) > 0.95
high_performing_methods = [method for method, prob in superiority_probs.items() if prob > 0.95]

print(f"\nMethods with P(all greater) > 0.95: {high_performing_methods}")

# Calculate probability of seeing at least one method with P(all greater) > 0.95
def calculate_multiple_testing_probability(superiority_probs, threshold=0.95, n_columns=8, n_samples=5):
    """
    Calculate the probability of seeing at least one method exceed the threshold
    under the null hypothesis of no systematic superiority.
    
    This is complex because we need to account for:
    1. Multiple testing across methods
    2. Dependence across columns
    3. The specific threshold and sample sizes
    """
    
    n_methods = len(superiority_probs)
    
    # Under null hypothesis, each method has some base probability of beating baseline
    # This depends on the number of columns and sample variance
    # For simplicity, we'll estimate this empirically or use a conservative approach
    
    # Conservative approach: assume independence and use Bonferroni-like correction
    # Under null, P(single comparison > baseline) ≈ 0.5 for each column
    # P(all columns > baseline) ≈ 0.5^n_columns under independence
    null_prob_single_method = 0.5 ** n_columns
    
    # Probability that at least one method exceeds threshold under null
    # P(at least one > threshold) = 1 - P(all <= threshold)
    # Under null, each method has probability null_prob_single_method of exceeding any threshold
    prob_none_exceed = (1 - null_prob_single_method) ** n_methods
    prob_at_least_one_exceeds = 1 - prob_none_exceed
    
    print(f"\nMULTIPLE TESTING ANALYSIS:")
    print("-" * 40)
    print(f"Number of methods tested: {n_methods}")
    print(f"Number of columns: {n_columns}")
    print(f"Under null hypothesis:")
    print(f"  P(single method > baseline for all columns) ≈ {null_prob_single_method:.6f}")
    print(f"  P(at least one method appears superior) ≈ {prob_at_least_one_exceeds:.6f}")
    
    # More sophisticated analysis using the actual observed probabilities
    observed_high_count = len(high_performing_methods)
    if observed_high_count > 0:
        print(f"\nOBSERVED RESULTS:")
        print(f"  Number of methods with P(all greater) > {threshold}: {observed_high_count}")
        print(f"  This suggests potential systematic superiority beyond chance")
    else:
        print(f"\nOBSERVED RESULTS:")
        print(f"  No methods exceed P(all greater) > {threshold}")
        print(f"  Results consistent with null hypothesis of no systematic superiority")
    
    return prob_at_least_one_exceeds, null_prob_single_method

multiple_testing_prob, null_single_prob = calculate_multiple_testing_probability(
    superiority_probs, threshold=0.95, n_columns=len(columns_to_test), n_samples=5
)

# Create visualization for consistent superiority results
plt.figure(figsize=(12, 8))

methods = list(superiority_probs.keys())
probs = list(superiority_probs.values())
colors = ['red' if prob > 0.95 else 'lightblue' for prob in probs]

bars = plt.bar(methods, probs, color=colors, alpha=0.7, edgecolor='black')

# Add threshold line
plt.axhline(y=0.95, color='red', linestyle='--', linewidth=2, 
           label='P(all greater) = 0.95 threshold')

# Add null expectation line
plt.axhline(y=null_single_prob, color='gray', linestyle=':', linewidth=2,
           label=f'Null expectation ≈ {null_single_prob:.6f}')

# Annotate bars with exact values
for bar, prob in zip(bars, probs):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{prob:.4f}', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Methods')
plt.ylabel('P(method > Baseline for ALL columns)')
plt.title('Consistent Superiority Test Results\n' + 
          f'P(method outperforms Baseline across all {len(columns_to_test)} columns simultaneously)')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, max(max(probs) * 1.1, 1.0))

# Add text box with summary
textstr = f'Methods with P > 0.95: {len(high_performing_methods)}\n'
textstr += f'Expected under null: {multiple_testing_prob:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('consistent_superiority_test.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 100)
print("INTERPRETATION:")
print("- Values close to 1.0 indicate consistent superiority across all metrics")
print("- Values close to 0.0 indicate the method rarely beats baseline across all metrics")
print("- Threshold of 0.95 represents strong evidence of systematic superiority")
print("- Multiple testing correction suggests caution in interpreting high values")
print("=" * 100)