import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('results.csv')

# Define the reference models for H0 parameters
reference_models = ['LC', 'BALD', 'EKM', 'PW']

# Get all models except baseline for testing
test_models = df[df['model'] != 'Baseline']['model'].tolist()

# Columns to test (excluding 'model' column)
test_columns = [col for col in df.columns if col != 'model']

# Store results
results = {}

print("Hypothesis Test Results")
print("="*60)
print("H0: All results (except baseline) are normally distributed around")
print("    the mean and std of LC, BALD, EKM, and PW")
print("H1: The distribution is not normal with these parameters")
print("Additional one-sided test: H0 vs H1 (values > H0 mean)")
print("Note: Each value is an average of n=5 repeats with 10% std dev")
print("="*60)

# Parameters for measurement uncertainty
n_repeats = 5  # Each value is average of 5 repeats
measurement_cv = 0.10  # 10% coefficient of variation (std/mean)

for column in test_columns:
    print(f"\nTesting column: {column}")
    print("-" * 40)
    
    # Handle missing values (like 'n.a.' in QQP for BALD)
    column_data = df[column].copy()
    
    # Convert to numeric, handling 'n.a.' as NaN
    column_data = pd.to_numeric(column_data, errors='coerce')
    
    # Get reference data (LC, BALD, EKM, PW) for parameter estimation
    reference_mask = df['model'].isin(reference_models)
    reference_data = column_data[reference_mask].dropna()
    
    if len(reference_data) < 2:
        print(f"  Insufficient reference data for {column}")
        continue
    
    # Estimate H0 parameters from reference models
    h0_mean = reference_data.mean()
    h0_std = reference_data.std(ddof=1)  # Sample standard deviation
    
    # Calculate measurement uncertainty for each data point
    # Each observed value x_obs ~ N(x_true, (x_true * cv)^2 / n_repeats)
    # Standard error of the mean for each measurement
    def measurement_se(value):
        return (value * measurement_cv) / np.sqrt(n_repeats)
    
    # Adjust H0 parameters for measurement uncertainty
    # The reference data also has measurement error, so we need to account for this
    reference_measurement_vars = [(val * measurement_cv)**2 / n_repeats for val in reference_data]
    h0_measurement_se = np.sqrt(np.mean(reference_measurement_vars))
    
    print(f"  H0 parameters: μ = {h0_mean:.3f}, σ = {h0_std:.3f}")
    print(f"  Measurement SE per point: ~{h0_measurement_se:.3f} (varies by value)")

    
    # Get test data (all non-baseline models)
    test_mask = df['model'] != 'Baseline'
    test_data = column_data[test_mask].dropna()
    
    if len(test_data) < 3:
        print(f"  Insufficient test data for {column}")
        continue
    
    # Perform Kolmogorov-Smirnov test
    # Test if test_data follows normal distribution N(h0_mean, h0_std)
    ks_statistic, ks_p_value = stats.kstest(
        test_data, 
        lambda x: stats.norm.cdf(x, loc=h0_mean, scale=h0_std)
    )
    
    # Perform Shapiro-Wilk test on standardized data
    # Standardize the test data using H0 parameters
    standardized_data = (test_data - h0_mean) / h0_std
    if len(standardized_data) >= 3:
        sw_statistic, sw_p_value = stats.shapiro(standardized_data)
    else:
        sw_statistic, sw_p_value = np.nan, np.nan
    
    # Anderson-Darling test
    # Transform data to standard normal under H0
    ad_statistic, ad_critical_values, ad_significance_levels = stats.anderson(
        standardized_data, dist='norm'
    )
    
    # One-sided test: Are values significantly GREATER than H0 mean?
    # H0: μ_test = μ_h0, H1: μ_test > μ_h0
    test_mean = test_data.mean()
    test_std = test_data.std(ddof=1)
    n_test = len(test_data)
    
    # Account for measurement uncertainty in the test
    # Each test point has measurement variance = (value * cv)^2 / n_repeats
    test_measurement_vars = [(val * measurement_cv)**2 / n_repeats for val in test_data]
    avg_measurement_var = np.mean(test_measurement_vars)
    
    # Adjusted standard error for the test mean
    # SE = sqrt(sample_var/n + avg_measurement_var)
    sample_se = test_std / np.sqrt(n_test)
    measurement_se_mean = np.sqrt(avg_measurement_var)
    total_se = np.sqrt(sample_se**2 + measurement_se_mean**2)
    
    # One-sample t-test with measurement uncertainty
    t_statistic_adj = (test_mean - h0_mean) / total_se
    # Use Welch-Satterthwaite equation for degrees of freedom approximation
    # This is conservative given the measurement uncertainty
    df_adj = n_test - 1  # Conservative approximation
    t_p_value_one_sided_adj = 1 - stats.t.cdf(t_statistic_adj, df=df_adj)
    
    # Standard t-test (without measurement uncertainty correction)
    t_statistic = (test_mean - h0_mean) / (test_std / np.sqrt(n_test))
    t_p_value_one_sided = 1 - stats.t.cdf(t_statistic, df=n_test-1)
    
    # Z-test approximation (for larger samples) - also adjust for measurement uncertainty
    z_statistic_adj = (test_mean - h0_mean) / np.sqrt((h0_std**2 + avg_measurement_var) / n_test)
    z_p_value_one_sided_adj = 1 - stats.norm.cdf(z_statistic_adj)
    
    # Standard Z-test
    z_statistic = (test_mean - h0_mean) / (h0_std / np.sqrt(n_test))
    z_p_value_one_sided = 1 - stats.norm.cdf(z_statistic)
    
    # Store results
    results[column] = {
        'h0_mean': h0_mean,
        'h0_std': h0_std,
        'h0_measurement_se': h0_measurement_se,
        'test_mean': test_mean,
        'test_std': test_std,
        'avg_measurement_var': avg_measurement_var,
        'total_se': total_se,
        'n_reference': len(reference_data),
        'n_test': len(test_data),
        'ks_statistic': ks_statistic,
        'ks_p_value': ks_p_value,
        'sw_statistic': sw_statistic,
        'sw_p_value': sw_p_value,
        'ad_statistic': ad_statistic,
        't_statistic': t_statistic,
        't_p_value_one_sided': t_p_value_one_sided,
        't_statistic_adj': t_statistic_adj,
        't_p_value_one_sided_adj': t_p_value_one_sided_adj,
        'z_statistic': z_statistic,
        'z_p_value_one_sided': z_p_value_one_sided,
        'z_statistic_adj': z_statistic_adj,
        'z_p_value_one_sided_adj': z_p_value_one_sided_adj,
        'test_data': test_data.values,
        'reference_data': reference_data.values
    }
    
    # Print results
    print(f"  Sample sizes: Reference = {len(reference_data)}, Test = {len(test_data)}")
    print(f"  Test mean = {test_mean:.3f} vs H0 mean = {h0_mean:.3f}")
    print(f"  Total SE (with measurement uncertainty) = {total_se:.4f}")
    print(f"  Kolmogorov-Smirnov test:")
    print(f"    Statistic = {ks_statistic:.4f}, p-value = {ks_p_value:.4f}")
    
    if not np.isnan(sw_p_value):
        print(f"  Shapiro-Wilk test (standardized data):")
        print(f"    Statistic = {sw_statistic:.4f}, p-value = {sw_p_value:.4f}")
    
    print(f"  Anderson-Darling test:")
    print(f"    Statistic = {ad_statistic:.4f}")
    for i, (critical_val, significance) in enumerate(zip(ad_critical_values, ad_significance_levels)):
        result = "Reject H0" if ad_statistic > critical_val else "Fail to reject H0"
        print(f"    At {significance}% significance: {result} (critical = {critical_val:.4f})")
    
    print(f"  One-sided t-test (H1: μ_test > μ_h0):")
    print(f"    Standard: t = {t_statistic:.4f}, p = {t_p_value_one_sided:.4f}")
    print(f"    Adjusted for measurement uncertainty: t = {t_statistic_adj:.4f}, p = {t_p_value_one_sided_adj:.4f}")
    print(f"  One-sided z-test (H1: μ_test > μ_h0):")
    print(f"    Standard: z = {z_statistic:.4f}, p = {z_p_value_one_sided:.4f}")
    print(f"    Adjusted for measurement uncertainty: z = {z_statistic_adj:.4f}, p = {z_p_value_one_sided_adj:.4f}")
    
    # Interpretation
    alpha = 0.05
    print(f"  Interpretation at α = {alpha}:")
    if ks_p_value > alpha:
        print(f"    KS test: Fail to reject H0 (p = {ks_p_value:.4f})")
    else:
        print(f"    KS test: Reject H0 (p = {ks_p_value:.4f})")
    
    if not np.isnan(sw_p_value):
        if sw_p_value > alpha:
            print(f"    SW test: Fail to reject H0 (p = {sw_p_value:.4f})")
        else:
            print(f"    SW test: Reject H0 (p = {sw_p_value:.4f})")
    
    if t_p_value_one_sided_adj < alpha:
        print(f"    One-sided t-test (adj): Reject H0 - values significantly GREATER (p = {t_p_value_one_sided_adj:.4f})")
    else:
        print(f"    One-sided t-test (adj): Fail to reject H0 (p = {t_p_value_one_sided_adj:.4f})")

# Calculate compounded probabilities
print("\n" + "="*60)
print("COMPOUNDED PROBABILITY ANALYSIS")
print("="*60)

# Get p-values for one-sided tests from all valid columns
one_sided_p_values = []
one_sided_p_values_adj = []  # Adjusted for measurement uncertainty
column_names = []
for column, result in results.items():
    if not np.isnan(result['t_p_value_one_sided']):
        one_sided_p_values.append(result['t_p_value_one_sided'])
        one_sided_p_values_adj.append(result['t_p_value_one_sided_adj'])
        column_names.append(column)

print(f"Testing {len(one_sided_p_values)} columns: {', '.join(column_names)}")
print(f"Standard p-values: {[f'{p:.4f}' for p in one_sided_p_values]}")
print(f"Adjusted p-values (with measurement uncertainty): {[f'{p:.4f}' for p in one_sided_p_values_adj]}")
print()

# Perform analysis for both standard and adjusted p-values
for analysis_type, p_vals in [("Standard", one_sided_p_values), ("Adjusted", one_sided_p_values_adj)]:
    print(f"{analysis_type} Analysis:")
    print("-" * 20)
    
    # Method 1: Bonferroni correction (conservative)
    alpha_bonferroni = 0.05 / len(p_vals)
    bonferroni_significant = sum(1 for p in p_vals if p < alpha_bonferroni)
    
    # Method 2: Fisher's combined probability test
    fisher_statistic = -2 * sum(np.log(p) for p in p_vals)
    fisher_p_value = 1 - stats.chi2.cdf(fisher_statistic, df=2*len(p_vals))
    
    # Method 3: Stouffer's Z-score method
    z_scores = [stats.norm.ppf(1 - p) for p in p_vals]
    combined_z = sum(z_scores) / np.sqrt(len(z_scores))
    stouffer_p_value = 1 - stats.norm.cdf(combined_z)
    
    # Method 4: Probability that ALL columns differ
    prob_all_differ = np.prod([1 - p for p in p_vals])
    
    print(f"1. Bonferroni correction (α = {alpha_bonferroni:.6f}):")
    print(f"   Significant columns: {bonferroni_significant}/{len(p_vals)}")
    print(f"   Conservative conclusion: {'At least one column significantly > H0' if bonferroni_significant > 0 else 'No strong evidence of differences'}")
    print()
    print(f"2. Fisher's combined probability test:")
    print(f"   Test statistic: {fisher_statistic:.4f}")
    print(f"   Combined p-value: {fisher_p_value:.6f}")
    print(f"   Conclusion: {'Reject H0 - evidence that at least one column > H0' if fisher_p_value < 0.05 else 'Fail to reject H0'}")
    print()
    print(f"3. Stouffer's Z-score method:")
    print(f"   Combined Z-score: {combined_z:.4f}")
    print(f"   Combined p-value: {stouffer_p_value:.6f}")
    print(f"   Conclusion: {'Reject H0 - evidence that at least one column > H0' if stouffer_p_value < 0.05 else 'Fail to reject H0'}")
    print()
    print(f"4. Probability that ALL columns differ from H0:")
    print(f"   P(all columns > H0 mean) = {prob_all_differ:.6f}")
    print(f"   This represents the probability that ALL columns simultaneously")
    print(f"   have means greater than their respective H0 means.")
    print()
    
    if analysis_type == "Adjusted":
        print("Note: Adjusted analysis accounts for 10% measurement uncertainty")
        print("      with n=5 repeats per data point, making tests more conservative.")
    print("=" * 60)
# Summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Column':<10} {'KS p-val':<10} {'SW p-val':<10} {'t p-val':<10} {'t p-adj':<10} {'Test Mean':<10} {'H0 Mean':<10}")
print("-" * 80)

for column, result in results.items():
    ks_p = f"{result['ks_p_value']:.4f}"
    sw_p = f"{result['sw_p_value']:.4f}" if not np.isnan(result['sw_p_value']) else "N/A"
    t_p = f"{result['t_p_value_one_sided']:.4f}" if not np.isnan(result['t_p_value_one_sided']) else "N/A"
    t_p_adj = f"{result['t_p_value_one_sided_adj']:.4f}" if not np.isnan(result['t_p_value_one_sided_adj']) else "N/A"
    test_mean = f"{result['test_mean']:.3f}"
    h0_mean = f"{result['h0_mean']:.3f}"
    print(f"{column:<10} {ks_p:<10} {sw_p:<10} {t_p:<10} {t_p_adj:<10} {test_mean:<10} {h0_mean:<10}")

print("\nColumn definitions:")
print("- KS p-val: Kolmogorov-Smirnov test for normality")
print("- SW p-val: Shapiro-Wilk test for normality")  
print("- t p-val: One-sided t-test (standard)")
print("- t p-adj: One-sided t-test adjusted for measurement uncertainty")

# Create visualizations
n_cols = len(test_columns)
fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))
if n_cols == 1:
    axes = axes.reshape(2, 1)

for i, column in enumerate(test_columns):
    if column not in results:
        continue
        
    result = results[column]
    test_data = result['test_data']
    h0_mean = result['h0_mean']
    h0_std = result['h0_std']
    
    # Histogram with H0 distribution overlay
    axes[0, i].hist(test_data, bins=8, alpha=0.7, density=True, 
                    color='skyblue', edgecolor='black', label='Observed data')
    
    # Overlay H0 normal distribution
    x_range = np.linspace(test_data.min() - 2*h0_std, test_data.max() + 2*h0_std, 100)
    h0_pdf = stats.norm.pdf(x_range, h0_mean, h0_std)
    axes[0, i].plot(x_range, h0_pdf, 'r-', linewidth=2, label=f'H0: N({h0_mean:.1f}, {h0_std:.1f})')
    
    # Mark reference points
    reference_data = result['reference_data']
    for ref_val in reference_data:
        axes[0, i].axvline(ref_val, color='green', linestyle='--', alpha=0.7)
    
    axes[0, i].set_title(f'{column}\nTest: {result["test_mean"]:.2f} vs H0: {result["h0_mean"]:.2f}\nt p-val (adj): {result["t_p_value_one_sided_adj"]:.4f}')
    axes[0, i].set_xlabel('Value')
    axes[0, i].set_ylabel('Density')
    axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)
    
    # Add error bars showing measurement uncertainty
    for val in test_data:
        se = (val * measurement_cv) / np.sqrt(n_repeats)
        axes[0, i].errorbar(val, 0, xerr=se, fmt='o', color='red', alpha=0.5, markersize=3)
    
    # Q-Q plot
    standardized = (test_data - h0_mean) / h0_std
    stats.probplot(standardized, dist="norm", plot=axes[1, i])
    axes[1, i].set_title(f'{column} Q-Q Plot\n(Standardized)')
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hypothesis_test_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved as 'hypothesis_test_results.png'")
print("\nGreen dashed lines in histograms show reference model values (LC, BALD, EKM, PW)")
print("Red line shows the H0 normal distribution")