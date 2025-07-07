import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from statsmodels.stats.power import TTestIndPower
import pandas as pd

def split_test_plot(
    series_1: pd.Series,
    series_2: pd.Series,
    bootstrap: bool = False,
    samples: int = 1000,
    two_tailed: bool = True,
    random_state: int = 42
):
    """
    Compare two distributions (series_1 and series_2) using either a classical t-test or bootstrapping.
    
    Parameters:
    ----------
    series_1 : pd.Series
        First sample distribution
    series_2 : pd.Series
        Second sample distribution
    bootstrap : bool
        Whether to use bootstrapping (True) or a classical t-test (False)
    samples : int
        Number of bootstrap samples to generate (if bootstrap=True)
    two_tailed : bool
        If True, performs a two-tailed test; otherwise, one-tailed
    random_state : int
        Random seed for reproducibility

    Returns:
    -------
    dict
        Dictionary with summary statistics, p-values, effect size, confidence intervals, and plotting data
    """

    alpha = 0.05
    mean_1 = np.mean(series_1)
    mean_2 = np.mean(series_2)
    std_1 = np.std(series_1, ddof=1)
    std_2 = np.std(series_2, ddof=1)
    diff_means = mean_2 - mean_1

    result = {
        "mean_1": mean_1,
        "mean_2": mean_2,
        "diff_means": diff_means
    }

    if not bootstrap:
        # ---------- Classical Welch's t-test ----------
        test_result = st.ttest_ind(series_1, series_2, equal_var=False)
        pvalue = test_result.pvalue

        if not two_tailed:
            pvalue /= 2

        # Welch’s t-test: compute CI manually
        se_diff = np.sqrt(std_1**2 / len(series_1) + std_2**2 / len(series_2))
        df_num = (std_1**2 / len(series_1) + std_2**2 / len(series_2))**2
        df_denom = ((std_1**2 / len(series_1))**2 / (len(series_1) - 1) +
                    (std_2**2 / len(series_2))**2 / (len(series_2) - 1))
        df = df_num / df_denom

        t_crit = st.t.ppf(1 - alpha/2 if two_tailed else 1 - alpha, df)
        ci_low = diff_means - t_crit * se_diff
        ci_high = diff_means + t_crit * se_diff if two_tailed else None

        # Effect size (Cohen's d with pooled std)
        pooled_std = np.sqrt(((std_1**2) + (std_2**2)) / 2)
        effect_size = diff_means / pooled_std

        # Power analysis
        analysis = TTestIndPower()
        power = analysis.solve_power(
            effect_size=effect_size,
            nobs1=len(series_1),
            alpha=alpha,
            ratio=len(series_2) / len(series_1),
            alternative='two-sided' if two_tailed else 'larger'
        )
        beta = 1 - power

        dist_x = np.linspace(diff_means - 4 * se_diff, diff_means + 4 * se_diff, 500)
        dist_y = st.norm.pdf(dist_x, diff_means, se_diff)

        result.update({
            "method": "t-test",
            "pvalue": pvalue,
            "95%_CI": (ci_low, ci_high),
            "effect_size": effect_size,
            "se_diff": se_diff,
            "df": df,
            "power": power,
            "beta": beta,
            "dist_x": dist_x,
            "dist_y": dist_y,
            "ci_low": ci_low,
            "ci_high": ci_high
        })

        return result

    else:
        # ---------- Bootstrapping ----------
        rng = np.random.default_rng(random_state)
        pooled = pd.concat([series_1, series_2]).reset_index(drop=True)
        observed_diff = diff_means
        boot_diffs = []

        for _ in range(samples):
            sample = pooled.sample(frac=1, replace=True, random_state=rng.integers(1e9))
            s1 = sample.iloc[:len(series_1)]
            s2 = sample.iloc[len(series_1):]
            boot_diffs.append(np.mean(s2) - np.mean(s1))

        boot_diffs = np.array(boot_diffs)

        ci_low = np.percentile(boot_diffs, 2.5)
        ci_high = np.percentile(boot_diffs, 97.5)

        if two_tailed:
            pvalue = np.mean(np.abs(boot_diffs) >= np.abs(observed_diff))
        else:
            pvalue = np.mean(boot_diffs >= observed_diff)

        # ---------- Plot ----------
        plt.figure(figsize=(14, 8))

        # Top plot: fitted normal distributions
        plt.subplot(2, 1, 1)
        x = np.linspace(min(series_1.min(), series_2.min()) - 10,
                        max(series_1.max(), series_2.max()) + 10, 500)
        pdf1 = st.norm.pdf(x, mean_1, std_1)
        pdf2 = st.norm.pdf(x, mean_2, std_2)

        plt.plot(x, pdf1, 'b-', lw=2, label=f'Series 1 (Mean={mean_1:.2f})')
        plt.plot(x, pdf2, 'r-', lw=2, label=f'Series 2 (Mean={mean_2:.2f})')
        plt.axvline(mean_1, color='b', linestyle='--', lw=2)
        plt.axvline(mean_2, color='r', linestyle='--', lw=2)
        plt.title('Fitted Normal Distributions')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

        # Bottom plot: bootstrap distribution
        plt.subplot(2, 1, 2)
        bins = max(20, int(np.sqrt(samples)))
        density, edges, _ = plt.hist(boot_diffs, bins=bins, color='skyblue', alpha=0.6,
                                     density=True, label='Bootstrap Distribution')

        x_vals = np.linspace(min(boot_diffs), max(boot_diffs), 1000)
        kde = st.gaussian_kde(boot_diffs)
        y_vals = kde(x_vals)

        # Highlight rejection regions
        if two_tailed:
            plt.fill_between(x_vals, 0, y_vals, where=(x_vals <= ci_low), color='red', alpha=0.3,
                             label='Rejection Region (α/2)')
            plt.fill_between(x_vals, 0, y_vals, where=(x_vals >= ci_high), color='red', alpha=0.3)
        else:
            plt.fill_between(x_vals, 0, y_vals, where=(x_vals >= ci_high), color='red', alpha=0.3,
                             label='Rejection Region (α)')

        plt.plot(x_vals, y_vals, color='black', lw=2, label='KDE Curve')
        plt.axvline(observed_diff, color='red', linestyle='--', lw=2, label=f'Observed Diff = {observed_diff:.2f}')
        plt.axvline(ci_low, color='purple', linestyle='--', lw=2, label=f'95% CI Low = {ci_low:.2f}')
        if two_tailed:
            plt.axvline(ci_high, color='purple', linestyle='--', lw=2, label=f'95% CI High = {ci_high:.2f}')

        plt.title('Hypothesis Test: Difference in Means (Bootstrap)')
        plt.xlabel('Difference in Means')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.show()

        result.update({
            "method": "bootstrap",
            "bootstrap_samples": samples,
            "bootstrap_diffs": boot_diffs,
            "observed_diff": observed_diff,
            "pvalue": pvalue,
            "95%_CI": (ci_low, ci_high if two_tailed else None)
        })

        return result


# # Example usage:
# if __name__ == "__main__":
#     np.random.seed(0)
#     sample_before = np.random.normal(loc=50, scale=10, size=100)
#     sample_after = np.random.normal(loc=55, scale=10, size=100)
#     split_test_plot(sample_before, sample_after)