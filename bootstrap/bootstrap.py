"""
Library for boostraping statistics

Citation:
Efron, Bradley, and Robert J. Tibshirani.
An introduction to the bootstrap. CRC press, 1994.
"""

from collections import namedtuple
import numpy as np
from scipy import stats

def bootstrap_sample(data, parametric=False):
    """
    Resamples data by random sample with replacement

    Args:
        data (np.array): array of data to resample
        parametric (str) in ['normal', 'uniform']: parametric distribution to resample from
            if False, use nonparametric bootstrap sampling

    Returns:
        bootstrap resampled data (array)
    """
    sample_size = len(data)
    if parametric == 'normal':
        mean_estimate = np.mean(data)
        std_estimate = np.std(data)
        return np.random.normal(mean_estimate, std_estimate, size=sample_size)
    elif parametric == 'uniform':
        min_estimate, max_estimate = np.min(data), np.max(data)
        return np.random.uniform(min_estimate, max_estimate, size=sample_size)
    else:
        indices = [np.random.randint(0, sample_size) for i in range(sample_size)]
        return data[indices]

def bootstrap_matrixsample(data, axis=0):
    """
    Resamples a matrix by rows or columns

    Args:
     data (np.matrix): matrix of data to resample
     axis (int) in [0, 1]: axis to resample by
         if 0: resample rows
         if 1: resample columns

    Returns:
        bootstrap resampled data (matrix)
    """

    if axis == 0:
        n_rows = np.shape(data)[0]
        samples = np.random.randint(n_rows, size=n_rows)
        bootstrap_matrix = data[samples, :]
    elif axis == 1:
        n_cols = np.shape(data)[1]
        samples = np.random.randint(n_cols, size=n_cols)
        bootstrap_matrix = data[:, samples]
    return bootstrap_matrix

def jackknife_sample(data, index):
    """
    Single jackknife sample of data

    Args:
        data (np.array): array of data to resample
        index (int): Index of array to leave out in jackknife sample

    Returns:
        jackknife resampled data (array)
    """
    jackknife = np.delete(data, index)
    return jackknife

def bootstrap_statistic(data, func=np.mean, n_samples=50,
                        parametric=False, bias_correction=False,
                        alpha=0.05, bca=False, axis=0):
    """
    Bootstraps a statistic and calculates the standard error of the statistic

    Args:
        data (array or matrix): array or matrix of data to calculate statistic
            and SE of statistic
        func (function): statistical function to calculate on data
            examples: np.mean, np.median
        n_samples (int): number of bootstrap samples to calculate statistic for
        parametric (str) in ['normal', 'uniform']: parametric distribution to resample from
            if False, use nonparametric bootstrap sampling
        bias_correction (bool): if True, bias correct bootstrap statistic
        bca (bool): If true, use bias correction and accelerated (BCa) method to
            calculate bootstrap
        axis (int) in [0, 1]: if type(data) == np.matrix, axis to resample by
            if 0: resample rows
            if 1: resample columns

    Returns:
        tuple: (statistic (float), bias (float), sem (float))
        Returns the bootstrapped statistic, its bias and SEM.
    """
    plugin_estimate = func(data)
    statistics = []
    for sample in range(n_samples):
        if isinstance(data, np.matrix):
            resample = bootstrap_matrixsample(data, axis=axis)
        else:
            resample = bootstrap_sample(data, parametric=parametric)
        statistic = func(resample)
        statistics.append(statistic)
    statistic = np.mean(statistics)
    confidence_interval = calculate_ci(data, statistics, func=func, alpha=alpha, bca=bca)
    bias = statistic - plugin_estimate
    if bias_correction:
        statistic = statistic - bias
    sem = stats.sem(statistics)
    bootstrap_results = namedtuple('bootstrap_results', 'statistics statistic bias sem ci')
    results = bootstrap_results(statistics=statistics, statistic=statistic, bias=bias, 
                                sem=sem, ci=confidence_interval)
    return results

def jackknife_statistic(data, func=np.mean):
    """
    Jackknifes a statistic and calculates the standard error of the statistic

    Args:
        data (array): array of data to calculate statistic and SE of statistic
        func (function): statistical function to calculate on data
            examples: np.mean, np.median

    Returns:
        statistic (tuple), sem (tuple)
        Returns the jackknifed statistic and the SEM of the statistic
    """
    n_samples = len(data)
    statistics = []
    for sample in range(n_samples):
        jack_sample = jackknife_sample(data, sample)
        statistic = func(jack_sample)
        statistics.append(statistic)
    return (np.mean(statistics), stats.sem(statistics), statistics)

def calculate_ci(data, statistics, func=np.mean,
                 alpha=0.05, bca=False):
    """
    Calculates bootstrapped confidence interval using percentile
    intervals.

    Args:
        statistics (array): array of bootstrapped statistics to calculate
            confidence interval for
        alpha (float): percentile used for upper and lower bounds of confidence
            interval.  NOTE: Currently, both upper and lower bounds can have
            the same alpha.
        bca (bool): If true, use bias correction and accelerated (BCa) method
        theta_hat (float): Original estimate of the statistic from the data.
            Used to calculate BCa confidence interval.

    Returns: tuple (ci_low, ci_high)
        ci_low (float): lower bound on confidence interval
        ci_high (float): upper bound on confidence interval
    """
    # If BCa method, update alpha
    if bca:
        #Calculate bias term, z
        plugin_estimate = func(data)
        bias_fraction = len(np.where(statistics < plugin_estimate)[0]) / len(statistics)
        z = stats.norm.ppf(bias_fraction)
        # Calculate acceleration term, a
        j_statistic, j_sem, j_values = jackknife_statistic(data, func)
        numerator, denominator = 0, 0
        for value in j_values:
            numerator += (value - j_statistic)**3
            denominator += (value - j_statistic)**2
        a = numerator / (6 * denominator**(3/2))
        bca_alpha = stats.norm.cdf(z + (z + stats.norm.ppf(alpha)) /
                                   1 - a * (z + stats.norm.ppf(alpha)))
        alpha = bca_alpha
    sorted_statistics = np.sort(statistics)
    low_index = int(np.floor(alpha * len(statistics)))
    high_index = int(np.ceil((1 - alpha) * len(statistics)))
    #Correct for 0 based indexing
    if low_index > 0:
        low_index -= 1
    high_index -= 1
    low_value = sorted_statistics[low_index]
    high_value = sorted_statistics[high_index]
    return (low_value, high_value)

def two_sample_testing(sampleA, sampleB,
                       statistic_func=None, n_samples=50):
    """
    Compares two samples via bootstrapping to determine if they came from
    the same distribution.

    Args:
        sampleA (np.array): Array of data from sample A
        sampleB (np.array): Array of data form sample B
        statistic_func (function): Function that compares two data sets
            returns statistic. Function must exept two args (np.array, np.array)
            where each array is a sample.
            Example: lambda data1, data2: np.mean(data1) - np.mean(data2)
            This fucntion compares the means of two data sets
        n_samples: number of bootstrap samples to generate

    Returns:
        bootstrapped achieved significance level (float)
    """
    if statistic_func is None:
        statistic_func = compare_means
    observed_statistic = statistic_func(sampleA, sampleB)
    combined_sample = np.append(sampleA, sampleB)
    #Store sample sizes
    m = len(sampleA)
    n = len(sampleB)
    #Count the number of bootstrap samples with statistic > observed_statistic
    counter = 0
    for sample in range(n_samples):
        boot_sample = bootstrap_sample(combined_sample)
        boot_sampleA = boot_sample[:m]
        boot_sampleB = boot_sample[m:]
        boot_statistic = statistic_func(boot_sampleA, boot_sampleB)
        if boot_statistic > observed_statistic:
            counter += 1
    ASL = counter / float(n_samples)
    return ASL

def compare_means(sampleA, sampleB):
    """
    Compares the mean of two samples

    Args:
        sampleA (np.array): Array of data from sample A
        sampleB (np.array): Array of data form sample B

    Returns:
        difference (float): difference in mean between the two samples
    """
    difference = np.mean(sampleA) - np.mean(sampleB)
    return difference

def t_test_statistic(sampleA, sampleB):
    """
    Computes the t test statistic of two samples

    Args:
        sampleA (np.array): Array of data from sample A
        sampleB (np.array): Array of data form sample B

    Returns:
        t_stat (float): t test statistic of two sampels
    """
    difference = compare_means(sampleA, sampleB)
    #Store lengths of samples
    n = len(sampleA)
    m = len(sampleB)
    stdev = (np.var(sampleA)/n + np.var(sampleB)/m)**0.5
    t_stat = difference / stdev
    return t_stat
