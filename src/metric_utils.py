# src/metrics_utils.py
import numpy as np
from scipy.stats import ks_2samp

def population_stability_index(expected, actual, bins=10):
    expected_perc, _ = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)
    expected_perc = expected_perc / np.sum(expected_perc)
    actual_perc = actual_perc / np.sum(actual_perc)
    psi = np.sum((expected_perc - actual_perc) * np.log((expected_perc + 1e-6) / (actual_perc + 1e-6)))
    return psi

def kolmogorov_smirnov(expected, actual):
    _, p_value = ks_2samp(expected, actual)
    return p_value
