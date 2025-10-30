# tests/test_metrics.py
import numpy as np
import pytest
from src.metric_utils import population_stability_index, kolmogorov_smirnov

@pytest.fixture
def synthetic_distributions():
    np.random.seed(42)
    base = np.random.normal(0, 1, 1000)
    drifted = np.random.normal(0.5, 1, 1000)
    return base, drifted

def test_psi_detects_drift(synthetic_distributions):
    base, drifted = synthetic_distributions
    psi = population_stability_index(base, drifted, bins=10)
    assert psi > 0.08, f"PSI too low ({psi}) for obvious drift" #PSI values between 0.05 and 0.1 indicate mild drift

def test_ks_detects_drift(synthetic_distributions):
    base, drifted = synthetic_distributions
    ks_p = kolmogorov_smirnov(base, drifted)
    assert ks_p < 0.05, f"KS test failed to detect drift (p={ks_p})"
