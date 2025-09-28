
from deriv_pricing.monte_carlo import mc_european_call

def test_mc_runs():
    p, se = mc_european_call(100, 100, 0.02, 0.2, 1.0, n_paths=10_000)
    assert p > 0 and se > 0
