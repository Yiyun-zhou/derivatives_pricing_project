
from deriv_pricing.binomial import binomial_european_option

def test_binomial_runs():
    price = binomial_european_option(100, 100, 0.02, 0.2, 1.0, 50, "call")
    assert price > 0
