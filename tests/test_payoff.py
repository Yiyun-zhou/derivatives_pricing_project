
import numpy as np
from deriv_pricing.payoff import call_payoff, put_payoff, put_call_parity_price

def test_payoffs():
    S = np.array([80, 100, 120], dtype=float)
    K = 100
    cp = call_payoff(S, K)
    pp = put_payoff(S, K)
    assert (cp == np.array([0, 0, 20])).all()
    assert (pp == np.array([20, 0, 0])).all()

def test_parity():
    call = 10.0
    S0, K, r, T = 100.0, 100.0, 0.02, 1.0
    put = put_call_parity_price(call, S0, K, r, T)
    # loose check
    assert 7.0 < put < 12.0
