# Compare three root-finding algorithms for solving implied vol

from scipy.optimize import brentq, newton, bisect
from deriv_pricing.black_scholes import bs_european_option
import numpy as np
from deriv_pricing.greeks import vega_fd

def option_market_price(S0, K, r, q, sigma, T, option: str = "call", sigma_noise: int = 0.01, seed: int | None = 42):
    """Compute a hypothetical market price for solving implied vol"""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma_noise)
    market_price = bs_european_option(S0, K, r, q, sigma, T, option) * (1+noise)

    return market_price


def f_sigma(sigma, S0, K, r, q, T, V_mkt, option="call"):
    diff = bs_european_option(S0, K, r, q, sigma, T, option) - V_mkt
    return float(diff)

# Brentq

def IV_brentq(f_sigma, lo, hi, S0, K, r, q, T, V_mkt, option):

    iv_brent = brentq(
        f_sigma,
        lo, hi,
        args=(S0, K, r, q, T, V_mkt, option),
        xtol=1e-12, rtol=1e-12, maxiter=100
    )

    return iv_brent

# Newton-Raphson
x0 = 0.2  #Initial Guess

def IV_Newton(x0, S0, K, r, q, T, V_mkt, option, h):

    iv_newton = newton(
        func=lambda s: float(f_sigma(s, S0, K, r, q, T, V_mkt, option)),
        x0=x0,
        fprime=lambda s: float(vega_fd(
            pricer=lambda vol: bs_european_option(S0, K, r, q, vol, T, option),
            sigma=s,
            h=h)
        ),
        tol=1e-12, maxiter=50
    )

    return iv_newton


# Bisect
lo, hi = 1e-8, 5.0
def IV_Bisect(lo, hi, S0, K, r, q, T, V_mkt, option):
    iv_bisect = bisect(
    f_sigma,
    lo, hi,
    args=(S0, K, r, q, T, V_mkt, option),
    xtol=1e-12, rtol=1e-12, maxiter=200
)
    
    return iv_bisect


