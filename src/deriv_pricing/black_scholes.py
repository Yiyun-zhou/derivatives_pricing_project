from __future__ import annotations
import numpy as np
from scipy.stats import norm

def bs_european_option(S0: float, K: float, r: float, q: float, sigma: float, T: float, option: str = "call"):
    """Black-Scholes Realisation using Martingale Approach"""

    d1 = (np.log(S0/K) + (r - q + 0.5* sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0/K) + (r - q - 0.5* sigma**2)*T) / (sigma * np.sqrt(T))

    if option == 'call':
        bs_value = S0 * norm.cdf(d1) * np.exp(-q*T) - K * np.exp(-r*T) * norm.cdf(d2)
    elif option == 'put':
        bs_value = K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1) * np.exp(-q*T)

    return bs_value

def bs_european_greeks(S0: float, K: float, r: float, q: float, sigma: float, T: float, option: str = "call"):
    """Compute Deltas from BS model"""

    d1 = (np.log(S0/K) + (r + 0.5* sigma**2)*T) / (sigma * np.sqrt(T))

    if option == 'call':
        delta = norm.cdf(d1) * np.exp(-q*T)
    elif option == 'put':
        delta = (norm.cdf(d1) - 1) * np.exp(-q*T)
    
    # gamma and vega for call and put is the same (put-call parity)
    gamma = np.exp(-q*T - 0.5* d1**2) / (S0 * sigma * np.sqrt(2*np.pi*T))
    vega = S0 * norm.cdf(d1) * np.sqrt(T) * np.exp(-q*T)

    return delta, gamma, vega

    