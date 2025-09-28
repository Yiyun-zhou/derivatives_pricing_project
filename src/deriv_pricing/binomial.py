
from __future__ import annotations
import numpy as np
from .utils import discount_factor, up_down_factors
from .payoff import call_payoff, put_payoff

def binomial_european_option(S0: float, K: float, r: float, q: float, sigma: float, T: float, steps: int, option: str = "call"):
    """Cox-Ross-Rubinstein binomial tree for European call/put."""
    dt = T / steps #discretize time
    u, d = up_down_factors(None, None, sigma, dt)
    # risk-neutral prob
    p = (np.exp((r - q) * dt) - d) / (u - d) #risk-neutral probability
    if not (0.0 <= p <= 1.0):
        raise ValueError("Arbitrage detected: adjust parameters.")

    # terminal prices
    ST = np.array([S0 * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)], dtype=float) #all possible terminal prices

    if option == "call":
        values = call_payoff(ST, K)
    elif option == "put":
        values = put_payoff(ST, K)
    else:
        raise ValueError("option must be 'call' or 'put'")

    disc = np.exp(-r * dt)
    
    # backward induction
    for _ in range(steps, 0, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1]) #values[1:] and values[:-1] make sure we deal with the rights nodes for backward pricing

    return float(values[0])

def delta_binomial(S0: float, K: float, r: float, q: float, sigma: float, T: float, steps: int, option: str = "call", h: float = 1e-4):
    """Finite-difference Delta using small bump h."""
    up = binomial_european_option(S0 * (1 + h), K, r, q, sigma, T, steps, option)
    dn = binomial_european_option(S0 * (1 - h), K, r, q, sigma, T, steps, option)
    return (up - dn) / (2 * S0 * h)

def binomial_american_option(S0: float, K: float, r: float, q: float, sigma: float, T: float, steps: int, option: str = "call"):
    dt = T / steps #discretize time
    u, d = up_down_factors(None, None, sigma, dt)
    # risk-neutral prob
    p = (np.exp((r - q) * dt) - d) / (u - d) #risk-neutral probability
    if not (0.0 <= p <= 1.0):
        raise ValueError("Arbitrage detected: adjust parameters.")

    # terminal prices
    ST = np.array([S0 * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)], dtype=float) #all possible terminal prices
    
    disc = np.exp(-r * dt)

    if option == "call":
        values = call_payoff(ST, K)
    elif option == "put":
        values = put_payoff(ST, K)
    else:
        raise ValueError("option must be 'call' or 'put'")

    for i in range(steps-1, -1, -1): #backward induction
            
        S_i = np.array([S0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)], dtype=float)
        if option == 'call':
            exercise = call_payoff(S_i, K)
        elif option == 'put':
            exercise = put_payoff(S_i, K)
        
        hold = disc * (p * values[1:] + (1-p) * values[:-1])

        values = np.maximum(hold, exercise)

    return float(values[0])


if __name__ == "__main__":
    price = binomial_european_option(S0=100, K=100, r=0.02, sigma=0.2, T=1.0, steps=200, option="call")
    print(f"CRR European call price ~ {price:.4f}")
    delta = delta_binomial(100, 100, 0.02, 0.2, 1.0, 200, "call")
    print(f"Delta ~ {delta:.4f}")
