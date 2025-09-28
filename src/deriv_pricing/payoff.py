
from __future__ import annotations
import numpy as np

def call_payoff(S: float | np.ndarray, K: float) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    return np.maximum(S - K, 0.0)

def put_payoff(S: float | np.ndarray, K: float) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    return np.maximum(K - S, 0.0)

def put_call_parity_price(call_price: float, S0: float, K: float, r: float, q: float, T: float) -> float:
    """Return the theoretical put price from parity: C + Ke^{-rT} - S0 = P"""
    return call_price + K * np.exp(-r * T) + q - S0
