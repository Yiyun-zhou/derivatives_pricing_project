
from __future__ import annotations
import numpy as np

def discount_factor(r: float, T: float) -> float:
    return float(np.exp(-r * T))

def up_down_factors(u: float | None, d: float | None, sigma: float | None, dt: float):
    """Return (u, d). If u/d not given, use CRR with sigma, dt."""
    if u is not None and d is not None:
        return float(u), float(d)
    assert sigma is not None, "Provide either (u,d) or sigma."
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    return float(u), float(d)
