
from __future__ import annotations
from typing import Callable
import numpy as np

def finite_diff(func: Callable[[float], float], x: float, h: float = 1e-4) -> float:
    return (func(x + h) - func(x - h)) / (2 * h)

def delta_fd(pricer: Callable[[float], float], S0: float, h: float = 1e-4) -> float:
    return finite_diff(pricer, S0, h)

def gamma_fd(pricer: Callable[[float], float], S0: float, h: float = 1e-4) -> float:
    return (pricer(S0+h) - 2*pricer(S0) + pricer(S0-h)) / (h**2)

def vega_fd(pricer: Callable[[float], float], sigma: float, h: float = 1e-4) -> float:
    return finite_diff(pricer, sigma, h)
