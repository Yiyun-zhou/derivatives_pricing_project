
from __future__ import annotations
import numpy as np

def mc_european_call(S0: float, K: float, r: float, q: float, sigma: float, T: float, n_paths: int = 100_000, antithetic: bool = True, seed: int | None = 42):
    rng = np.random.default_rng(seed)
    if antithetic:
        z = rng.standard_normal(n_paths // 2)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(ST - K, 0.0)
    price = np.exp(-r * T) * payoff.mean()
    stderr = np.exp(-r * T) * payoff.std(ddof=1) / np.sqrt(len(payoff))
    return float(price), float(stderr)


def control_var(S0: float, K: float, r: float, q: float, sigma: float, T: float, n_paths: int = 100_000, seed: int | None = 42):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(ST - K, 0.0)
    Y = np.exp(-r*T) * payoff
    X = np.exp(-r*T) * ST
    b_star = np.cov(X, Y)[0,1] / np.var(X)
    Y_i = Y - b_star * (X - X.mean())
    return Y_i.mean(), Y_i.std() / len(Y_i)**0.5


if __name__ == "__main__":
    p, se = mc_european_call(100, 100, 0.02, 0.2, 1.0, n_paths=100_000)
    print(f"MC European call ~ {p:.4f} ± {1.96*se:.4f} (95% CI)")
