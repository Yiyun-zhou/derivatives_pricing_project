
# Derivatives Pricing (Python)

This is a focused mini-project on **derivatives pricing and risk analytics**, implemented in Python.  
It covers **European and American options**, multiple pricing approaches (Binomial Tree, Monte Carlo, Black–Scholes closed-form), key **Greeks**, and **implied volatility** estimation.  
Extensions include **variance reduction techniques** and **continuous dividend yield** support.

---

## 📝 Project Status
- **First Version Released:** 2025-09-15  
- **Last Update:** 2025-09-28  

### Planned Extensions
- [ ] Additional Greeks (Theta, Rho)  
- [ ] Hedging strategy simulation (e.g., Delta-hedging P&L)  
- [ ] Exotic options prototypes (Asian, Barrier)  
- [ ] Volatility surface construction  

---

## 🚀 Quickstart (macOS / VS Code)

```bash
# 1) Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -e ".[dev]"
# or:
pip install -r requirements.txt

# 3) Enable the venv in VS Code
# (bottom-right prompt or Cmd+Shift+P → Python: Select Interpreter)

# 4) Run tests
pytest

# 5) Try example scripts
python src/deriv_pricing/binomial.py
python src/deriv_pricing/monte_carlo.py
```
---

## 📁 Package Layout

src/deriv_pricing/
  payoff.py         # Vanilla option payoffs + put-call parity helpers
  binomial.py       # CRR binomial tree pricer (Euro & American) + Delta
  monte_carlo.py    # Monte Carlo pricer with antithetic variates
  greeks.py         # Shared Greeks utilities (finite difference)
  utils.py          # Common helpers (e.g., discounting)
  black_scholes.py  # Black–Scholes pricers + closed-form Greeks
  implied_vol.py    # Implied volatility solvers (Brentq, Newton, Bisection)

tests/
  test_payoff.py
  test_binomial.py
  test_monte_carlo.py

notebooks/
  DerivativesPricing.ipynb  # Full project walkthrough + ideas for extensions

---

## 📚 Features

- **Pricing Models**
  - Black–Scholes closed-form (with dividends)
  - Monte Carlo (with variance reduction)
  - Binomial tree (European & American)

- **Risk Metrics**
  - Greeks (Delta, Gamma, Vega; extendable to Theta, Rho)
  - Implied Volatility
  - Root-finding with Brent, Newton–Raphson, Bisection
  - Volatility smile construction

- **Extensions**
  - Dividend yield support
  - Framework for adding exotic options

---

## © License

© 2025-09-15 — Ongoing updates will be tracked in this README.