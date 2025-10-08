
# Derivatives Pricing (Python)

This is a focused mini-project on **derivatives pricing and risk analytics**, implemented in Python.  
It covers **European and American options**, multiple pricing approaches (Binomial Tree, Monte Carlo, Black–Scholes closed-form), key **Greeks**, and **implied volatility** estimation.  
Extensions include **variance reduction techniques** and **continuous dividend yield** support.

---

## 📝 Project Status
- **First Version Released:** 2025-09-15  
- **Last Update:** 2025-10-08

### Planned Extensions
- [ ] Additional Greeks (Theta, Rho)  
- [ ] Exotic options prototypes (Asian, Barrier)  *Note: Barrier done on 10-08-2025*
- [ ] Volatility Surface deeper extensions (sticky-delta, sticky-strike)

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

```bash
src/deriv_pricing/
│
├── payoff.py          # Vanilla option payoffs + put-call parity helpers
├── binomial.py        # CRR binomial tree pricer (Euro & American) + Delta
├── monte_carlo.py     # Monte Carlo pricer with antithetic variates
├── greeks.py          # Shared Greeks utilities (finite difference)
├── utils.py           # Common helpers (e.g., discounting)
├── black_scholes.py   # Black–Scholes pricers + closed-form Greeks
├── implied_vol.py     # Implied volatility solvers (Brentq, Newton, Bisection)
├── barrier.py         # Pricing Barrier option using closed-form Reiner–Rubinstein and Monte Carlo
└── vol_surface.py     # Simulate market vol data, calibrate under SVI and interpret volatility surface

notebooks/
├── DerivativesPricing.ipynb        # Full project walkthrough + ideas for extensions
├── Exotics.ipynb                   # Project extension to exotics pricing with hedging simulation & PnL tracking
└── Volatility_Surface.ipynb        # Project extension to volatility surface calibration
```
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
  - Volatility surface calibration [Updated 08-10-25]
  - Risk & Hedging simulation [Updated 07-10-25]
  - Structuring/QIS extension 

---

## © License

© 2025-09-15 — Ongoing updates will be tracked in this README.