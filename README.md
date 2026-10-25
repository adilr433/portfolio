# quantkit

![tests](https://github.com/adilr433/portfolio/actions/workflows/tests.yml/badge.svg)

A small, tested Python library covering four core parts of a quant workflow:
**pricing derivatives, building portfolios, measuring risk and backtesting signals.**
Every function is checked against closed-form results or independent numerical
methods, and all examples run offline on reproducible synthetic data.

| Module | What it does |
|---|---|
| `quantkit.options` | Black-Scholes-Merton (with dividend yield), full Greeks, implied volatility (Brent), CRR binomial tree with American exercise |
| `quantkit.optimization` | Mean-variance: minimum variance, maximum Sharpe, target return, efficient frontier; long-only and weight caps |
| `quantkit.risk` | VaR and Expected Shortfall three ways (parametric, historical, Monte Carlo), horizon scaling, max drawdown, scenario stress tests |
| `quantkit.backtest` | Vectorized backtester with one-bar execution lag and turnover costs; MA-crossover and momentum signals; Sharpe, Sortino, Calmar, drawdown |
| `quantkit.data` | Synthetic GBM prices and fat-tailed (Student-t) correlated returns |

## Quick start

```bash
git clone https://github.com/adilr433/portfolio.git
cd portfolio
pip install -r requirements.txt
python examples/demo.py      # end-to-end walkthrough
python -m pytest             # 27 tests
```

```python
from quantkit import options, optimization, risk, data

options.bsm_price(S=100, K=100, T=1, r=0.05, sigma=0.2, kind="call")   # 10.4506
options.binomial_price(100, 100, 1, 0.05, 0.2, "put", steps=1000, american=True)  # 6.0896

rets = data.simulate_returns(5, seed=42)
mu, cov = optimization.annualize(rets)
port = optimization.max_sharpe(mu, cov, rf=0.03, max_weight=0.6)
risk.risk_report(rets, port.weights.values, alpha=0.99)
```

## How it's verified

- **Options:** BSM matches the textbook reference (call 10.4506 / put 5.5735 for S=K=100, T=1, r=5%, σ=20%); put-call parity holds with dividends; every Greek matches a finite-difference derivative; implied vol round-trips; the binomial tree converges to BSM, and the American put carries an early-exercise premium while the American call doesn't.
- **Optimization:** unconstrained minimum-variance and tangency weights match their closed-form solutions; the max-Sharpe portfolio dominates every frontier point; weight caps and target returns are respected.
- **Risk:** on Gaussian data, parametric, historical and Monte Carlo VaR/ES agree; ES ≥ VaR; √T horizon scaling is exact.
- **Backtest:** an always-long position reproduces buy-and-hold exactly; positions are lagged one bar (no look-ahead); transaction costs reduce returns.

## Sample output (synthetic data, `examples/demo.py`)

| | Min variance | Max Sharpe (60% cap) |
|---|---|---|
| Expected return | 5.88% | 14.64% |
| Volatility | 8.14% | 13.81% |
| Sharpe (rf 3%) | 0.35 | 0.84 |

1-day 99% risk of the max-Sharpe portfolio: parametric VaR 1.97%, historical VaR 2.35%, historical ES 2.72%. The gap between parametric and historical figures comes from the fat tails (Student-t, 5 df) in the simulated returns, which a Gaussian model understates.

These numbers come from simulated data and say nothing about real-market performance.

## Design choices

- **Simple returns, positive-loss convention** for VaR/ES, so reports read as "lose at most X% on 99% of days."
- **SLSQP** for constrained optimization, so the library only needs SciPy. The closed-form tests guard its accuracy.
- **No look-ahead by construction:** a position decided at the close of day *t* earns day *t+1*'s return, and costs are charged on turnover.

## License

MIT © Adil Rafi
