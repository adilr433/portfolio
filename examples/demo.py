"""End-to-end walkthrough of quantkit on reproducible synthetic data.

Run from the repo root:  python examples/demo.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

from quantkit import backtest, data, optimization, options, risk

pd.set_option("display.float_format", "{:,.4f}".format)


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# 1. Option pricing ----------------------------------------------------------
section("1. Option pricing (S=100, K=100, T=1y, r=5%, vol=20%)")
S, K, T, r, vol = 100, 100, 1.0, 0.05, 0.20
call = options.bsm_price(S, K, T, r, vol, "call")
put = options.bsm_price(S, K, T, r, vol, "put")
print(f"BSM call {call:.4f} | put {put:.4f}")
print(f"Binomial (1000 steps) European put {options.binomial_price(S, K, T, r, vol, 'put', 1000):.4f}"
      f" | American put {options.binomial_price(S, K, T, r, vol, 'put', 1000, american=True):.4f}")
print("Call Greeks:", options.bsm_greeks(S, K, T, r, vol, "call"))
print(f"Implied vol recovered from call price: {options.implied_vol(call, S, K, T, r):.4%}")

# 2. Portfolio optimization ---------------------------------------------------
section("2. Portfolio optimization (5 assets, long-only)")
rets = data.simulate_returns(5, seed=42)
mu, cov = optimization.annualize(rets)
rf = 0.03
mv = optimization.min_variance(mu, cov, rf)
ms = optimization.max_sharpe(mu, cov, rf, max_weight=0.6)
print(pd.DataFrame({"Min variance": mv.weights, "Max Sharpe (cap 60%)": ms.weights}))
for name, p in [("Min variance", mv), ("Max Sharpe", ms)]:
    print(f"{name:13s} return {p.expected_return:.2%}  vol {p.volatility:.2%}  Sharpe {p.sharpe:.2f}")
frontier = optimization.efficient_frontier(mu, cov, 10, rf)
print("\nEfficient frontier:\n", frontier.to_string(index=False))

# 3. Risk ------------------------------------------------------------------------
section("3. Risk of the max-Sharpe portfolio (1-day, 99%)")
print(risk.risk_report(rets, ms.weights.values, alpha=0.99))
shocks = {
    "Equity crash (-20% all)": np.full(5, -0.20),
    "Rotation (+5/-10)": np.array([0.05, 0.05, -0.10, -0.10, 0.0]),
}
print("\nStress tests (P&L as fraction of NAV):\n", risk.stress_test(ms.weights.values, shocks))

# 4. Backtest --------------------------------------------------------------------
section("4. Backtest: MA crossover vs momentum vs buy & hold (5 bps costs)")
prices = data.simulate_gbm(seed=11, days=252 * 10)
results = {
    "MA 50/200": backtest.run_backtest(prices, backtest.ma_crossover(prices, 50, 200)),
    "Momentum 126d": backtest.run_backtest(prices, backtest.momentum(prices, 126)),
    "Buy & hold": backtest.run_backtest(prices, pd.Series(1.0, index=prices.index), 0),
}
print(pd.DataFrame({k: v.stats for k, v in results.items()}))
