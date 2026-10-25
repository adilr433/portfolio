"""European option pricing: Black-Scholes-Merton, Greeks, implied volatility,
and a Cox-Ross-Rubinstein binomial tree (European and American exercise)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bsm_price(S, K, T, r, sigma, kind="call", q=0.0):
    """Black-Scholes-Merton price with continuous dividend yield q."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    if kind == "call":
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    if kind == "put":
        return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float   # per 1.00 change in vol (divide by 100 for per vol point)
    theta: float  # per year (divide by 365 for per day)
    rho: float    # per 1.00 change in rate


def bsm_greeks(S, K, T, r, sigma, kind="call", q=0.0) -> Greeks:
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    pdf = norm.pdf(d1)
    gamma = df_q * pdf / (S * sigma * np.sqrt(T))
    vega = S * df_q * pdf * np.sqrt(T)
    common = -S * df_q * pdf * sigma / (2 * np.sqrt(T))
    if kind == "call":
        delta = df_q * norm.cdf(d1)
        theta = common - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)
        rho = K * T * df_r * norm.cdf(d2)
    elif kind == "put":
        delta = -df_q * norm.cdf(-d1)
        theta = common + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)
        rho = -K * T * df_r * norm.cdf(-d2)
    else:
        raise ValueError("kind must be 'call' or 'put'")
    return Greeks(float(delta), float(gamma), float(vega), float(theta), float(rho))


def implied_vol(price, S, K, T, r, kind="call", q=0.0, lo=1e-6, hi=5.0) -> float:
    """Implied volatility by Brent's method; raises if price is outside no-arbitrage bounds."""
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    intrinsic = max(S * df_q - K * df_r, 0) if kind == "call" else max(K * df_r - S * df_q, 0)
    upper = S * df_q if kind == "call" else K * df_r
    if not intrinsic <= price < upper:
        raise ValueError("price violates no-arbitrage bounds")
    return brentq(lambda s: bsm_price(S, K, T, r, s, kind, q) - price, lo, hi, xtol=1e-10)


def binomial_price(S, K, T, r, sigma, kind="call", steps=500, american=False, q=0.0):
    """Cox-Ross-Rubinstein binomial tree. Supports early exercise."""
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    prices = S * u ** (steps - j) * d**j
    payoff = (lambda x: np.maximum(x - K, 0)) if kind == "call" else (lambda x: np.maximum(K - x, 0))
    values = payoff(prices)
    for n in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            prices = prices[:-1] / u
            values = np.maximum(values, payoff(prices))
    return float(values[0])
