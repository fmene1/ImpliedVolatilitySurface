import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def option_price(
    S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: str
) -> float:
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "c":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "p":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Must be 'c' or 'p'.")


def implied_volatility(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    market_price: float,
    option_type: str = "c",
    tol: float = 1e-6,
    max_iter: int = 100,
):
    if T <= 0 or market_price <= 0:
        return np.nan
    def target_fn(sigma: float) -> float:
        return (
            option_price(S, K, T, r, sigma, q, option_type) - market_price
        )

    try:
        return brentq(target_fn, 1e-6, 200, xtol=tol, maxiter=max_iter)
    except (ValueError, RuntimeError):
        return np.nan
