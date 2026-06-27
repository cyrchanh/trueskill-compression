import numpy as np
from scipy.stats import norm
from typing import Tuple
from .model import GaussianBelief, BETA

_pdf = norm.pdf
_cdf = norm.cdf


def v_win(t: float, eps: float) -> float:
    denom = _cdf(t - eps)
    if denom < 1e-10:
        return float(eps - t)
    return float(_pdf(t - eps) / denom)


def w_win(t: float, eps: float) -> float:
    v = v_win(t, eps)
    return v * (v + t - eps)


def v_draw(t: float, eps: float) -> float:
    denom = _cdf(eps - t) - _cdf(-eps - t)
    if abs(denom) < 1e-10:
        return 0.0
    return float((_pdf(-eps - t) - _pdf(eps - t)) / denom)


def w_draw(t: float, eps: float) -> float:
    v = v_draw(t, eps)
    denom = _cdf(eps - t) - _cdf(-eps - t)
    if abs(denom) < 1e-10:
        return 1.0
    return float(v ** 2 + ((eps - t) * _pdf(eps - t) + (eps + t) * _pdf(eps + t)) / denom)


def two_player_update(
    b1: GaussianBelief,
    b2: GaussianBelief,
    outcome: int,
    beta: float = BETA,
    epsilon: float = 0.0,
) -> Tuple[GaussianBelief, GaussianBelief]:
    c_sq = b1.sigma ** 2 + b2.sigma ** 2 + 2.0 * beta ** 2
    sqrt_c = np.sqrt(c_sq)

    if outcome == 1:
        t = (b1.mu - b2.mu) / sqrt_c
        eps_norm = epsilon / sqrt_c
        v = v_win(t, eps_norm)
        w = w_win(t, eps_norm)
        sign1, sign2 = 1.0, -1.0
    elif outcome == -1:
        t = (b2.mu - b1.mu) / sqrt_c
        eps_norm = epsilon / sqrt_c
        v = v_win(t, eps_norm)
        w = w_win(t, eps_norm)
        sign1, sign2 = -1.0, 1.0
    else:
        t = (b1.mu - b2.mu) / sqrt_c
        eps_norm = epsilon / sqrt_c
        v = v_draw(t, eps_norm)
        w = w_draw(t, eps_norm)
        sign1, sign2 = 1.0, -1.0

    mu1_new = b1.mu + sign1 * (b1.sigma ** 2 / sqrt_c) * v
    mu2_new = b2.mu + sign2 * (b2.sigma ** 2 / sqrt_c) * v
    sigma1_new = np.sqrt(max(b1.sigma ** 2 * (1.0 - (b1.sigma ** 2 / c_sq) * w), 1e-10))
    sigma2_new = np.sqrt(max(b2.sigma ** 2 * (1.0 - (b2.sigma ** 2 / c_sq) * w), 1e-10))

    return GaussianBelief(mu1_new, sigma1_new), GaussianBelief(mu2_new, sigma2_new)
