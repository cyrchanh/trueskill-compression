import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import norm
from .model import GaussianBelief, BETA


def match_quality(
    b1: GaussianBelief,
    b2: GaussianBelief,
    beta: float = BETA,
) -> float:
    beta_sq = beta ** 2
    denom = 2.0 * beta_sq + b1.sigma ** 2 + b2.sigma ** 2
    sqrt_term = np.sqrt(2.0 * beta_sq / denom)
    exp_term = np.exp(-0.5 * (b1.mu - b2.mu) ** 2 / denom)
    return float(sqrt_term * exp_term)


def win_probability(
    b1: GaussianBelief,
    b2: GaussianBelief,
    beta: float = BETA,
) -> float:
    denom = np.sqrt(2.0 * beta ** 2 + b1.sigma ** 2 + b2.sigma ** 2)
    return float(norm.cdf((b1.mu - b2.mu) / denom))


def naive_best_match(
    target: str,
    pool: List[str],
    beliefs: Dict[str, GaussianBelief],
    beta: float = BETA,
) -> Tuple[str, float]:
    best_opponent = None
    best_quality = -1.0
    b_target = beliefs[target]

    for candidate in pool:
        if candidate == target:
            continue
        q = match_quality(b_target, beliefs[candidate], beta)
        if q > best_quality:
            best_quality = q
            best_opponent = candidate

    return best_opponent, best_quality
