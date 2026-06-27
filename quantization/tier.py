import numpy as np
from typing import List, Tuple


def lloyd_max_quantizer(
    data: np.ndarray,
    n_levels: int,
    n_iter: int = 200,
    tol: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray]:
    quantile_values = np.percentile(data, np.linspace(0, 100, n_levels + 1))
    boundaries = quantile_values[1:-1].copy()

    centroids = np.zeros(n_levels)
    for k in range(n_levels):
        low = boundaries[k - 1] if k > 0 else -np.inf
        high = boundaries[k] if k < n_levels - 1 else np.inf
        mask = (data >= low) & (data < high)
        centroids[k] = np.mean(data[mask]) if np.any(mask) else (quantile_values[k] + quantile_values[k + 1]) / 2.0

    for _ in range(n_iter):
        assignments = np.searchsorted(boundaries, data)
        new_centroids = np.zeros(n_levels)
        for k in range(n_levels):
            mask = assignments == k
            new_centroids[k] = np.mean(data[mask]) if np.any(mask) else centroids[k]

        new_boundaries = 0.5 * (new_centroids[:-1] + new_centroids[1:])

        if np.max(np.abs(new_centroids - centroids)) < tol:
            break

        centroids = new_centroids
        boundaries = new_boundaries

    return boundaries, centroids


class TierQuantizer:
    def __init__(
        self,
        n_tiers: int,
        boundaries: np.ndarray,
        centroids: np.ndarray,
        tier_names: List[str] = None,
    ):
        self.n_tiers = n_tiers
        self.boundaries = boundaries
        self.centroids = centroids
        self.tier_names = tier_names or [f"Tier_{i + 1}" for i in range(n_tiers)]

    @classmethod
    def fit(
        cls,
        conservative_scores: np.ndarray,
        n_tiers: int,
        tier_names: List[str] = None,
    ) -> "TierQuantizer":
        boundaries, centroids = lloyd_max_quantizer(conservative_scores, n_tiers)
        return cls(n_tiers, boundaries, centroids, tier_names)

    def quantize(self, score: float) -> int:
        return int(np.clip(np.searchsorted(self.boundaries, score), 0, self.n_tiers - 1))

    def reconstruct(self, tier: int) -> float:
        return float(self.centroids[np.clip(tier, 0, self.n_tiers - 1)])

    def distortion(self, scores: np.ndarray) -> float:
        tiers = np.clip(np.searchsorted(self.boundaries, scores), 0, self.n_tiers - 1)
        reconstructed = self.centroids[tiers]
        return float(np.mean((scores - reconstructed) ** 2))

    def population_balance(self, scores: np.ndarray) -> np.ndarray:
        tiers = np.clip(np.searchsorted(self.boundaries, scores), 0, self.n_tiers - 1)
        counts = np.bincount(tiers, minlength=self.n_tiers)
        return counts / len(scores)

    def matchmaking_distortion(
        self,
        scores: np.ndarray,
        sigmas: np.ndarray,
        beta: float,
        max_pairs: int = 5000,
    ) -> float:
        from trueskill_core.matchmaking import match_quality
        from trueskill_core.model import GaussianBelief

        n = min(len(scores), 150)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(scores), size=n, replace=False)
        sub_scores = scores[idx]
        sub_sigmas = sigmas[idx]

        total_error = 0.0
        n_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                mu_i = sub_scores[i] + 3.0 * sub_sigmas[i]
                mu_j = sub_scores[j] + 3.0 * sub_sigmas[j]

                b_i = GaussianBelief(mu_i, sub_sigmas[i])
                b_j = GaussianBelief(mu_j, sub_sigmas[j])
                q_true = match_quality(b_i, b_j, beta)

                mu_iq = self.reconstruct(self.quantize(sub_scores[i])) + 3.0 * sub_sigmas[i]
                mu_jq = self.reconstruct(self.quantize(sub_scores[j])) + 3.0 * sub_sigmas[j]
                b_iq = GaussianBelief(mu_iq, sub_sigmas[i])
                b_jq = GaussianBelief(mu_jq, sub_sigmas[j])
                q_approx = match_quality(b_iq, b_jq, beta)

                total_error += (q_true - q_approx) ** 2
                n_pairs += 1

                if n_pairs >= max_pairs:
                    return total_error / n_pairs

        return total_error / max(n_pairs, 1)
