import numpy as np
from typing import Dict, List, Tuple
from sklearn.neighbors import KDTree
from trueskill_core.model import GaussianBelief, BETA
from trueskill_core.matchmaking import match_quality, naive_best_match


def _belief_to_feature(belief: GaussianBelief, beta: float) -> np.ndarray:
    return np.array([belief.mu, np.sqrt(belief.sigma ** 2 + beta ** 2)])


def build_index(
    beliefs: Dict[str, GaussianBelief],
    beta: float = BETA,
    leaf_size: int = 40,
) -> Tuple[KDTree, List[str]]:
    player_ids = list(beliefs.keys())
    features = np.array([_belief_to_feature(beliefs[pid], beta) for pid in player_ids])
    tree = KDTree(features, leaf_size=leaf_size)
    return tree, player_ids


def approx_best_match(
    target: str,
    beliefs: Dict[str, GaussianBelief],
    tree: KDTree,
    player_ids: List[str],
    beta: float = BETA,
    k: int = 20,
) -> Tuple[str, float]:
    b_target = beliefs[target]
    query = _belief_to_feature(b_target, beta).reshape(1, -1)
    k_actual = min(k + 1, len(player_ids))
    _, indices = tree.query(query, k=k_actual)

    best_opponent = None
    best_quality = -1.0

    for idx in indices[0]:
        candidate = player_ids[int(idx)]
        if candidate == target:
            continue
        q = match_quality(b_target, beliefs[candidate], beta)
        if q > best_quality:
            best_quality = q
            best_opponent = candidate

    return best_opponent, best_quality


def matchmaking_quality_loss(
    beliefs: Dict[str, GaussianBelief],
    beta: float = BETA,
    k: int = 20,
    sample_size: int = 100,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    player_ids = list(beliefs.keys())
    sample_ids = rng.choice(
        player_ids,
        size=min(sample_size, len(player_ids)),
        replace=False,
    ).tolist()

    tree, indexed_ids = build_index(beliefs, beta)

    naive_qualities = []
    approx_qualities = []

    for target in sample_ids:
        _, q_naive = naive_best_match(target, player_ids, beliefs, beta)
        _, q_approx = approx_best_match(target, beliefs, tree, indexed_ids, beta, k)
        naive_qualities.append(q_naive)
        approx_qualities.append(q_approx)

    naive_arr = np.array(naive_qualities)
    approx_arr = np.array(approx_qualities)

    return {
        "mean_naive_quality": float(np.mean(naive_arr)),
        "mean_approx_quality": float(np.mean(approx_arr)),
        "mean_quality_loss": float(np.mean(naive_arr - approx_arr)),
        "max_quality_loss": float(np.max(naive_arr - approx_arr)),
        "relative_loss_pct": float(100.0 * np.mean((naive_arr - approx_arr) / np.clip(naive_arr, 1e-10, None))),
    }
