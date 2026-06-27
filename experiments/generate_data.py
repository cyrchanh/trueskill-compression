import numpy as np
from typing import Dict, List, Tuple


def generate_player_skills(
    n_players: int,
    mu_true: float = 25.0,
    sigma_true: float = 8.33,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    if rng is None:
        rng = np.random.default_rng(42)
    skills = rng.normal(mu_true, sigma_true, n_players)
    return {f"player_{i:04d}": float(s) for i, s in enumerate(skills)}


def simulate_outcome(
    skill1: float,
    skill2: float,
    beta: float = 4.167,
    draw_margin: float = 0.0,
    rng: np.random.Generator = None,
) -> int:
    if rng is None:
        rng = np.random.default_rng()
    diff = rng.normal(skill1, beta) - rng.normal(skill2, beta)
    if diff > draw_margin:
        return 1
    elif diff < -draw_margin:
        return -1
    return 0


def generate_game_log(
    skills: Dict[str, float],
    n_games: int,
    beta: float = 4.167,
    draw_margin: float = 0.0,
    matchmaking_strength: float = 0.0,
    rng: np.random.Generator = None,
) -> List[Tuple[str, str, int]]:
    if rng is None:
        rng = np.random.default_rng(42)

    player_ids = list(skills.keys())
    games = []

    for _ in range(n_games):
        if matchmaking_strength > 0.0 and len(player_ids) > 1:
            idx1 = int(rng.integers(0, len(player_ids)))
            p1 = player_ids[idx1]
            others = [p for p in player_ids if p != p1]
            diffs = np.array([abs(skills[p1] - skills[p]) for p in others])
            probs = np.exp(-matchmaking_strength * diffs)
            probs /= probs.sum()
            p2 = others[int(rng.choice(len(others), p=probs))]
        else:
            idxs = rng.choice(len(player_ids), size=2, replace=False)
            p1, p2 = player_ids[int(idxs[0])], player_ids[int(idxs[1])]

        outcome = simulate_outcome(skills[p1], skills[p2], beta, draw_margin, rng)
        games.append((p1, p2, outcome))

    return games
