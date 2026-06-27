import numpy as np
import matplotlib.pyplot as plt
from typing import List
from trueskill_core.system import TrueSkillSystem
from trueskill_core.model import BETA
from quantization.tier import TierQuantizer
from experiments.generate_data import generate_player_skills, generate_game_log


def run_tier_experiment(
    n_players: int = 500,
    n_games: int = 5000,
    tier_counts: List[int] = None,
    save_path: str = "results_tier.png",
) -> None:
    if tier_counts is None:
        tier_counts = [3, 5, 7, 10, 15, 20, 30]

    rng = np.random.default_rng(0)
    skills = generate_player_skills(n_players, rng=rng)
    games = generate_game_log(skills, n_games, rng=rng)

    system = TrueSkillSystem()
    for p1, p2, outcome in games:
        system.update_1v1(p1, p2, outcome)

    player_ids = list(system.beliefs.keys())
    conservative_scores = np.array([system.beliefs[pid].conservative for pid in player_ids])
    sigmas = np.array([system.beliefs[pid].sigma for pid in player_ids])

    distortions = []
    mm_distortions = []

    for n_tiers in tier_counts:
        q = TierQuantizer.fit(conservative_scores, n_tiers)
        distortions.append(q.distortion(conservative_scores))
        mm_distortions.append(q.matchmaking_distortion(conservative_scores, sigmas, BETA))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(tier_counts, distortions, "o-", color="steelblue", linewidth=2, markersize=6)
    axes[0].set_xlabel("Number of Tiers")
    axes[0].set_ylabel("Mean Squared Distortion")
    axes[0].set_title("Skill Score Distortion vs Tier Count")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tier_counts, mm_distortions, "s-", color="coral", linewidth=2, markersize=6)
    axes[1].set_xlabel("Number of Tiers")
    axes[1].set_ylabel("Matchmaking Quality Loss (MSE)")
    axes[1].set_title("Matchmaking Distortion vs Tier Count")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    best_q = TierQuantizer.fit(conservative_scores, 10)
    tier_assignments = np.array([best_q.quantize(s) for s in conservative_scores])
    balance = best_q.population_balance(conservative_scores)
    axes[2].bar(range(10), balance * 100, color="steelblue", edgecolor="white", alpha=0.85)
    axes[2].set_xlabel("Tier")
    axes[2].set_ylabel("Population Share (%)")
    axes[2].set_title("Player Distribution Across 10 Tiers (Lloyd-Max)")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    run_tier_experiment()
