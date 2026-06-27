import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from trueskill_core.system import TrueSkillSystem
from trueskill_core.matchmaking import naive_best_match
from quantization.approx_matchmaking import build_index, approx_best_match, matchmaking_quality_loss
from experiments.generate_data import generate_player_skills, generate_game_log


def run_matchmaking_experiment(
    player_counts: List[int] = None,
    games_per_player: int = 10,
    save_path: str = "results_matchmaking.png",
) -> None:
    if player_counts is None:
        player_counts = [50, 100, 250, 500, 1000, 2000]

    rng = np.random.default_rng(7)
    naive_times, approx_times, quality_losses = [], [], []

    for n_players in player_counts:
        skills = generate_player_skills(n_players, rng=rng)
        games = generate_game_log(skills, n_players * games_per_player, rng=rng)

        system = TrueSkillSystem()
        for p1, p2, outcome in games:
            system.update_1v1(p1, p2, outcome)

        player_ids = list(system.beliefs.keys())
        n_sample = min(30, n_players)
        targets = player_ids[:n_sample]

        t0 = time.perf_counter()
        for tgt in targets:
            naive_best_match(tgt, player_ids, system.beliefs)
        naive_times.append((time.perf_counter() - t0) / n_sample)

        tree, indexed_ids = build_index(system.beliefs)
        t0 = time.perf_counter()
        for tgt in targets:
            approx_best_match(tgt, system.beliefs, tree, indexed_ids)
        approx_times.append((time.perf_counter() - t0) / n_sample)

        metrics = matchmaking_quality_loss(system.beliefs, sample_size=min(40, n_players))
        quality_losses.append(metrics["mean_quality_loss"])

        print(
            f"N={n_players:5d} | "
            f"Naive {naive_times[-1]*1000:.2f}ms | "
            f"Approx {approx_times[-1]*1000:.2f}ms | "
            f"ΔQ {quality_losses[-1]:.5f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(player_counts, [t * 1000 for t in naive_times], "o-", label="Naive O(N)", color="coral", linewidth=2)
    axes[0].plot(player_counts, [t * 1000 for t in approx_times], "s-", label="KD-Tree", color="steelblue", linewidth=2)
    axes[0].set_xlabel("Number of Players")
    axes[0].set_ylabel("Avg Query Time (ms)")
    axes[0].set_title("Matchmaking Search: Naive vs KD-Tree")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(player_counts, quality_losses, "o-", color="navy", linewidth=2)
    axes[1].set_xlabel("Number of Players")
    axes[1].set_ylabel("Mean Match Quality Loss")
    axes[1].set_title("Quality Cost of Approximate Matchmaking")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    run_matchmaking_experiment()
