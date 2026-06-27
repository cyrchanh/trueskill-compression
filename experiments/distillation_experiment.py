import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from trueskill_core.system import TrueSkillSystem
from distillation.soft_labels import TrueSkillTeacher
from distillation.elo_student import EloStudent
from experiments.generate_data import generate_player_skills, generate_game_log


def run_distillation_experiment(
    n_players: int = 200,
    n_games: int = 4000,
    checkpoint_interval: int = 100,
    save_path: str = "results_distillation.png",
) -> None:
    rng = np.random.default_rng(42)
    true_skills = generate_player_skills(n_players, rng=rng)
    games = generate_game_log(true_skills, n_games, rng=rng)

    ts = TrueSkillSystem()
    elo_hard = EloStudent()
    elo_soft = EloStudent()
    teacher = TrueSkillTeacher(ts)

    checkpoints = list(range(checkpoint_interval, n_games + 1, checkpoint_interval))
    ts_corrs, hard_corrs, soft_corrs = [], [], []

    for i, (p1, p2, outcome) in enumerate(games):
        ts.update_1v1(p1, p2, outcome)
        elo_hard.update_hard(p1, p2, outcome)
        label = teacher.soft_label(p1, p2, outcome)
        elo_soft.update_soft(p1, p2, outcome, label)

        if (i + 1) in checkpoints:
            common = [p for p in ts.beliefs if p in true_skills]
            ts_mu = [ts.beliefs[p].mu for p in common]
            hard_r = [elo_hard.get_rating(p) for p in common]
            soft_r = [elo_soft.get_rating(p) for p in common]
            true_v = [true_skills[p] for p in common]

            ts_corrs.append(spearmanr(ts_mu, true_v).statistic)
            hard_corrs.append(spearmanr(hard_r, true_v).statistic)
            soft_corrs.append(spearmanr(soft_r, true_v).statistic)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(checkpoints, ts_corrs, label="TrueSkill (Teacher)", color="navy", linewidth=2)
    axes[0].plot(checkpoints, soft_corrs, label="Elo + Soft Labels", color="steelblue", linewidth=2, linestyle="--")
    axes[0].plot(checkpoints, hard_corrs, label="Elo Hard Labels", color="coral", linewidth=2, linestyle=":")
    axes[0].set_xlabel("Games Processed")
    axes[0].set_ylabel("Spearman ρ with True Skills")
    axes[0].set_title("Ranking Accuracy: Teacher vs Student")
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    gap_soft = np.array(ts_corrs) - np.array(soft_corrs)
    gap_hard = np.array(ts_corrs) - np.array(hard_corrs)
    axes[1].plot(checkpoints, gap_hard, label="Teacher − Hard Elo", color="coral", linewidth=2)
    axes[1].plot(checkpoints, gap_soft, label="Teacher − Soft Elo", color="steelblue", linewidth=2, linestyle="--")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Games Processed")
    axes[1].set_ylabel("Correlation Gap")
    axes[1].set_title("Distillation Gap Over Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    run_distillation_experiment()
