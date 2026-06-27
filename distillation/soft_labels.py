import numpy as np
from typing import Dict, List, Tuple
from trueskill_core.model import GaussianBelief, BETA
from trueskill_core.ep import two_player_update
from trueskill_core.matchmaking import win_probability


class TrueSkillTeacher:
    def __init__(self, system, beta: float = BETA):
        self.system = system
        self.beta = beta

    def soft_label(self, player1: str, player2: str, outcome: int) -> Dict[str, float]:
        b1 = self.system.get_belief(player1)
        b2 = self.system.get_belief(player2)
        b1_post, b2_post = two_player_update(b1, b2, outcome, self.beta, self.system.epsilon)

        p_win = win_probability(b1, b2, self.beta)

        if outcome == 1:
            surprise = -np.log(max(p_win, 1e-10))
        elif outcome == -1:
            surprise = -np.log(max(1.0 - p_win, 1e-10))
        else:
            surprise = -np.log(max(2.0 * min(p_win, 1.0 - p_win), 1e-10))

        return {
            "delta_mu1": b1_post.mu - b1.mu,
            "delta_mu2": b2_post.mu - b2.mu,
            "delta_sigma1": b1_post.sigma - b1.sigma,
            "delta_sigma2": b2_post.sigma - b2.sigma,
            "win_prob_before": p_win,
            "surprise": float(surprise),
            "sigma1_before": b1.sigma,
            "sigma2_before": b2.sigma,
        }

    def generate_soft_dataset(
        self,
        games: List[Tuple[str, str, int]],
    ) -> List[Dict]:
        records = []
        for p1, p2, outcome in games:
            label = self.soft_label(p1, p2, outcome)
            label["player1"] = p1
            label["player2"] = p2
            label["outcome"] = outcome
            records.append(label)
        return records

    def temperature_scaled_win_prob(
        self,
        player1: str,
        player2: str,
        temperature: float = 1.0,
    ) -> float:
        b1 = self.system.get_belief(player1)
        b2 = self.system.get_belief(player2)
        effective_beta = self.beta * np.sqrt(temperature)
        return win_probability(b1, b2, effective_beta)
