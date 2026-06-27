import numpy as np
from typing import Dict, List, Tuple
from .model import GaussianBelief, MU_0, SIGMA_0, BETA, GAMMA
from .ep import two_player_update


class TrueSkillSystem:
    def __init__(
        self,
        mu_0: float = MU_0,
        sigma_0: float = SIGMA_0,
        beta: float = BETA,
        gamma: float = GAMMA,
        epsilon: float = 0.0,
    ):
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.beliefs: Dict[str, GaussianBelief] = {}
        self.game_history: List[Tuple[str, str, int]] = []

    def get_belief(self, player_id: str) -> GaussianBelief:
        if player_id not in self.beliefs:
            self.beliefs[player_id] = GaussianBelief(self.mu_0, self.sigma_0)
        return self.beliefs[player_id]

    def update_1v1(self, player1: str, player2: str, outcome: int) -> None:
        b1 = self.get_belief(player1).with_dynamics(self.gamma)
        b2 = self.get_belief(player2).with_dynamics(self.gamma)
        b1_new, b2_new = two_player_update(b1, b2, outcome, self.beta, self.epsilon)
        self.beliefs[player1] = b1_new
        self.beliefs[player2] = b2_new
        self.game_history.append((player1, player2, outcome))

    def conservative_skill(self, player_id: str) -> float:
        return self.get_belief(player_id).conservative

    def leaderboard(self) -> List[Tuple[str, float, float, float]]:
        entries = [
            (pid, b.mu, b.sigma, b.conservative)
            for pid, b in self.beliefs.items()
        ]
        return sorted(entries, key=lambda x: x[3], reverse=True)

    def win_probability(self, player1: str, player2: str) -> float:
        from .matchmaking import win_probability as _wp
        return _wp(self.get_belief(player1), self.get_belief(player2), self.beta)
