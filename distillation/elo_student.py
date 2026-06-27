import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Tuple


class EloStudent:
    def __init__(
        self,
        k_factor: float = 24.0,
        initial_rating: float = 1200.0,
        surprise_weight: float = 0.5,
    ):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.surprise_weight = surprise_weight
        self.ratings: Dict[str, float] = {}
        self.game_counts: Dict[str, int] = {}

    def get_rating(self, player_id: str) -> float:
        return self.ratings.get(player_id, self.initial_rating)

    def _expected_score(self, r1: float, r2: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))

    def _score_from_outcome(self, outcome: int) -> Tuple[float, float]:
        if outcome == 1:
            return 1.0, 0.0
        elif outcome == -1:
            return 0.0, 1.0
        return 0.5, 0.5

    def update_hard(self, player1: str, player2: str, outcome: int) -> None:
        r1 = self.get_rating(player1)
        r2 = self.get_rating(player2)
        e1 = self._expected_score(r1, r2)
        e2 = self._expected_score(r2, r1)
        s1, s2 = self._score_from_outcome(outcome)

        self.ratings[player1] = r1 + self.k_factor * (s1 - e1)
        self.ratings[player2] = r2 + self.k_factor * (s2 - e2)
        self.game_counts[player1] = self.game_counts.get(player1, 0) + 1
        self.game_counts[player2] = self.game_counts.get(player2, 0) + 1

    def update_soft(
        self,
        player1: str,
        player2: str,
        outcome: int,
        soft_label: Dict,
    ) -> None:
        r1 = self.get_rating(player1)
        r2 = self.get_rating(player2)
        e1 = self._expected_score(r1, r2)
        e2 = self._expected_score(r2, r1)
        s1, s2 = self._score_from_outcome(outcome)

        surprise = soft_label.get("surprise", 1.0)
        k_effective = self.k_factor * (1.0 + self.surprise_weight * np.tanh(surprise - 1.0))

        self.ratings[player1] = r1 + k_effective * (s1 - e1)
        self.ratings[player2] = r2 + k_effective * (s2 - e2)
        self.game_counts[player1] = self.game_counts.get(player1, 0) + 1
        self.game_counts[player2] = self.game_counts.get(player2, 0) + 1

    def leaderboard(self) -> List[Tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def rank_correlation(self, ground_truth: Dict[str, float]) -> float:
        common = [p for p in self.ratings if p in ground_truth]
        if len(common) < 2:
            return 0.0
        elo_vals = [self.ratings[p] for p in common]
        true_vals = [ground_truth[p] for p in common]
        rho, _ = spearmanr(elo_vals, true_vals)
        return float(rho)
