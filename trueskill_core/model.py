import numpy as np
from dataclasses import dataclass

MU_0 = 25.0
SIGMA_0 = 25.0 / 3.0
BETA = 25.0 / 6.0
GAMMA = 25.0 / 300.0


@dataclass
class GaussianBelief:
    mu: float = MU_0
    sigma: float = SIGMA_0

    @property
    def pi(self) -> float:
        return self.sigma ** -2

    @property
    def tau(self) -> float:
        return self.pi * self.mu

    @property
    def conservative(self) -> float:
        return self.mu - 3.0 * self.sigma

    def with_dynamics(self, gamma: float = GAMMA) -> "GaussianBelief":
        return GaussianBelief(mu=self.mu, sigma=np.sqrt(self.sigma ** 2 + gamma ** 2))

    @classmethod
    def from_canonical(cls, pi: float, tau: float) -> "GaussianBelief":
        pi = max(pi, 1e-10)
        return cls(mu=tau / pi, sigma=pi ** -0.5)

    def __mul__(self, other: "GaussianBelief") -> "GaussianBelief":
        return GaussianBelief.from_canonical(self.pi + other.pi, self.tau + other.tau)

    def __truediv__(self, other: "GaussianBelief") -> "GaussianBelief":
        return GaussianBelief.from_canonical(self.pi - other.pi, self.tau - other.tau)

    def __repr__(self) -> str:
        return f"GaussianBelief(mu={self.mu:.4f}, sigma={self.sigma:.4f})"
