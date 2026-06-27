# Theoretical Background

## The Probabilistic Model

Each player i maintains a Gaussian belief over their latent skill:

    s_i ~ N(μ_i, σ_i²)

At game time, player i draws a performance:

    p_i ~ N(s_i, β²)

The team performance is the sum of its members' performances. The game
outcome is determined by the ordering of team performances against a draw
margin ε.

## Why EP and Not Exact Inference

The ordering constraints introduce truncations that make the posterior
non-Gaussian. Expectation Propagation approximates the non-Gaussian messages
from the win/draw factors as Gaussians via moment matching, which minimises
KL divergence. The graph is acyclic so a single forward-backward sweep
suffices in the two-player case.

## The V and W Functions

V is the additive mean correction from a truncated Gaussian observation.
W is the multiplicative variance reduction. Together they encode how
surprising the observed outcome was and how much information it provides.

For a win observation (d > ε):

    V(t, ε) = φ(t − ε) / Φ(t − ε)
    W(t, ε) = V(t, ε) · (V(t, ε) + t − ε)

For a draw observation (|d| ≤ ε):

    V(t, ε) = [φ(−ε − t) − φ(ε − t)] / [Φ(ε − t) − Φ(−ε − t)]
    W(t, ε) = V²(t, ε) + [(ε − t)φ(ε − t) + (ε + t)φ(ε + t)] / [Φ(ε − t) − Φ(−ε − t)]

## Displayed Skill and Matchmaking Quality

Displayed skill is the conservative estimate μ_i − 3σ_i, the approximate
1st percentile of the belief. The matchmaking quality between two players is:

    q(β², μ_i, μ_j, σ_i, σ_j) = √(2β² / (2β² + σ_i² + σ_j²))
                                  · exp(−(μ_i − μ_j)² / (2(2β² + σ_i² + σ_j²)))

This penalises both skill difference and skill uncertainty simultaneously.
