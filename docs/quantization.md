# Quantization in TrueSkill

## What Is Already Quantized

TrueSkill itself is a quantization of the true posterior. The EP moment
matching step is a projection onto the nearest Gaussian in KL divergence.
Further quantizing the two-float belief state is not meaningful.

## Tier Quantization (Scalar Quantization)

The continuous conservative estimate μ − 3σ is mapped to a discrete tier.
This is scalar quantization. The Lloyd-Max algorithm places boundaries
optimally for a given population distribution, minimising mean squared
distortion.

The distortion metric that matters for the system is not MSE on raw scores
but MSE on matchmaking quality values. A tier system that clusters similar
players together loses less matchmaking accuracy than one that draws uniform
interval boundaries.

The `TierQuantizer.matchmaking_distortion` method computes this directly
by comparing match quality values before and after tier-based reconstruction.

## Approximate Matchmaking (Search Quantization)

Finding the best match for a player in a pool of N candidates is O(N) per
query. Across M simultaneous waiting players this becomes O(NM).

The KD-Tree approach embeds each player as a 2D feature vector [μ, σ_eff]
where σ_eff = √(σ² + β²) captures the effective uncertainty relevant to
match quality. Approximate nearest neighbours in this space are then
re-ranked by the exact q_draw formula.

This reduces query time from O(N) to O(log N) for the search phase, with a
small and measurable quality loss tracked by `matchmaking_quality_loss`.

## β² as Temperature

The performance variance β² controls how soft the win probability is:

    P(p₁ > p₂ | s₁, s₂) = Φ((s₁ − s₂) / √(2β²))

High β² → soft, near-uniform win probabilities.
Low β² → sharp, near-deterministic win probabilities.

This is the exact analog of temperature scaling in neural network
classification. It governs information content per game.
