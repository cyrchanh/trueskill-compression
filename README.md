# TrueSkill Compression

A study of quantization and knowledge distillation concepts applied to the
TrueSkill Bayesian skill rating system (Herbrich, Minka, Graepel 2006).

## Key Thesis

TrueSkill is already an extremely lightweight system — two floats per player,
one EP pass per game. The compression targets addressed here are therefore the
genuine bottlenecks: the O(N²) matchmaking search, the scalar quantization of
displayed skill into discrete tiers, and the teacher–student transfer from
TrueSkill to a simpler Elo-based system.

## Repository Structure

```
trueskill_core/     Core EP inference, belief model, matchmaking quality
quantization/       Tier (Lloyd-Max) quantization, approximate matchmaking
distillation/       Soft label generation, Elo student training
experiments/        Reproducible experiment scripts
docs/               Theoretical background and design notes
```

## Installation

```bash
pip install -e .
```

## Running All Experiments

```bash
python run_all.py
```

Or individually:

```bash
python -m experiments.tier_experiment
python -m experiments.distillation_experiment
python -m experiments.matchmaking_experiment
```

## Experiment Outputs

| Script | Output File | What It Shows |
|---|---|---|
| `tier_experiment.py` | `results_tier.png` | Distortion vs tier count (Lloyd-Max) |
| `distillation_experiment.py` | `results_distillation.png` | TrueSkill teacher vs Elo student convergence |
| `matchmaking_experiment.py` | `results_matchmaking.png` | Naive O(N) vs KD-Tree query time and quality loss |

## Core Parameters

| Symbol | Default | Role |
|---|---|---|
| μ₀ | 25.0 | Prior mean skill |
| σ₀ | 25/3 ≈ 8.33 | Prior skill uncertainty |
| β | 25/6 ≈ 4.17 | Performance noise |
| γ | 25/300 ≈ 0.08 | Skill dynamics drift |
| ε | 0.0 | Draw margin |

## References

## References

Herbrich, R., Minka, T., Graepel, T. (2006).
*TrueSkill: A Bayesian Skill Rating System.*
Advances in Neural Information Processing Systems 20.
[[Microsoft Research](https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system-2/)]
[[NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2006)]
