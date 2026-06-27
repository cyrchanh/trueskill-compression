# Knowledge Distillation in TrueSkill

## The Hierarchy

    True Bayesian Posterior  (intractable teacher)
            ↓  EP + Gaussian approximation
    TrueSkill               (already the compressed model)
            ↓  soft label transfer
    Elo Student             (deployed on constrained systems)

## Soft Label Generation

For each game, TrueSkill computes a soft label containing:

- delta_mu1, delta_mu2: posterior mean shifts for each player
- delta_sigma1, delta_sigma2: uncertainty reductions
- win_prob_before: predicted win probability before the game
- surprise: negative log-probability of the observed outcome

The surprise signal is the critical piece. It tells the student how
informative this game was. A highly surprising outcome carries more
signal about true skill differences.

## Student Training

The Elo student receives the same hard outcome (win/loss/draw) but uses
TrueSkill's surprise estimate to modulate its K-factor:

    K_effective = K_base · (1 + α · tanh(surprise − 1))

When TrueSkill is surprised (surprise > 1 nat), the student learns faster.
When TrueSkill expects the outcome, the student's update is conservative.

This is a principled use of teacher soft information: the student's
learning rate is calibrated to the teacher's uncertainty, not a fixed
hyperparameter.

## Cross-Mode Transfer

TrueSkill parameters (β², ε, γ²) estimated from 1v1 data can be used as
warm-started priors for team game modes where individual attribution is
harder and data is noisier. The teacher (1v1 TrueSkill) constrains the
student (team TrueSkill) through shared hyperparameter priors.

## What Distillation Cannot Fix

TrueSkill's additive team performance model breaks when team synergy
matters. A neural teacher trained on raw telemetry could capture synergy
effects. TrueSkill would then serve as the deployable student — structured,
interpretable, and fast — distilled from a richer but expensive teacher.
