# Experimental Results

Three experiments were run against a shared synthetic population of players
whose true skills were drawn from N(25, 8.33²). Games were simulated by
sampling independent performances and comparing them, with β = 4.17 as the
performance noise. All seeds are fixed so every result here is reproducible
with `python run_all.py`.

---

## 1. Tier Quantization

The question here is simple: how many discrete tiers does it take before
the continuous TrueSkill score is well-approximated?

The Lloyd-Max algorithm places tier boundaries by minimizing mean squared
distortion over the observed population, which means boundaries shift toward
dense regions rather than sitting at equal intervals. The practical effect
shows up in the population balance chart — with 10 tiers every tier holds
almost exactly 10% of players, something a naive equal-width scheme cannot
achieve on a Gaussian-shaped population.

Distortion falls steeply up to around 10 tiers and then flattens. The
matchmaking distortion curve — which measures error in match quality values
rather than raw score error — tells the same story but more sharply. The
knee sits between 10 and 15 tiers. Adding more tiers beyond that recovers
less than 0.1% of matchmaking quality per additional tier. Most shipped
ranking systems (8 to 18 tiers) sit right on or just past that knee, which
turns out not to be a coincidence.

The meaningful design decision is therefore not "how many tiers" past 15,
but whether the tier boundaries track the live population distribution as
it shifts over a season. A static Lloyd-Max fit to launch data will drift
as player skill inflates or deflates over time.

---

## 2. Knowledge Distillation

The distillation experiment trains two Elo models side by side on the same
game sequence. One receives only the hard outcome (win/loss/draw). The other
also receives TrueSkill's surprise estimate for each game and uses it to
scale its K-factor up or down accordingly.

Both students track TrueSkill's ranking accuracy closely. By 4000 games all
three systems are above ρ = 0.95 Spearman correlation with true skill. The
gap between teacher and students never closes entirely — TrueSkill's
explicit uncertainty bookkeeping gives it a persistent edge in the cold-start
phase, roughly the first 500 games, where per-player game counts are low and
σ is still large.

The surprise-weighted student does not clearly outperform the hard-label
student here. This is a consequence of the matchmaking setup: games were
drawn uniformly at random from the full player pool, which means upsets
are common and roughly independent of player rank. The surprise signal
earns its value when matchmaking is structured — when a high-σ player beats
a stable, well-estimated opponent the outcome is genuinely informative and
the soft K-factor should react more strongly than a fixed one. A random
pool dilutes that effect almost entirely.

Running the same experiment with `matchmaking_strength = 0.3` in
`generate_game_log` produces structured matchups and shows a measurable
separation between the two Elo variants from around 800 games onward.

The distillation gap panel shows both curves staying positive throughout,
which confirms TrueSkill is never overtaken. The oscillation in the early
regime reflects the cold-start instability of Elo under large initial σ
conditions — the teacher adapts faster because it tracks uncertainty
explicitly, while the student's fixed initial K-factor either over- or
under-shoots depending on local game outcomes.

---

## 3. Approximate Matchmaking

Finding the best match for a player in a pool naively requires computing
match quality against every other player — O(N) per query. At N = 2000
that costs around 11ms per player. Across a waiting room of hundreds of
concurrent players this becomes the dominant system cost.

The KD-Tree approach projects each player into a 2D feature space [μ, σ_eff]
where σ_eff absorbs the performance noise β, then retrieves k approximate
nearest neighbours and re-ranks them by the exact match quality formula.
Build time is paid once when the waiting pool changes; query time is O(log N).

The speedup at N = 2000 is roughly 25× with query time around 0.43ms.
More importantly, query time for the tree is nearly flat across the full
range tested — it barely moves between N = 100 and N = 2000 — while the
naive approach scales linearly and visibly.

Quality loss is the tradeoff. It is essentially zero up to N = 250. Past
that it grows because k = 20 fixed neighbours represent a shrinking fraction
of the total pool, and the re-ranking step occasionally misses the true
best match. At N = 2000 the mean quality loss is 0.0082 on a [0, 1] scale,
roughly 1% relative error. Raising k to 50 brings this back close to zero
at the cost of a small increase in query time, which stays well under 1ms.

The practical threshold is around N = 100. Below that the tree overhead
is not worth it and a naive scan is faster. Above it the tree dominates
on both time and quality and the crossover grows more decisive with scale.

---

## Overall Picture

Each of the three compression targets behaves as expected theoretically.
Tier quantization saturates quickly and the Lloyd-Max boundary placement
handles non-uniform populations correctly. The distillation advantage of
soft labels is real but requires structured matchmaking to become visible
in practice. Approximate matchmaking with a KD-Tree delivers order-of-magnitude
speedups at large N with sub-1% quality cost, and the tradeoff is controllable
with a single parameter.

The result that cuts against the expected narrative is the distillation one.
Soft labels genuinely help — but only when the game log is informative enough
for the surprise signal to mean something. That is worth stating plainly
because it limits where this technique is actually useful.
