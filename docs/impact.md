# Impact on Video Game Systems

This document translates the three experimental findings into concrete
consequences for shipped game systems. The study is synthetic and small,
so claims are scoped to what the data actually supports rather than what
the theory alone might suggest.

---

## Background

Skill rating in competitive games has two jobs that pull in opposite
directions. The first is inference — estimating each player's true ability
as accurately as possible from a noisy sequence of game outcomes. The second
is operation — running that inference cheaply enough, and presenting its
output simply enough, that it can function inside a live game service
handling millions of concurrent players.

TrueSkill handles inference well. It treats skill as a Gaussian random
variable, updates beliefs through Expectation Propagation after each game,
and carries explicit uncertainty estimates that govern how aggressively
ratings move. The operational side is where pressure accumulates. Displaying
a continuous float to players is impractical, so it gets bucketed into tiers.
Finding good matches in a large waiting pool requires comparing every
candidate, which gets expensive. Running the full EP update on constrained
hardware is sometimes not feasible at all.

The three experiments in this study each address one of those operational
pressure points directly.

---

## 1. Tier Design and Ranked Population Health

### The Finding

The `results_tier.png` shows the result most directly: with 10 tiers fitted by the
Lloyd-Max algorithm, each tier holds almost exactly 10% of the player
population. This does not happen automatically. Conservative skill scores
follow a roughly Gaussian distribution — the majority of players cluster
near the mean, with thin tails at either extreme. Equal-width tier boundaries
cut that distribution into intervals of the same size on the skill axis, not
the same number of players, which means two or three central tiers absorb
most of the population while the outer tiers sit nearly empty.

Lloyd-Max boundaries move inward toward the dense center of the distribution,
making middle-tier intervals narrower and outer-tier intervals wider until each
tier captures the same share of players. The flat bars in the population chart
are the signature of this compression.

The first two plots show matchmaking quality loss against tier count, both on
a log scale. The visual impression of a uniformly declining curve is an
artifact of the log axis. In absolute terms the marginal quality gain per
additional tier drops sharply past 10 tiers. Moving from 3 to 10 tiers
recovers roughly 93% of the total possible quality gain. Moving from 10 to
30 tiers recovers most of the remaining 7%, spread across twenty additional
tiers. The practical knee is at 10 to 15 tiers — past that, adding tiers
costs design and engineering complexity without returning meaningful
matchmaking improvement.

The inflation implication follows from the same logic applied to a moving
population. At launch, Lloyd-Max boundaries are fitted to the current player
distribution and tiers are balanced. As a season progresses, casual players
churn out and the remaining population becomes more experienced — the skill
distribution shifts upward. Boundaries set at launch now cut the shifted
distribution at the wrong positions, concentrating players in upper tiers
not because their skill improved but because the reference point moved.
This dynamic was not directly simulated in the experiment, but it is a
direct consequence of using static boundaries on a non-stationary population.
The input required to re-fit boundaries — the current distribution of
conservative scores — is always available to the rating server.

### What This Means for Games

The two variables game designers argue about most in ranked system design
are the number of tiers and where the boundaries between them sit. The
experiment suggests the boundary placement question matters considerably
more than the tier count question, as long as count is past the knee.

A system with 10 well-placed boundaries and a system with 18 well-placed
boundaries produce nearly indistinguishable matchmaking quality. A system
with 10 poorly-placed boundaries — for example boundaries drawn at equal
skill intervals on a skewed or inflated population — produces significantly
worse player distribution and worse matchmaking, because most of the
population falls into two or three central tiers while the rest sit empty.

This has a direct bearing on tier inflation, which is one of the most
common complaints in live ranked games. Inflation typically occurs when
the player skill distribution shifts over a season — skill clusters upward
as casual players churn and the remaining population becomes more
experienced — but tier boundaries stay fixed at their original positions.
Players accumulate in the upper tiers not because their skill improved but
because the boundaries are no longer calibrated to where the population
actually sits.

Lloyd-Max boundaries fitted at launch will drift in exactly this way. No
major shipped ranking system currently re-fits its tier boundaries
dynamically against the live population during a season. This is a direct
engineering gap the result points to, and it is a gap that is technically
straightforward to close — the algorithm is inexpensive and the input data
(the distribution of current conservative scores) is always available.

### Practical Recommendation

Fit tier boundaries against the live skill distribution at regular intervals
during a season — monthly at minimum, or triggered when the mean conservative
score shifts by more than a threshold. Boundaries should be treated as a
parameter of the running system, not a design constant. Player-facing tier
names can stay fixed while the underlying boundaries move, which avoids
communicating the change as a promotion or demotion event.

---

## 2. Matchmaking Server Cost at Scale

### The Finding

Naive best-match search is O(N) per player query. A KD-Tree built over
a 2D projection of each player's belief state reduces this to O(log N)
for the candidate retrieval step, with a small re-ranking pass over the
k nearest neighbours using the exact match quality formula.

The measured speedups against pool size were:

| Pool Size | Naive (ms) | KD-Tree (ms) | Speedup | Quality Loss |
|---|---|---|---|---|
| 100 | 0.50 | 0.26 | 1.9× | ~0% |
| 250 | 1.57 | 0.41 | 3.8× | ~0% |
| 500 | 1.96 | 0.39 | 5.0× | 0.06% |
| 1000 | 5.26 | 0.35 | 15× | 0.4% |
| 2000 | 11.08 | 0.43 | 25× | 0.8% |

KD-Tree query time is nearly flat across the full range tested. Naive
query time scales linearly and the gap widens with every increase in pool
size.

### What This Means for Games

A matchmaking service handling a peak waiting room of ten thousand concurrent
players using naive search is performing on the order of a hundred seconds
of compute per second. At that scale, matchmaking is not a background
process — it is the dominant server cost. The KD-Tree reduces that by an
order of magnitude at the pool sizes tested, and the scaling behavior
indicates the advantage grows further with larger pools.

The quality cost is the honest tradeoff. At N = 2000 the mean match quality
loss is 0.0082 on a zero-to-one scale, roughly 1% relative error. This is
almost certainly below player perception thresholds. Players notice large
skill gaps between opponents. The difference between a match quality of
0.85 and 0.842 is not something a human registers from a single game
experience. Over millions of matches it would appear as a marginal increase
in outcome variance, which is difficult to isolate from other sources of
variance in a live game environment.

The quality loss is controllable. Raising k from 20 to 50 brings the loss
back close to zero at N = 2000 while keeping query time well under 1ms.
The k parameter therefore serves as a tunable dial between speed and quality,
which is more operationally useful than the binary choice the naive approach
offers.

### Practical Caveats

Real matchmaking has constraints this study does not model. Regional server
topology splits the effective pool size, often dramatically. Queue time
tolerance forces quality tradeoffs when a player has been waiting too long
and the system accepts a worse match to avoid a worse wait. Party matching
adds combinatorial structure that the two-player KD-Tree projection does not
capture. These constraints require additional logic on top of the core search
and the study does not address them. The result is that the KD-Tree approach
solves the search subproblem efficiently but sits inside a larger system
where other bottlenecks exist.

---

## 3. Lightweight Rating for Mobile and Offline Play

### The Finding

An Elo student trained against a TrueSkill teacher achieves above ρ = 0.95
Spearman correlation with true player skill after 4000 games. The gap between
teacher and student never closes entirely but narrows steadily throughout
training. The student requires no Gaussian belief state, no EP message
passing, and no covariance tracking — its full state is a single float per
player updated by arithmetic after each game.

The surprise-weighted K-factor, which scales the student's learning rate
by the teacher's estimate of how informative each game was, does not
produce a clear advantage under random matchmaking. It does produce a
measurable advantage under structured matchmaking where high-uncertainty
players regularly face well-estimated opponents and upsets carry real signal.

### What This Means for Games

Mobile games and cross-platform games increasingly want to run partial
rating logic on-device. Reasons include offline play support, local
leaderboards that function without a server connection, latency reduction
for the rating update step, and reduced server dependency in markets with
unreliable connectivity.

TrueSkill is not a practical on-device model. It requires per-player
Gaussian state, scipy-level numerical routines for the V and W functions,
and careful handling of edge cases in the EP update. The Elo student has
none of these requirements. A rating table is a flat array of floats, and
an update is four arithmetic operations. It can run on any hardware that
runs the game.

The distillation pattern that makes this useful is:

```
Server (TrueSkill)
    ↓  trains student periodically on recent game log
    ↓  ships updated Elo weights and K-schedule to client
Client (Elo student)
    ↓  runs locally for offline sessions and fast local ranking
    ↓  syncs back to server when connected
```

The soft label component — specifically the surprise signal — is the part
of this pipeline that has no deployed equivalent in any known shipped
game. It represents the main speculative contribution of the study. The
mechanism is sound but its benefit is conditional on the game's matchmaking
producing structured game logs. A game with a large concurrent playerbase
and strong skill-based matchmaking will generate exactly that structure
as a byproduct of normal operation. A smaller or more casual game with
thin queues and random or social matchmaking probably will not, and a
fixed K-factor Elo is likely sufficient there.

### The Attribution Problem in Team Games

The study only models 1v1. Team games introduce a problem neither TrueSkill
nor Elo resolve cleanly: a player on a winning team may have performed
poorly, and a player on a losing team may have been the best performer on
the server. The rating update has no way to distinguish these cases from
the game outcome alone.

A richer teacher model trained on in-game performance telemetry — damage
dealt, objective contribution, survival time, position data — could produce
soft labels that carry individual attribution signal rather than just team
outcome. The distilled Elo student would then receive a more informative
surprise estimate, one that reflects the player's individual contribution
rather than just the team's result. This is a natural and practically
important extension of the distillation approach but it is a meaningfully
harder problem that requires telemetry infrastructure this study does not
address.

---

## Honest Scope of the Claims

The study is synthetic. Player skills were drawn from a Gaussian, games
were simulated from a performance noise model, and matchmaking was either
fully random or weakly structured. Real game populations are not Gaussian,
real game outcomes have confounds TrueSkill does not model, and real
matchmaking operates under queue time and regional constraints that change
the effective pool size dramatically from what is tested here.

The findings that transfer most directly are the structural ones:
Lloyd-Max boundary placement is better than equal-width placement regardless
of population shape, and KD-Tree search is faster than naive search
regardless of what quality metric is being optimized. These hold by the
nature of the algorithms, not by properties of the synthetic data.

The finding that transfers least directly is the distillation result,
because its practical value is conditional on matchmaking structure that
may or may not be present in a given game's operating conditions.

| Finding | Confidence | Deployment Readiness |
|---|---|---|
| Lloyd-Max boundaries reduce tier inflation | High | Ready — low engineering cost |
| Dynamic boundary refitting during seasons | High (theoretically) | Gap — not yet deployed anywhere |
| KD-Tree matchmaking at N > 100 | High | Ready — well-understood tradeoffs |
| Elo student viability on mobile | Moderate | Ready under structured matchmaking |
| Surprise-weighted K-factor advantage | Conditional | Needs validation on real game logs |
| Telemetry-informed distillation for teams | Speculative | Open research problem |

The two results with the clearest near-term impact are approximate
matchmaking and Lloyd-Max tier placement. Both are ready to deploy,
both have measurable and interpretable benefits, and neither requires
changes to the core rating model — they sit on top of whatever skill
estimation system is already running.
