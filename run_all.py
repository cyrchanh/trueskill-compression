from experiments.tier_experiment import run_tier_experiment
from experiments.distillation_experiment import run_distillation_experiment
from experiments.matchmaking_experiment import run_matchmaking_experiment

if __name__ == "__main__":
    print("=== Tier Quantization ===")
    run_tier_experiment()

    print("\n=== Knowledge Distillation ===")
    run_distillation_experiment()

    print("\n=== Matchmaking Approximation ===")
    run_matchmaking_experiment()
