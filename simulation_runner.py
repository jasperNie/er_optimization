import random
from classes import ERSimulation
from evolutionary_optimizer import EvolutionaryTriageOptimizer
from triage_policies import compare_policies, mts_policy, esi_policy


def run_evolutionary_optimization():
    """Run evolutionary algorithm to find optimal triage policy"""
    print("Starting Evolutionary Triage Optimization...")
    
    optimizer = EvolutionaryTriageOptimizer(
        num_generations=50,
        population_size=100,
        num_nurses=3,
        total_time=96,  # 24 hours simulation
        arrival_prob=0.5,
        seed=123
    )
    
    best_policy = optimizer.run()
    
    print(f"Best evolved triage policy: {best_policy}")
    
    # Test the best policy
    print("\nTesting the best evolved policy...")
    sim = ERSimulation(
        num_nurses=3,
        total_time=96,
        arrival_prob=0.5,
        triage_policy=best_policy,
        verbose=True,
        seed=123
    )
    
    result = sim.run()
    print(f"Final evolved policy results: {result}")
    
    return best_policy


def run_policy_comparison():
    """Compare MTS and ESI policies"""
    print("\nComparing MTS vs ESI policies...")
    
    results = compare_policies(
        num_nurses=3,
        total_time=96,  # 24 hours simulation
        arrival_prob=0.5,
        seed=123
    )
    
    print("\nPolicy Comparison Results:")
    for policy_name, result in results.items():
        print(f"{policy_name} Policy:")
        for key, value in result.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: None")
        print()
    
    return results


def run_single_simulation(policy_type="evolved", custom_policy=None):
    """Run a single simulation with specified policy"""
    if policy_type == "mts":
        policy = mts_policy
        policy_name = "MTS"
    elif policy_type == "esi":
        policy = esi_policy
        policy_name = "ESI"
    elif policy_type == "evolved":
        # Use a reasonable default evolved policy if none provided
        if custom_policy is None:
            policy = {
                'severity': 1.2,
                'deterioration': 1.5,
                'wait_time': 0.8
            }
        else:
            policy = custom_policy
        policy_name = "Evolved"
    else:
        raise ValueError("Policy type must be 'mts', 'esi', or 'evolved'")
    
    print(f"Running single simulation with {policy_name} policy...")
    
    sim = ERSimulation(
        num_nurses=3,
        total_time=96,  # 24 hours simulation
        arrival_prob=0.5,
        triage_policy=policy,
        verbose=True,
        seed=123
    )
    
    result = sim.run()
    print(f"\n{policy_name} Policy Results: {result}")
    
    return result


if __name__ == "__main__":
    # Example usage - you can comment/uncomment what you want to run
    
    # Run evolutionary optimization
    best_policy = run_evolutionary_optimization()
    
    # Run policy comparison
    # comparison_results = run_policy_comparison()
    
    # Run single simulation with evolved policy
    # single_result = run_single_simulation("evolved", best_policy)