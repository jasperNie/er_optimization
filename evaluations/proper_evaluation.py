# Proper evaluation with train/test split
import random
import math
from classes import ERSimulation, Patient
from evolutionary_optimizer import EvolutionaryTriageOptimizer
from improved_evolutionary_optimizer import ImprovedEvolutionaryTriageOptimizer

def generate_arrivals(total_time, arrival_prob, seed):
    """Generate patient arrivals for a given scenario"""
    random.seed(seed)
    arrivals = []
    patient_id = 0
    timesteps_per_day = 96
    
    for t in range(total_time):
        base_prob = arrival_prob
        time_factor = 0.15 * (1 + math.sin(2 * math.pi * t / timesteps_per_day))
        burst = 0.15 if random.random() < 0.05 else 0
        prob = min(1.0, max(0.0, base_prob + time_factor + burst))
        
        if random.random() < prob:
            severity = random.choices([1,2,3,4,5], weights=[0.1,0.15,0.25,0.25,0.25])[0]
            det_base = 0.1 + 0.15 * (severity-1)
            deterioration = round(random.uniform(det_base, min(0.6, det_base+0.15)), 2)
            treat_base = 5 + 2 * severity
            treatment_time = random.randint(treat_base, treat_base+5)
            presenting = random.choice(['chest_pain', 'abdominal_pain', 'shortness_of_breath', 'minor_injury', 'headache'])
            vitals = {
                'hr': random.randint(50, 130),
                'rr': random.randint(12, 30),
                'spo2': random.randint(88, 100),
                'bp_sys': random.randint(90, 160)
            }
            expected_resources = random.choice([0, 1, 2, 3])
            p = Patient(patient_id, severity, deterioration, treatment_time)
            patient_id += 1
            p.presenting = presenting
            p.vitals = vitals
            p.expected_resources = expected_resources
            arrivals.append(p)
        else:
            arrivals.append(None)
    
    return arrivals

def mts_policy(patient):
    if patient is None:
        return 0
    v = getattr(patient, 'vitals', {})
    if v.get('spo2', 100) < 85 or v.get('hr', 0) > 140 or v.get('bp_sys', 0) < 70:
        return 5
    if hasattr(patient, 'presenting') and patient.presenting in ('shortness_of_breath', 'chest_pain'):
        if v.get('spo2', 100) < 92 or v.get('hr', 0) > 110:
            return 4
        return 3
    if hasattr(patient, 'presenting') and patient.presenting in ('abdominal_pain',):
        return 3
    if hasattr(patient, 'presenting') and patient.presenting in ('minor_injury', 'headache'):
        return 2
    return 1

def esi_policy(patient):
    if patient is None:
        return 0
    v = getattr(patient, 'vitals', {})
    if v.get('spo2', 100) < 85 or v.get('hr', 0) > 140 or v.get('bp_sys', 0) < 70:
        return 5
    if v.get('spo2', 100) < 92 or v.get('hr', 0) > 120 or v.get('rr', 0) > 25:
        return 4
    resources = getattr(patient, 'expected_resources', 1)
    if resources == 0:
        return 1
    if resources == 1:
        return 2
    if resources >= 2:
        if v.get('hr', 0) > 120 or v.get('rr', 0) > 24 or v.get('spo2', 100) < 92:
            return 4
        return 3

def evaluate_policy_on_scenarios(policy, test_seeds, num_nurses=4, total_time=96, arrival_prob=0.3):
    """Evaluate a policy on multiple test scenarios"""
    results = []
    
    for seed in test_seeds:
        arrivals = generate_arrivals(total_time, arrival_prob, seed)
        sim = ERSimulation(
            num_nurses=num_nurses, 
            total_time=total_time, 
            arrival_prob=0,  # Use pre-generated arrivals
            triage_policy=policy, 
            verbose=False, 
            seed=seed, 
            use_shifts=True
        )
        sim.patient_arrivals = arrivals
        metrics = sim.run()
        results.append(metrics)
    
    return results

def main():
    print("=== PROPER TRAIN/TEST EVALUATION ===")
    
    # Training seeds (used during optimization)
    train_seeds = [1000, 1001, 1002]
    
    # Testing seeds (completely separate, never seen during training)
    test_seeds = [2000, 2001, 2002, 2003, 2004]
    
    print(f"Training on seeds: {train_seeds}")
    print(f"Testing on seeds: {test_seeds}")
    print()
    
    # Train Original Algorithm
    print("Training Original Algorithm...")
    original_optimizer = EvolutionaryTriageOptimizer(
        num_generations=100, 
        population_size=100, 
        num_nurses=3, 
        total_time=96, 
        arrival_prob=0.3, 
        seed=train_seeds[0]  # Use first training seed
    )
    original_policy = original_optimizer.run("train_original_log.txt")
    
    # Train Improved Algorithm  
    print("Training Improved Algorithm...")
    improved_optimizer = ImprovedEvolutionaryTriageOptimizer(
        num_generations=100, 
        population_size=100, 
        num_nurses=3, 
        total_time=96, 
        arrival_prob=0.3, 
        seed=train_seeds[0]  # Use same training seed for fair comparison
    )
    improved_policy = improved_optimizer.run("train_improved_log.txt")
    
    print("\n=== TESTING ON UNSEEN DATA ===")
    
    # Test all policies on unseen scenarios
    policies = {
        'Original Optimized': original_policy,
        'Improved Optimized': improved_policy, 
        'MTS': mts_policy,
        'ESI': esi_policy
    }
    
    all_results = {}
    
    for name, policy in policies.items():
        print(f"\nTesting {name}...")
        results = evaluate_policy_on_scenarios(policy, test_seeds)
        all_results[name] = results
        
        # Calculate average performance across test scenarios
        avg_wait_times = [r['avg_wait'] for r in results if r['avg_wait'] is not None]
        avg_weighted_waits = [r['avg_weighted_wait'] for r in results if r['avg_weighted_wait'] is not None]
        completed_counts = [r['completed'] for r in results]
        
        print(f"  Average completed patients: {sum(completed_counts)/len(completed_counts):.1f}")
        if avg_wait_times:
            print(f"  Average wait time: {sum(avg_wait_times)/len(avg_wait_times):.2f}")
        if avg_weighted_waits:
            print(f"  Average weighted wait: {sum(avg_weighted_waits)/len(avg_weighted_waits):.2f}")
    
    print("\n=== SUMMARY ===")
    for name in policies.keys():
        results = all_results[name]
        avg_weighted_waits = [r['avg_weighted_wait'] for r in results if r['avg_weighted_wait'] is not None]
        if avg_weighted_waits:
            avg_performance = sum(avg_weighted_waits) / len(avg_weighted_waits)
            print(f"{name:20}: {avg_performance:.2f} avg weighted wait")

if __name__ == "__main__":
    main()