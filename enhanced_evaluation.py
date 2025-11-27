# Enhanced evaluation with multiple advanced algorithms - FAIR VERSION
import random
import math
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from classes import ERSimulation, Patient
from optimizers.linear_elite_optimizer import EvolutionaryTriageOptimizer
from optimizers.linear_tournament_optimizer import ImprovedEvolutionaryTriageOptimizer
from optimizers.advanced_optimizer import FairAdvancedEvolutionaryOptimizer
from optimizers.hybrid_optimizer import HybridOptimizer
from optimizers.neural_optimizer import FairNeuralEvolutionOptimizer, create_fair_neural_triage_function

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

def log_patient_arrivals(arrivals, seed, folder_path, scenario_type):
    """Log patient arrival details to a text file"""
    os.makedirs(folder_path, exist_ok=True)
    
    filename = os.path.join(folder_path, f"patient_arrivals_seed_{seed}.txt")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"PATIENT ARRIVALS LOG\n")
        f.write(f"Scenario Type: {scenario_type}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Total timesteps: {len(arrivals)}\n")
        f.write(f"=" * 60 + "\n\n")
        
        patient_count = 0
        for timestep, patient in enumerate(arrivals):
            if patient is not None:
                patient_count += 1
                f.write(f"T{timestep:2d}: Patient {patient.id} | Sev:{patient.severity} Det:{patient.deterioration_chance:.2f} Treat:{patient.treatment_time}min {patient.presenting} | HR:{patient.vitals['hr']} SpO2:{patient.vitals['spo2']} Res:{patient.expected_resources}\n")
            else:
                f.write(f"T{timestep:2d}: No arrival\n")
        
        f.write(f"\nSUMMARY\n")
        f.write(f"Total patients arrived: {patient_count}\n")
        f.write(f"Arrival rate: {patient_count/len(arrivals)*100:.1f}%\n")

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

def evaluate_policy_on_scenarios(policy, test_seeds, num_nurses=4, total_time=96, arrival_prob=0.3, print_schedules=False):
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
        
        if print_schedules:
            print(f"\n--- Nurse Schedule for Seed {seed} ---")
            sim.print_nurse_schedule()
            
        results.append(metrics)
    
    return results

def main():
    print("=== ENHANCED ALGORITHM EVALUATION ===")
    print("Training and testing multiple advanced algorithms against ESI and MTS\n")
    
    # Training seeds (used during optimization)
    train_seeds = [1000, 1001, 1002]
    
    # Testing seeds (completely separate, never seen during training)
    test_seeds = [2000, 2001, 2002, 2003, 2004]
    
    # Log training scenario patient arrivals
    print("Logging training scenario patient arrivals...")
    for seed in train_seeds:
        arrivals = generate_arrivals(96, 0.3, seed)
        log_patient_arrivals(arrivals, seed, "logs/patient_arrivals/training", "Training")
    
    print(f"Training on seeds: {train_seeds}")
    print(f"Testing on seeds: {test_seeds}")
    print()
    
    trained_policies = {}
    
    # Train all algorithms with same parameters for fair comparison
    training_params = {
        'num_nurses': 3,
        'total_time': 96,
        'arrival_prob': 0.3,
        'seed': train_seeds[0]
    }
    
    print("=== TRAINING PHASE ===\n")
    
    # 1. Linear Elite Algorithm (baseline)
    print("1. Training Linear Elite Algorithm...")
    original_optimizer = EvolutionaryTriageOptimizer(
        num_generations=100, 
        population_size=100, 
        **training_params
    )
    trained_policies['Linear Elite'] = original_optimizer.run("logs/train_linear_elite.txt")
    print("   Complete")
    
    # 2. Linear Tournament Algorithm (with tournament selection)
    print("2. Training Linear Tournament Algorithm...")
    improved_optimizer = ImprovedEvolutionaryTriageOptimizer(
        num_generations=100, 
        population_size=100, 
        **training_params
    )
    trained_policies['Linear Tournament'] = improved_optimizer.run("logs/train_linear_tournament.txt")
    print("   Complete")
    
    # 3. Advanced Algorithm (expanded features but fair data)
    print("3. Training Advanced Algorithm...")
    fair_advanced_optimizer = FairAdvancedEvolutionaryOptimizer(
        num_generations=100, 
        population_size=100, 
        **training_params
    )
    trained_policies['Advanced'] = fair_advanced_optimizer.run("logs/train_advanced.txt")
    print("   Complete")
    
    # 4. Hybrid Algorithm (multi-strategy)
    print("4. Training Hybrid Algorithm...")
    hybrid_optimizer = HybridOptimizer(
        num_generations=100, 
        population_size=100, 
        **training_params
    )
    trained_policies['Hybrid'] = hybrid_optimizer.run("logs/train_hybrid.txt")
    print("   Complete")
    
    # 5. Neural Evolution Algorithm
    print("5. Training Neural Algorithm...")
    fair_neural_optimizer = FairNeuralEvolutionOptimizer(
        num_generations=100, 
        population_size=80, 
        **training_params
    )
    neural_policy = fair_neural_optimizer.run("logs/train_neural.txt")
    trained_policies['Neural'] = create_fair_neural_triage_function(neural_policy, fair_neural_optimizer)
    print("   Complete")
    
    print("\n=== TESTING PHASE ===")
    print("Evaluating all algorithms on unseen test data...")
    
    # Log testing scenario patient arrivals
    print("\nLogging testing scenario patient arrivals...")
    for seed in test_seeds:
        arrivals = generate_arrivals(96, 0.3, seed)
        log_patient_arrivals(arrivals, seed, "logs/patient_arrivals/testing", "Testing")
    
    # Test all policies on unseen scenarios
    all_policies = {
        **trained_policies,
        'ESI': esi_policy,
        'MTS': mts_policy
    }
    
    all_results = {}
    
    for name, policy in all_policies.items():
        print(f"\nTesting {name}...")
        # Save nurse schedule for ESI as example
        print_schedules = (name == 'ESI')
        results = evaluate_policy_on_scenarios(policy, test_seeds, print_schedules=print_schedules)
        all_results[name] = results
        
        # Save nurse schedule to file for ESI
        if name == 'ESI':
            # Run one simulation to get nurse schedule
            from classes import ERSimulation
            arrivals = generate_arrivals(96, 0.3, 2000)
            sim = ERSimulation(
                num_nurses=3,
                total_time=96,
                arrival_prob=0,
                triage_policy=policy,
                verbose=False,
                seed=2000,
                use_shifts=True
            )
            sim.patient_arrivals = arrivals
            sim.run()
            sim.print_nurse_schedule("logs/nurse_schedule_esi_example.txt")
        
        # Calculate performance metrics
        avg_wait_times = [r['avg_wait'] for r in results if r['avg_wait'] is not None]
        avg_weighted_waits = [r['avg_weighted_wait'] for r in results if r['avg_weighted_wait'] is not None]
        completed_counts = [r['completed'] for r in results]
        still_waiting_counts = [r['still_waiting'] for r in results]
        
        avg_completed = sum(completed_counts) / len(completed_counts)
        avg_still_waiting = sum(still_waiting_counts) / len(still_waiting_counts)
        avg_wait = sum(avg_wait_times) / len(avg_wait_times) if avg_wait_times else 0
        avg_weighted = sum(avg_weighted_waits) / len(avg_weighted_waits) if avg_weighted_waits else 0
        
        print(f"  Completed patients: {avg_completed:.1f}")
        print(f"  Unattended patients: {avg_still_waiting:.1f}")
        print(f"  Average wait time: {avg_wait:.2f}")
        print(f"  Average weighted wait: {avg_weighted:.2f}")
    
    print("\n" + "="*60)
    print("FINAL PERFORMANCE RANKING")
    print("="*60)
    
    # Save results to logs file
    results_file = "logs/evaluation_results.txt"
    with open(results_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("FINAL PERFORMANCE RANKING\n")
        f.write("="*60 + "\n")
    
    # Calculate final rankings
    performance_summary = []
    for name in all_policies.keys():
        results = all_results[name]
        avg_weighted_waits = [r['avg_weighted_wait'] for r in results if r['avg_weighted_wait'] is not None]
        completed_counts = [r['completed'] for r in results]
        still_waiting_counts = [r['still_waiting'] for r in results]
        
        if avg_weighted_waits:
            avg_performance = sum(avg_weighted_waits) / len(avg_weighted_waits)
            avg_completed = sum(completed_counts) / len(completed_counts)
            avg_unattended = sum(still_waiting_counts) / len(still_waiting_counts)
            
            # Combined score: lower weighted wait + higher completion - penalty for unattended
            completion_rate = avg_completed / 40  # Assume ~40 patients is good
            unattended_penalty = avg_unattended * 0.5  # Penalty for leaving patients unattended
            combined_score = avg_performance - (completion_rate - 1) * 10 + unattended_penalty
            
            performance_summary.append((name, avg_performance, avg_completed, avg_unattended, combined_score))
    
    # Sort by combined score (lower is better)
    performance_summary.sort(key=lambda x: x[4])
    
    # Display and save results
    header = f"{'Rank':<4} {'Algorithm':<15} {'Weighted Wait':<13} {'Completed':<10} {'Unattended':<11} {'Combined Score':<13}"
    separator = "-" * 80
    
    print(header)
    print(separator)
    
    with open(results_file, "a") as f:
        f.write(header + "\n")
        f.write(separator + "\n")
    
    for rank, (name, weighted_wait, completed, unattended, combined) in enumerate(performance_summary, 1):
        line = f"{rank:<4} {name:<15} {weighted_wait:<13.2f} {completed:<10.1f} {unattended:<11.1f} {combined:<13.2f}"
        print(line)
        with open(results_file, "a") as f:
            f.write(line + "\n")
    
    print("\n" + "="*60)
    print("IMPROVEMENT ANALYSIS")
    print("="*60)
    
    # Get baseline performances
    esi_performance = None
    mts_performance = None
    for name, weighted_wait, completed, unattended, combined in performance_summary:
        if name == 'ESI':
            esi_performance = weighted_wait
        elif name == 'MTS':
            mts_performance = weighted_wait
    
    # Save analysis header to file
    with open(results_file, "a") as f:
        f.write("\n" + "="*60 + "\n")
        f.write("IMPROVEMENT ANALYSIS\n") 
        f.write("="*60 + "\n")
    
    # Compare against baselines
    if esi_performance and mts_performance:
        esi_msg = f"ESI baseline performance: {esi_performance:.2f}"
        mts_msg = f"MTS baseline performance: {mts_performance:.2f}"
        print(esi_msg)
        print(mts_msg)
        print("\nComparison vs baselines:")
        
        with open(results_file, "a") as f:
            f.write(esi_msg + "\n")
            f.write(mts_msg + "\n\n")
            f.write("Comparison vs baselines:\n")
        
        improvements = []
        for name, weighted_wait, completed, unattended, combined in performance_summary:
            if name != 'ESI' and name != 'MTS':
                esi_improvement = ((esi_performance - weighted_wait) / esi_performance) * 100
                mts_improvement = ((mts_performance - weighted_wait) / mts_performance) * 100
                improvements.append((name, esi_improvement, mts_improvement, weighted_wait))
                
        improvements.sort(key=lambda x: x[1], reverse=True)  # Sort by ESI improvement
        
        for name, esi_imp, mts_imp, performance in improvements:
            esi_sign = "+" if esi_imp > 0 else ""
            mts_sign = "+" if mts_imp > 0 else ""
            line = f"  {name:<15}: ESI {esi_sign}{esi_imp:5.1f}%, MTS {mts_sign}{mts_imp:5.1f}% (score: {performance:.2f})"
            print(line)
            with open(results_file, "a", encoding='utf-8') as f:
                f.write(line + "\n")
    
    # Identify best performing algorithm
    best_algorithm = performance_summary[0][0]
    best_weighted_wait = performance_summary[0][1]  # Weighted wait score
    best_combined_score = performance_summary[0][4]  # Combined score
    
    winner_msg1 = f"*** WINNER: {best_algorithm} ***"
    winner_msg2 = f"   Best weighted wait score: {best_weighted_wait:.2f} (combined: {best_combined_score:.2f})"
    
    print(f"\n{winner_msg1}")
    print(winner_msg2)
    
    with open(results_file, "a", encoding='utf-8') as f:
        f.write("\n" + winner_msg1 + "\n")
        f.write(winner_msg2 + "\n")
    
    if best_algorithm not in ['ESI', 'MTS']:
        success_msg = "   Successfully outperformed baseline algorithms!"
        print(success_msg)
        with open(results_file, "a", encoding='utf-8') as f:
            f.write(success_msg + "\n")
    else:
        need_opt_msg = "   Need further optimization to beat baseline algorithms."
        print(need_opt_msg)
        with open(results_file, "a", encoding='utf-8') as f:
            f.write(need_opt_msg + "\n")
    
    print(f"\nAll results saved to 'logs/evaluation_results.txt'")
    print(f"Nurse schedule example saved to 'logs/nurse_schedule_esi_example.txt'")
    print(f"Patient arrival logs saved to 'logs/patient_arrivals/' directory")
    print(f"\nDetailed logs saved in 'logs/' directory")
    print("Training complete! Done.")
    
    with open(results_file, "a", encoding='utf-8') as f:
        f.write(f"\nAll results saved to 'logs/evaluation_results.txt'\n")
        f.write(f"Nurse schedule example saved to 'logs/nurse_schedule_esi_example.txt'\n")
        f.write(f"Patient arrival logs saved to 'logs/patient_arrivals/' directory\n")
        f.write(f"\nDetailed logs saved in 'logs/' directory\n")
        f.write("Training complete! Done.\n")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    import os
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    main()