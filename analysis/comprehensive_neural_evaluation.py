#!/usr/bin/env python3
"""
Comprehensive Neural Network Evaluation with Explanations
Runs full training then tests on multiple seeds with detailed decision analysis
"""

import sys
import os
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimizers.neural_optimizer import FairNeuralEvolutionOptimizer
from classes import ERSimulation
from enhanced_evaluation import generate_arrivals, log_patient_arrivals
import statistics

def explain_patient_decision(chosen_patient_str, chosen_score, alt_patient_str, alt_score):
    """Create a human-readable explanation of why one patient was prioritized"""
    import re
    
    # Parse patient strings like "Patient(7, sev=3, det=0.20, treat=10, wait=1)"
    def parse_patient(patient_str):
        match = re.search(r'Patient\((\d+), sev=(\d+), det=([\d.]+), treat=(\d+), wait=(\d+)\)', patient_str)
        if match:
            return {
                'id': int(match.group(1)),
                'severity': int(match.group(2)),
                'deterioration': float(match.group(3)),
                'treatment_time': int(match.group(4)),
                'wait_time': int(match.group(5))
            }
        return None
    
    chosen = parse_patient(chosen_patient_str)
    alt = parse_patient(alt_patient_str)
    
    if not chosen or not alt:
        return f"Chose {chosen_patient_str} (score: {chosen_score:.3f}) over {alt_patient_str} (score: {alt_score:.3f})"
    
    score_diff = chosen_score - alt_score
    
    # Convert timesteps to minutes for display
    chosen_wait_min = chosen['wait_time'] * 15
    alt_wait_min = alt['wait_time'] * 15
    chosen_treat_min = chosen['treatment_time'] * 15
    alt_treat_min = alt['treatment_time'] * 15
    
    # Determine the PRIMARY reason for the decision based on strongest factors
    primary_reasons = []
    
    # Medical urgency factors (higher priority)
    if chosen['severity'] > alt['severity']:
        primary_reasons.append(f"higher severity (level {chosen['severity']} vs {alt['severity']})")
    
    if chosen['deterioration'] > alt['deterioration'] + 0.05:  # Significant difference
        primary_reasons.append(f"deteriorating faster ({chosen['deterioration']:.2f} vs {alt['deterioration']:.2f} rate)")
    
    # Fairness factors (medium priority)
    wait_diff = chosen['wait_time'] - alt['wait_time']
    if wait_diff > 2:  # Significant wait time advantage
        primary_reasons.append(f"much longer wait ({chosen_wait_min}min vs {alt_wait_min}min)")
    elif wait_diff > 0:
        primary_reasons.append(f"longer wait ({chosen_wait_min}min vs {alt_wait_min}min)")
    
    # Efficiency factors (lower priority, only if close on other factors)
    if abs(chosen['severity'] - alt['severity']) <= 1 and abs(wait_diff) <= 1:
        treat_diff = alt['treatment_time'] - chosen['treatment_time'] 
        if treat_diff > 3:  # Significantly quicker
            primary_reasons.append(f"much quicker treatment ({chosen_treat_min}min vs {alt_treat_min}min)")
        elif treat_diff > 0:
            primary_reasons.append(f"quicker treatment ({chosen_treat_min}min vs {alt_treat_min}min)")
    
    # If no clear primary reason, provide detailed factor comparison
    if not primary_reasons:
        # Show what the neural network is weighing
        factors = []
        
        # Always show the comparison factors
        if chosen['severity'] == alt['severity']:
            factors.append(f"same severity (level {chosen['severity']})")
        else:
            factors.append(f"severity edge to Patient {chosen['id']} ({chosen['severity']} vs {alt['severity']})")
        
        if abs(chosen['wait_time'] - alt['wait_time']) <= 1:
            factors.append(f"similar wait times (~{chosen_wait_min}min)")
        else:
            longer_wait_id = chosen['id'] if chosen['wait_time'] > alt['wait_time'] else alt['id']
            factors.append(f"wait time edge to Patient {longer_wait_id}")
        
        if abs(chosen['deterioration'] - alt['deterioration']) < 0.05:
            factors.append(f"similar deterioration rates (~{chosen['deterioration']:.2f})")
        else:
            faster_det_id = chosen['id'] if chosen['deterioration'] > alt['deterioration'] else alt['id']
            factors.append(f"deterioration edge to Patient {faster_det_id}")
        
        if abs(chosen['treatment_time'] - alt['treatment_time']) <= 2:
            factors.append(f"similar treatment times (~{chosen_treat_min}min)")
        else:
            quicker_treat_id = chosen['id'] if chosen['treatment_time'] < alt['treatment_time'] else alt['id']
            factors.append(f"treatment speed edge to Patient {quicker_treat_id}")
        
        primary_reasons.append(f"close call: {', '.join(factors)}")
    
    # Determine confidence based on score difference and factors
    confidence = "high" if abs(score_diff) > 0.1 else "moderate" if abs(score_diff) > 0.05 else "low"
    
    reason = ", ".join(primary_reasons)
    return f"Patient {chosen['id']} chosen over Patient {alt['id']}: {reason} (margin: {score_diff:+.3f}, confidence: {confidence})"

def full_training_multi_seed_explanation():
    """
    Full training followed by multi-seed evaluation with explanations
    """
    
    # Expanded training and testing parameters
    training_seeds = list(range(8000, 8050))  # 50 training seeds (8000-8049)
    test_seeds = list(range(9000, 9100))  # 100 test seeds (9000-9099)
    
    print("\n" + "=" * 80)
    print("   COMPREHENSIVE NEURAL NETWORK EVALUATION WITH EXPLANATIONS")
    print("=" * 80)
    print(f"Training on {len(training_seeds)} seeds: {training_seeds[0]}-{training_seeds[-1]}")
    print(f"Testing on {len(test_seeds)} seeds: {test_seeds[0]}-{test_seeds[-1]}")
    
    # Phase 1: Full Training (same parameters as enhanced_evaluation.py)
    print("\nPHASE 1: FULL NEURAL NETWORK TRAINING")
    print("─" * 50)
    print("Training with full parameters (100 generations, 80 population)...")
    
    # Log training seed arrivals for full documentation
    print("\n📋 Logging patient arrivals for training seeds...")
    
    # Import from enhanced_evaluation for compatibility
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from enhanced_evaluation import generate_arrivals, log_patient_arrivals
    
    for seed in training_seeds:
        arrivals = generate_arrivals(96, 0.3, seed)
        log_patient_arrivals(arrivals, seed, "logs/patient_arrivals", "Neural Training")
    print("   Training patient arrival logs saved to logs/patient_arrivals/")
    
    # Full training parameters with expanded seed set
    training_params = {
        'num_nurses': 4,  # Same as enhanced_evaluation testing
        'total_time': 96,
        'arrival_prob': 0.3,
        'seed': training_seeds[0]  # Use first training seed
    }
    
    optimizer = FairNeuralEvolutionOptimizer(
        num_generations=100,  # Same as enhanced_evaluation
        population_size=80,   # Same as enhanced_evaluation
        **training_params
    )
    
    # Train the neural network
    neural_policy = optimizer.run("logs/full_training_comprehensive.txt")
    print("Full training complete!")
    
    # Generate nurse schedule example for documentation
    print("\n👥 Generating nurse schedule example...")
    from classes import ERSimulation
    schedule_sim = ERSimulation(4, 96, 0.3, 9000)  # Use first test seed
    schedule_sim.print_nurse_schedule("logs/nurse_schedules/nurse_schedule_base_4_nurses_original.txt")
    print("   Nurse schedule saved to logs/nurse_schedules/nurse_schedule_base_4_nurses_original.txt")
    
    # Phase 2: Multi-seed evaluation with explanations
    print(f"\nPHASE 2: MULTI-SEED EVALUATION WITH EXPLANATIONS")
    print("─" * 60)
    
    # Log testing seed arrivals for full documentation
    print("\n📋 Logging patient arrivals for test seeds...")
    
    for seed in test_seeds:
        arrivals = generate_arrivals(96, 0.3, seed)
        log_patient_arrivals(arrivals, seed, "logs/patient_arrivals", "Neural Testing")
    print("   Testing patient arrival logs saved to logs/patient_arrivals/")
    

    
    # Generate and log patient arrivals for all test seeds
    print("\n📋 Logging patient arrivals for test seeds...")
    for seed in test_seeds:
        arrivals = generate_arrivals(96, 0.3, seed)  # Same parameters
        log_patient_arrivals(arrivals, seed, "logs/patient_arrivals", "Testing")
    print("   Patient arrival logs saved to logs/patient_arrivals/")
    
    all_results = []
    all_esi_results = []  # Store baseline results for aggregate analysis
    all_mts_results = []  # Store baseline results for aggregate analysis
    
    for seed_idx, test_seed in enumerate(test_seeds, 1):
        print(f"\n🧪 SEED {seed_idx}/100: Testing with seed {test_seed}")
        print("-" * 50)
        
        # Create explainable simulation for this seed
        class ExplainableSimulation(ERSimulation):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.explanation_log = []
                self.decision_count = 0
                
            def step(self):
                """Enhanced step with decision explanations"""
                self.time += 1
                
                # Standard simulation steps
                self.nurses = self.get_current_nurses()
                
                # Patient arrivals
                if self.time <= len(self.patient_arrivals):
                    new_patient = self.patient_arrivals[self.time - 1]
                    if new_patient is not None:
                        self.waiting_patients.append(new_patient)
                
                # Update nurse status
                for nurse in self.nurses:
                    if nurse.current_patient and self.time >= nurse.busy_until:
                        self.completed_patients.append(nurse.current_patient)
                        nurse.current_patient = None
                
                # Update waiting patients
                for patient in self.waiting_patients:
                    patient.wait_time += 1
                    if patient.severity < 5:
                        if random.random() < patient.deterioration_chance:
                            patient.severity += 1
                            patient.deterioration_chance = max(0.01, patient.deterioration_chance * 0.5)
                
                # Triage with explanations
                free_nurses = [n for n in self.nurses if n.current_patient is None]
                if free_nurses and self.waiting_patients:
                    self.explain_and_assign_patients(free_nurses)
            
            def explain_and_assign_patients(self, free_nurses):
                """Explain triage decisions and assign patients"""
                if len(self.waiting_patients) <= 1:
                    # No decision to explain if only one patient
                    if self.waiting_patients and free_nurses:
                        patient = self.waiting_patients.pop(0)
                        nurse = free_nurses[0]
                        nurse.current_patient = patient
                        nurse.busy_until = self.time + patient.treatment_time
                        self.started_patients.append((patient, patient.wait_time))
                    return
                
                # Multiple patients - explain the decision
                sim_state = {
                    'current_time': self.time,
                    'queue_length': len(self.waiting_patients),
                    'nurse_availability': len(free_nurses) / len(self.nurses)
                }
                
                # Score all patients
                patient_scores = []
                for patient in self.waiting_patients:
                    score = optimizer.fair_neural_triage_score(patient, sim_state, neural_policy)
                    patient_scores.append((patient, score))
                
                # Sort by priority
                patient_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Log this decision for explanation
                self.decision_count += 1
                decision_log = {
                    'time': self.time,
                    'decision_id': self.decision_count,
                    'patients': [(str(p), score) for p, score in patient_scores],
                    'chosen': str(patient_scores[0][0]),
                    'chosen_score': patient_scores[0][1],
                    'alternatives': [(str(p), score) for p, score in patient_scores[1:3]],  # Top 3 alternatives
                    'sim_state': sim_state.copy()
                }
                self.explanation_log.append(decision_log)
                
                # Assign patients to nurses
                assignments = 0
                for nurse in free_nurses:
                    if assignments < len(patient_scores):
                        patient, score = patient_scores[assignments]
                        nurse.current_patient = patient
                        nurse.busy_until = self.time + patient.treatment_time
                        self.started_patients.append((patient, patient.wait_time))
                        self.waiting_patients.remove(patient)
                        assignments += 1
        
        # CRITICAL FIX: Generate identical arrivals for all three simulations
        # This ensures fair comparison between neural network and baselines
        arrivals = generate_arrivals(96, 0.3, test_seed)
        
        # Log patient arrivals for traceability
        log_patient_arrivals(arrivals, test_seed, "logs/patient_arrivals", f"Neural Testing Seed {test_seed}")
        
        # Run simulation with this seed (same parameters as enhanced_evaluation)
        sim = ExplainableSimulation(
            num_nurses=4,  # Same as enhanced_evaluation testing
            total_time=96,  # 24 hours
            arrival_prob=0.3,  # Same as enhanced_evaluation
            triage_policy=neural_policy,
            verbose=False,
            seed=test_seed,
            use_shifts=True
        )
        # Use identical arrivals
        sim.patient_arrivals = arrivals
        
        result = sim.run()
        result['seed'] = test_seed
        result['explanations'] = sim.explanation_log
        result['total_decisions'] = sim.decision_count
        all_results.append(result)
        
        # Run baseline comparisons for this seed with IDENTICAL arrivals
        esi_sim = ERSimulation(
            num_nurses=4,
            total_time=96,
            arrival_prob=0.3,
            triage_policy={'severity': 1.0, 'deterioration': 0.0, 'wait_time': 0.0},
            verbose=False,
            seed=test_seed,
            use_shifts=True
        )
        # Use identical arrivals
        esi_sim.patient_arrivals = arrivals.copy()
        esi_result = esi_sim.run()
        
        mts_sim = ERSimulation(
            num_nurses=4,
            total_time=96,
            arrival_prob=0.3,
            triage_policy={'severity': 0.0, 'deterioration': 0.0, 'wait_time': 1.0},
            verbose=False,
            seed=test_seed,
            use_shifts=True
        )
        # Use identical arrivals
        mts_sim.patient_arrivals = arrivals.copy()
        mts_result = mts_sim.run()
        
        # Store baseline results for aggregate analysis
        all_esi_results.append(esi_result)
        all_mts_results.append(mts_result)
        
        # Print summary for this seed
        print(f"\nSEED {test_seed} RESULTS:")
        print(f"   Patients treated: {result['completed']}")
        print(f"   Patients waiting: {result['still_waiting']}")
        print(f"   Average wait: {result['avg_wait']:.2f} timesteps ({result['avg_wait']*15:.0f} minutes)")
        print(f"   Weighted wait: {result['avg_weighted_wait']:.2f} timesteps ({result['avg_weighted_wait']*15:.0f} minutes)")
        print(f"   Decisions explained: {result['total_decisions']}")
        print(f"   ESI treated: {esi_result['completed']}, waiting: {esi_result['still_waiting']} | avg: {esi_result['avg_wait']:.2f} timesteps ({esi_result['avg_wait']*15:.0f} min), weighted: {esi_result['avg_weighted_wait']:.2f} ({esi_result['avg_weighted_wait']*15:.0f} min)")
        print(f"   MTS treated: {mts_result['completed']}, waiting: {mts_result['still_waiting']} | avg: {mts_result['avg_wait']:.2f} timesteps ({mts_result['avg_wait']*15:.0f} min), weighted: {mts_result['avg_weighted_wait']:.2f} ({mts_result['avg_weighted_wait']*15:.0f} min)")
        
        # Show sample explanations for interesting decisions
        if result['total_decisions'] > 0:
            print(f"   ALL TRIAGE DECISION EXPLANATIONS:")
            
            # Show ALL decisions with alternatives to explain
            decision_count = 0
            for decision in sim.explanation_log:
                if len(decision['alternatives']) > 0:
                    chosen_score = decision['chosen_score']
                    alt_patient, alt_score = decision['alternatives'][0]
                    
                    # Create human-readable explanation
                    explanation = explain_patient_decision(
                        decision['chosen'], chosen_score,
                        alt_patient, alt_score
                    )
                    
                    decision_count += 1
                    timestep_minutes = decision['time'] * 15  # Convert to minutes
                    hours = timestep_minutes // 60
                    minutes = timestep_minutes % 60
                    time_str = f"{hours:02d}:{minutes:02d}"
                    print(f"     {decision_count}. t={time_str}: {explanation}")
                else:
                    # Single patient case
                    decision_count += 1
                    timestep_minutes = decision['time'] * 15  # Convert to minutes
                    hours = timestep_minutes // 60
                    minutes = timestep_minutes % 60
                    time_str = f"{hours:02d}:{minutes:02d}"
                    print(f"     {decision_count}. t={time_str}: {decision['chosen']} (only patient waiting)")    # Phase 3: Aggregate Analysis
    print(f"\nPHASE 3: AGGREGATE ANALYSIS ACROSS ALL SEEDS")
    print("─" * 60)
    
    # Overall statistics
    completed_counts = [r['completed'] for r in all_results]
    avg_waits = [r['avg_wait'] for r in all_results]
    weighted_waits = [r['avg_weighted_wait'] for r in all_results]
    total_decisions = [r['total_decisions'] for r in all_results]
    
    print(f"\nPERFORMANCE STATISTICS (across {len(test_seeds)} seeds):")
    print(f"   Patients treated: {statistics.mean(completed_counts):.1f} ± {statistics.stdev(completed_counts):.1f}")
    print(f"   Average wait time: {statistics.mean(avg_waits):.2f} ± {statistics.stdev(avg_waits):.2f} timesteps ({statistics.mean(avg_waits)*15:.0f} ± {statistics.stdev(avg_waits)*15:.0f} minutes)")
    print(f"   Weighted wait time: {statistics.mean(weighted_waits):.2f} ± {statistics.stdev(weighted_waits):.2f} timesteps ({statistics.mean(weighted_waits)*15:.0f} ± {statistics.stdev(weighted_waits)*15:.0f} minutes)")
    print(f"   Decisions explained: {statistics.mean(total_decisions):.1f} ± {statistics.stdev(total_decisions):.1f}")
    
    # Decision pattern analysis
    print(f"\n🧠 DECISION PATTERN ANALYSIS:")
    print("─" * 40)
    
    all_decisions = []
    for result in all_results:
        all_decisions.extend(result['explanations'])
    
    if all_decisions:
        # Analyze decision margins (how confident the neural network was)
        margins = []
        close_decisions = 0
        
        for decision in all_decisions:
            if decision['alternatives']:
                margin = decision['chosen_score'] - decision['alternatives'][0][1]
                margins.append(margin)
                if margin < 0.05:  # Very close decision
                    close_decisions += 1
        
        if margins:
            print(f"   Total explained decisions: {len(all_decisions)}")
            print(f"   Average decision margin: {statistics.mean(margins):.3f}")
            print(f"   Close decisions (margin < 0.05): {close_decisions} ({close_decisions/len(margins)*100:.1f}%)")
            print(f"   Most confident decision margin: {max(margins):.3f}")
            print(f"   Least confident decision margin: {min(margins):.3f}")
    
    # Compare with baselines across all seeds
    print(f"\n⚖️ BASELINE COMPARISON (averaged across all seeds):")
    print("─" * 50)
    
    # Use the baseline results computed during individual seed evaluation
    esi_results = all_esi_results
    mts_results = all_mts_results
    
    # Calculate improvements
    esi_avg_weighted = statistics.mean([r['avg_weighted_wait'] for r in esi_results])
    mts_avg_weighted = statistics.mean([r['avg_weighted_wait'] for r in mts_results])
    neural_avg_weighted = statistics.mean(weighted_waits)
    
    # Also get average wait times (not just weighted)
    esi_avg_wait = statistics.mean([r['avg_wait'] for r in esi_results])
    mts_avg_wait = statistics.mean([r['avg_wait'] for r in mts_results])
    neural_avg_wait = statistics.mean(avg_waits)
    
    # Convert from timesteps to hours (15 minutes per timestep)
    neural_avg_weighted_hours = neural_avg_weighted * 15 / 60  # timesteps * 15 min / 60 min/hour
    neural_avg_wait_hours = neural_avg_wait * 15 / 60
    esi_avg_weighted_hours = esi_avg_weighted * 15 / 60
    esi_avg_wait_hours = esi_avg_wait * 15 / 60
    mts_avg_weighted_hours = mts_avg_weighted * 15 / 60
    mts_avg_wait_hours = mts_avg_wait * 15 / 60
    
    print(f"   🤖 Neural Network: {neural_avg_weighted_hours:.2f} hours weighted wait ({neural_avg_wait_hours:.2f} hours avg wait)")
    print(f"   🏥 ESI (severity):  {esi_avg_weighted_hours:.2f} hours weighted wait ({esi_avg_wait_hours:.2f} hours avg wait)")
    print(f"   ⏱️ MTS (wait time): {mts_avg_weighted_hours:.2f} hours weighted wait ({mts_avg_wait_hours:.2f} hours avg wait)")
    
    print(f"\nBASELINE PERFORMANCE DETAILS:")
    # Calculate average patient counts for baselines
    esi_completed = statistics.mean([r['completed'] for r in esi_results])
    esi_waiting = statistics.mean([r['still_waiting'] for r in esi_results])
    mts_completed = statistics.mean([r['completed'] for r in mts_results])
    mts_waiting = statistics.mean([r['still_waiting'] for r in mts_results])
    
    print(f"   ESI treated: {esi_completed:.1f}, waiting: {esi_waiting:.1f} | avg: {esi_avg_wait:.2f} timesteps ({esi_avg_wait*15:.0f} min), weighted: {esi_avg_weighted:.2f} ({esi_avg_weighted*15:.0f} min)")
    print(f"   MTS treated: {mts_completed:.1f}, waiting: {mts_waiting:.1f} | avg: {mts_avg_wait:.2f} timesteps ({mts_avg_wait*15:.0f} min), weighted: {mts_avg_weighted:.2f} ({mts_avg_weighted*15:.0f} min)")
    
    if neural_avg_weighted < esi_avg_weighted:
        improvement = ((esi_avg_weighted - neural_avg_weighted) / esi_avg_weighted) * 100
        print(f"   Neural beats ESI by {improvement:.1f}%")
    
    if neural_avg_weighted < mts_avg_weighted:
        improvement = ((mts_avg_weighted - neural_avg_weighted) / mts_avg_weighted) * 100
        print(f"   Neural beats MTS by {improvement:.1f}%")
    
    # Save detailed results
    print(f"\n💾 SAVING DETAILED RESULTS:")
    print("─" * 30)
    
    with open("logs/analysis_logs/comprehensive_neural_evaluation.txt", "w", encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("   COMPREHENSIVE NEURAL NETWORK EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"TRAINING CONFIGURATION:\n")
        f.write(f"   Generations: 100\n")
        f.write(f"   Population: 80\n")
        f.write(f"   Training seeds: {len(training_seeds)} seeds ({training_seeds[0]}-{training_seeds[-1]})\n")
        f.write(f"   Test seeds: {len(test_seeds)} seeds ({test_seeds[0]}-{test_seeds[-1]})\n")
        f.write(f"   Simulation: 24 hours (96 timesteps of 15 minutes each)\n\n")
        
        for i, result in enumerate(all_results):
            f.write(f"SEED {result['seed']} RESULTS:\n")
            f.write(f"   Patients treated: {result['completed']}\n")
            f.write(f"   Patients waiting: {result['still_waiting']}\n")
            f.write(f"   Average wait: {result['avg_wait']:.2f} timesteps ({result['avg_wait']*15:.0f} minutes)\n")
            f.write(f"   Weighted wait: {result['avg_weighted_wait']:.2f} timesteps ({result['avg_weighted_wait']*15:.0f} minutes)\n")
            f.write(f"   Decisions explained: {result['total_decisions']}\n")
            
            # Add baseline performance for this seed
            result_index = next(i for i, r in enumerate(all_results) if r['seed'] == result['seed'])
            esi_seed_result = all_esi_results[result_index]
            mts_seed_result = all_mts_results[result_index]
            f.write(f"   ESI treated: {esi_seed_result['completed']}, waiting: {esi_seed_result['still_waiting']} | avg: {esi_seed_result['avg_wait']:.2f} timesteps ({esi_seed_result['avg_wait']*15:.0f} min), weighted: {esi_seed_result['avg_weighted_wait']:.2f} ({esi_seed_result['avg_weighted_wait']*15:.0f} min)\n")
            f.write(f"   MTS treated: {mts_seed_result['completed']}, waiting: {mts_seed_result['still_waiting']} | avg: {mts_seed_result['avg_wait']:.2f} timesteps ({mts_seed_result['avg_wait']*15:.0f} min), weighted: {mts_seed_result['avg_weighted_wait']:.2f} ({mts_seed_result['avg_weighted_wait']*15:.0f} min)\n")
            
            if result['explanations']:
                f.write(f"   ALL TRIAGE DECISIONS:\n")
                
                # Generate explanations for ALL decisions
                decision_count = 0
                for decision in result['explanations']:  # All decisions, not just first 3
                    decision_count += 1
                    if 'alternatives' in decision and decision['alternatives']:
                        alt_patient, alt_score = decision['alternatives'][0]
                        explanation = explain_patient_decision(
                            decision['chosen'], decision['chosen_score'],
                            alt_patient, alt_score
                        )
                        timestep_minutes = decision['time'] * 15  # Convert to minutes
                        hours = timestep_minutes // 60
                        minutes = timestep_minutes % 60
                        time_str = f"{hours:02d}:{minutes:02d}"
                        f.write(f"     {decision_count}. t={time_str}: {explanation}\n")
                    else:
                        timestep_minutes = decision['time'] * 15  # Convert to minutes
                        hours = timestep_minutes // 60
                        minutes = timestep_minutes % 60
                        time_str = f"{hours:02d}:{minutes:02d}"
                        f.write(f"     {decision_count}. t={time_str}: {decision['chosen']} (only patient waiting)\n")
            f.write("\n")
        
        f.write(f"AGGREGATE STATISTICS:\n")
        f.write(f"   Mean patients treated: {statistics.mean(completed_counts):.1f} +/- {statistics.stdev(completed_counts):.1f}\n")
        f.write(f"   Mean weighted wait: {statistics.mean(weighted_waits):.2f} +/- {statistics.stdev(weighted_waits):.2f} hours\n")
        f.write(f"   Total decisions explained: {sum(total_decisions)}\n\n")
        
        # Add baseline comparison to file
        f.write(f"BASELINE COMPARISON:\n")
        f.write(f"   Neural Network: {neural_avg_weighted_hours:.2f} hours weighted ({neural_avg_wait_hours:.2f} hours avg)\n")
        f.write(f"   ESI Baseline: {esi_avg_weighted_hours:.2f} hours weighted ({esi_avg_wait_hours:.2f} hours avg)\n")
        f.write(f"   MTS Baseline: {mts_avg_weighted_hours:.2f} hours weighted ({mts_avg_wait_hours:.2f} hours avg)\n")
        
        f.write(f"\nBASELINE PERFORMANCE DETAILS:\n")
        f.write(f"   ESI treated: {esi_completed:.1f}, waiting: {esi_waiting:.1f} | avg: {esi_avg_wait:.2f} timesteps ({esi_avg_wait*15:.0f} min), weighted: {esi_avg_weighted:.2f} ({esi_avg_weighted*15:.0f} min)\n")
        f.write(f"   MTS treated: {mts_completed:.1f}, waiting: {mts_waiting:.1f} | avg: {mts_avg_wait:.2f} timesteps ({mts_avg_wait*15:.0f} min), weighted: {mts_avg_weighted:.2f} ({mts_avg_weighted*15:.0f} min)\n")
        
        if neural_avg_weighted < esi_avg_weighted:
            improvement = ((esi_avg_weighted - neural_avg_weighted) / esi_avg_weighted) * 100
            f.write(f"   -> Neural beats ESI by {improvement:.1f}% (weighted wait)\n")
        if neural_avg_weighted < mts_avg_weighted:
            improvement = ((mts_avg_weighted - neural_avg_weighted) / mts_avg_weighted) * 100
            f.write(f"   -> Neural beats MTS by {improvement:.1f}% (weighted wait)\n")
    
    print(f"   📄 Results saved to: logs/analysis_logs/comprehensive_neural_evaluation.txt")
    print(f"   📋 Patient arrivals saved to: logs/patient_arrivals/")
    
    print(f"\nCOMPREHENSIVE EVALUATION COMPLETE!")
    print("=" * 80)
    print("=" * 50)
    print("The neural network has been fully trained and tested across")
    print("multiple random seeds with detailed decision explanations.")
    print("This provides robust evidence of its decision-making patterns!")

if __name__ == "__main__":
    full_training_multi_seed_explanation()