#!/usr/bin/env python3
"""
Comprehensive Neural Network Evaluation with Different Arrival Patterns
Same as comprehensive_neural_evaluation.py but tests different arrival patterns
"""

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import statistics
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimizers.neural_optimizer import FairNeuralEvolutionOptimizer
from classes import ERSimulation
from enhanced_evaluation import log_patient_arrivals
from arrival_patterns import ARRIVAL_PATTERNS, get_pattern_description

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

def evaluate_pattern(pattern_name='standard', num_nurses=4):
    """
    Comprehensive evaluation with specified arrival pattern and nurse count
    IDENTICAL to comprehensive_neural_evaluation.py except for arrival pattern and nurse count
    """
    
    # IDENTICAL training and testing parameters
    training_seeds = list(range(8000, 8050))  # 50 training seeds (8000-8049)
    test_seeds = list(range(9000, 9100))  # 100 test seeds (9000-9099)
    
    print("\n" + "=" * 80)
    print(f"   COMPREHENSIVE NEURAL NETWORK EVALUATION - {pattern_name.upper()} PATTERN")
    print("=" * 80)
    print(f"Pattern: {get_pattern_description(pattern_name)}")
    print(f"Training on {len(training_seeds)} seeds: {training_seeds[0]}-{training_seeds[-1]}")
    print(f"Testing on {len(test_seeds)} seeds: {test_seeds[0]}-{test_seeds[-1]}")
    
    # IDENTICAL Phase 1: Full Training
    print("\nPHASE 1: FULL NEURAL NETWORK TRAINING")
    print("─" * 50)
    print("Training with full parameters (100 generations, 80 population)...")
    
    # Get the arrival pattern function 
    generate_arrivals_func = ARRIVAL_PATTERNS[pattern_name]
    
    # IDENTICAL training parameters
    training_params = {
        'num_nurses': num_nurses,  # Use specified nurse count
        'total_time': 96,
        'arrival_prob': 0.3,
        'seed': training_seeds[0]  # Use first training seed
    }
    
    optimizer = FairNeuralEvolutionOptimizer(
        num_generations=100,  # Same as enhanced_evaluation
        population_size=80,   # Same as enhanced_evaluation
        **training_params
    )
    
    # Train the neural network using the specified pattern
    training_log_path = f"logs/complete_evaluation/neural/{pattern_name}_{num_nurses}nurses_training.txt"
    os.makedirs(os.path.dirname(training_log_path), exist_ok=True)
    neural_policy = optimizer.run(training_log_path)
    print("Full training complete!")
    
    # Create callable function from neural policy
    from optimizers.neural_optimizer import create_fair_neural_triage_function
    neural_triage_function = create_fair_neural_triage_function(neural_policy, optimizer)
    
    # IDENTICAL Phase 2: Multi-seed evaluation with explanations  
    print(f"\nPHASE 2: MULTI-SEED EVALUATION WITH EXPLANATIONS")
    print("─" * 60)
    

    
    # IDENTICAL explainable simulation class
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
    
    all_results = []
    all_esi_results = []  # Store baseline results for aggregate analysis
    all_mts_results = []  # Store baseline results for aggregate analysis
    
    for seed_idx, test_seed in enumerate(test_seeds, 1):
        print(f"\nTesting with seed {test_seed} ({seed_idx}/100)")
        print("-" * 50)
        
        # CRITICAL: Generate identical arrivals for all three simulations using the specified pattern
        arrivals = generate_arrivals_func(96, 0.3, test_seed)
        

        
        # IDENTICAL simulation runs
        sim = ExplainableSimulation(
            num_nurses=num_nurses,  # Use specified nurse count
            total_time=96,  # 24 hours
            arrival_prob=0.3,  # Same as enhanced_evaluation
            triage_policy=neural_triage_function,
            verbose=False,
            seed=test_seed,
            use_shifts=True
        )
        # Use identical arrivals
        sim.patient_arrivals = arrivals
        
        result = sim.run()
        result['seed'] = test_seed
        result['pattern'] = pattern_name
        result['explanations'] = sim.explanation_log
        result['total_decisions'] = sim.decision_count
        all_results.append(result)
        
        # IDENTICAL baseline comparisons with same arrivals
        from triage_policies import esi_policy
        esi_sim = ERSimulation(
            num_nurses=num_nurses,
            total_time=96,
            arrival_prob=0.3,
            triage_policy=esi_policy,
            verbose=False,
            seed=test_seed,
            use_shifts=True
        )
        esi_sim.patient_arrivals = arrivals.copy()
        esi_result = esi_sim.run()
        
        from triage_policies import mts_policy
        mts_sim = ERSimulation(
            num_nurses=num_nurses,
            total_time=96,
            arrival_prob=0.3,
            triage_policy=mts_policy,
            verbose=False,
            seed=test_seed,
            use_shifts=True
        )
        mts_sim.patient_arrivals = arrivals.copy()
        mts_result = mts_sim.run()
        
        # Store baseline results for aggregate analysis
        all_esi_results.append(esi_result)
        all_mts_results.append(mts_result)
        
        # Print summary for this seed
        print(f"SEED {test_seed} RESULTS:")
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
                    print(f"     {decision_count}. t={time_str}: {decision['chosen']} (only patient waiting)")
    
    # IDENTICAL Phase 3: Aggregate Analysis
    print(f"\nPHASE 3: AGGREGATE ANALYSIS ACROSS ALL SEEDS")
    print("─" * 60)
    
    # IDENTICAL overall statistics
    completed_counts = [r['completed'] for r in all_results]
    avg_waits = [r['avg_wait'] for r in all_results]
    weighted_waits = [r['avg_weighted_wait'] for r in all_results]
    total_decisions = [r['total_decisions'] for r in all_results]
    
    print(f"\nPERFORMANCE STATISTICS (across {len(test_seeds)} seeds) - {pattern_name.upper()} PATTERN:")
    print(f"   Patients treated: {statistics.mean(completed_counts):.1f} ± {statistics.stdev(completed_counts):.1f}")
    print(f"   Average wait time: {statistics.mean(avg_waits):.2f} ± {statistics.stdev(avg_waits):.2f} timesteps ({statistics.mean(avg_waits)*15:.0f} ± {statistics.stdev(avg_waits)*15:.0f} minutes)")
    print(f"   Weighted wait time: {statistics.mean(weighted_waits):.2f} ± {statistics.stdev(weighted_waits):.2f} timesteps ({statistics.mean(weighted_waits)*15:.0f} ± {statistics.stdev(weighted_waits)*15:.0f} minutes)")
    print(f"   Decisions explained: {statistics.mean(total_decisions):.1f} ± {statistics.stdev(total_decisions):.1f}")
    
    # IDENTICAL decision pattern analysis
    print(f"\nDECISION PATTERN ANALYSIS:")
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
    
    # IDENTICAL baseline comparison
    print(f"\nBASELINE COMPARISON (averaged across all seeds):")
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
    
    print(f"   Neural Network: {neural_avg_weighted_hours:.2f} hours weighted wait ({neural_avg_wait_hours:.2f} hours avg wait)")
    print(f"   ESI (severity):  {esi_avg_weighted_hours:.2f} hours weighted wait ({esi_avg_wait_hours:.2f} hours avg wait)")
    print(f"   MTS (wait time): {mts_avg_weighted_hours:.2f} hours weighted wait ({mts_avg_wait_hours:.2f} hours avg wait)")
    
    print(f"\nBASELINE PERFORMANCE DETAILS:")
    # Calculate average patient counts for baselines
    esi_completed = statistics.mean([r['completed'] for r in esi_results])
    esi_waiting = statistics.mean([r['still_waiting'] for r in esi_results])
    mts_completed = statistics.mean([r['completed'] for r in mts_results])
    mts_waiting = statistics.mean([r['still_waiting'] for r in mts_results])
    
    print(f"   ESI treated: {esi_completed:.1f}, waiting: {esi_waiting:.1f} | avg: {esi_avg_wait:.2f} timesteps ({esi_avg_wait*15:.0f} min), weighted: {esi_avg_weighted:.2f} ({esi_avg_weighted*15:.0f} min)")
    print(f"   MTS treated: {mts_completed:.1f}, waiting: {mts_waiting:.1f} | avg: {mts_avg_wait:.2f} timesteps ({mts_avg_wait*15:.0f} min), weighted: {mts_avg_weighted:.2f} ({mts_avg_weighted*15:.0f} min)")
    
    if neural_avg_weighted < esi_avg_weighted:
        weighted_improvement = ((esi_avg_weighted - neural_avg_weighted) / esi_avg_weighted) * 100
        avg_improvement = ((esi_avg_wait - neural_avg_wait) / esi_avg_wait) * 100
        print(f"   -> Neural beats ESI by {weighted_improvement:.1f}% (weighted wait) and {avg_improvement:.1f}% (avg wait)")
    
    if neural_avg_weighted < mts_avg_weighted:
        weighted_improvement = ((mts_avg_weighted - neural_avg_weighted) / mts_avg_weighted) * 100
        avg_improvement = ((mts_avg_wait - neural_avg_wait) / mts_avg_wait) * 100
        print(f"   -> Neural beats MTS by {weighted_improvement:.1f}% (weighted wait) and {avg_improvement:.1f}% (avg wait)")
    
    # Save detailed results
    print(f"\nSAVING DETAILED RESULTS:")
    print("─" * 30)
    
    analysis_log_path = f"logs/complete_evaluation/neural/{pattern_name}_{num_nurses}nurses_analysis.txt"
    os.makedirs(os.path.dirname(analysis_log_path), exist_ok=True)
    
    with open(analysis_log_path, "w", encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"   COMPREHENSIVE NEURAL NETWORK EVALUATION - {pattern_name.upper()} PATTERN - {num_nurses} NURSES\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"ARRIVAL PATTERN: {get_pattern_description(pattern_name)}\n\n")
        
        f.write(f"TRAINING CONFIGURATION:\n")
        f.write(f"   Generations: 100\n")
        f.write(f"   Population: 80\n")
        f.write(f"   Training seeds: {len(training_seeds)} seeds ({training_seeds[0]}-{training_seeds[-1]})\n")
        f.write(f"   Test seeds: {len(test_seeds)} seeds ({test_seeds[0]}-{test_seeds[-1]})\n")
        f.write(f"   Simulation: 24 hours (96 timesteps of 15 minutes each)\n\n")
        
        # Add detailed results exactly like comprehensive neural evaluation
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
    
    print(f"   Results saved to: {analysis_log_path}")
    
    # Create output directory for this run
    output_dir = f"logs/complete_evaluation/neural/{pattern_name}_{num_nurses}nurses_charts"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGENERATING INDIVIDUAL CHARTS...")
    print(f"   Output directory: {output_dir}/")
    
    # IDENTICAL chart generation to comprehensive neural evaluation
    # Chart 1: Performance Comparison Bar Chart
    plt.figure(figsize=(12, 8))
    policies = ['Neural\\nNetwork', 'ESI\\nBaseline', 'MTS\\nBaseline']
    weighted_times = [neural_avg_weighted_hours, esi_avg_weighted_hours, mts_avg_weighted_hours]
    avg_times = [neural_avg_wait_hours, esi_avg_wait_hours, mts_avg_wait_hours]
    
    x = np.arange(len(policies))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, weighted_times, width, label='Weighted Wait Time', 
                    color=['#2E8B57', '#CD853F', '#DC143C'], alpha=0.8)
    bars2 = plt.bar(x + width/2, avg_times, width, label='Average Wait Time',
                    color=['#3CB371', '#F4A460', '#FF6B6B'], alpha=0.8)
    
    plt.title(f'Triage Policy Performance - {pattern_name.title()} Pattern - {num_nurses} Nurses (Lower is Better)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Triage Policy', fontsize=12, fontweight='bold')
    plt.ylabel('Wait Time (Hours)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}h', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}h', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    chart1_path = f"{output_dir}/1_neural_performance_comparison.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Neural performance comparison saved: {chart1_path}")
    
    # IDENTICAL remaining charts
    # Chart 2: Neural decision margins distribution
    plt.figure(figsize=(10, 6))
    
    decision_margins = []
    if all_decisions:
        for decision in all_decisions:
            if decision['alternatives']:
                margin = decision['chosen_score'] - decision['alternatives'][0][1]
                decision_margins.append(margin)
    
    if decision_margins:
        plt.hist(decision_margins, bins=20, color='#3498DB', alpha=0.7, edgecolor='black')
        plt.title(f'Neural Decision Margins - {pattern_name.title()} Pattern ({len(decision_margins)} decisions)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Decision Margin (Confidence)')
        plt.ylabel('Frequency')
        plt.axvline(statistics.mean(decision_margins), color='red', linestyle='--', 
                   label=f'Mean: {statistics.mean(decision_margins):.3f}')
        plt.legend()
        plt.grid(alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No decision margin data available', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=14)
        plt.title(f'Neural Decision Margins - {pattern_name.title()} Pattern', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    chart2_path = f"{output_dir}/2_neural_decision_confidence.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Neural decision confidence saved: {chart2_path}")
    
    # Chart 3: Patient treatment distribution
    plt.figure(figsize=(10, 6))
    patients_treated = completed_counts
    plt.hist(patients_treated, bins=15, color='#E74C3C', alpha=0.7, edgecolor='black')
    plt.title(f'Patient Treatment Distribution - {pattern_name.title()} Pattern (100 Seeds)', fontsize=16, fontweight='bold')
    plt.xlabel('Patients Treated per Seed')
    plt.ylabel('Frequency')
    plt.axvline(statistics.mean(patients_treated), color='red', linestyle='--', 
               label=f'Mean: {statistics.mean(patients_treated):.1f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    chart3_path = f"{output_dir}/3_neural_patient_treatment.png"
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Neural patient treatment saved: {chart3_path}")
    
    # Chart 4: Wait time distribution
    plt.figure(figsize=(10, 6))
    wait_times_all = weighted_waits
    plt.hist(wait_times_all, bins=15, color='#27AE60', alpha=0.7, edgecolor='black')
    plt.title(f'Wait Time Distribution - {pattern_name.title()} Pattern (100 Seeds)', fontsize=16, fontweight='bold')
    plt.xlabel('Weighted Wait Time (Timesteps)')
    plt.ylabel('Frequency')
    plt.axvline(statistics.mean(wait_times_all), color='red', linestyle='--', 
               label=f'Mean: {statistics.mean(wait_times_all):.1f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    chart4_path = f"{output_dir}/4_neural_wait_time_distribution.png"
    plt.savefig(chart4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Neural wait time distribution saved: {chart4_path}")
    print(f"\nAll charts generated successfully in: {output_dir}/")

    print(f"\nCOMPREHENSIVE {pattern_name.upper()} EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"The neural network has been tested with {pattern_name} arrival pattern and {num_nurses} nurses")
    print("using identical methodology to the comprehensive evaluation!")

if __name__ == "__main__":
    import sys
    pattern = sys.argv[1] if len(sys.argv) > 1 else 'standard'
    num_nurses = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if pattern not in ARRIVAL_PATTERNS:
        print(f"Unknown pattern: {pattern}")
        print(f"Available patterns: {list(ARRIVAL_PATTERNS.keys())}")
    else:
        evaluate_pattern(pattern, num_nurses)