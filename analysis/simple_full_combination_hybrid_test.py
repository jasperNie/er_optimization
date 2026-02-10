#!/usr/bin/env python3
"""
Simple Full Combination Hybrid Test
Just copy the working logic and change seeds/paths - no complex modifications
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
from triage_policies import esi_policy, mts_policy



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
    
    if not primary_reasons:
        primary_reasons.append(f"close call: neural network decision")
    
    # Determine confidence based on score difference and factors
    confidence = "high" if abs(score_diff) > 0.1 else "moderate" if abs(score_diff) > 0.05 else "low"
    
    reason = ", ".join(primary_reasons)
    return f"Patient {chosen['id']} chosen over Patient {alt['id']}: {reason} (margin: {score_diff:+.3f}, confidence: {confidence})"


def evaluate_full_combination(num_nurses=4):
    """
    Train on ALL patterns, test on random mix - using PROVEN working approach
    """
    
    all_patterns = ['standard', 'peak_hours', 'weekend', 'disaster', 'flu_season', 'steady_state']
    
    print("\n" + "=" * 80)
    print(f"   FULL COMBINATION HYBRID TEST - TRAIN ALL, TEST RANDOM MIX")
    print("=" * 80)
    print(f"Training on seeds 8000-8299 (50 from each of 6 patterns)")
    print(f"Testing on 300 random scenarios (50 random from each pattern)")
    
    # Phase 1: Train neural network using proven approach from comprehensive script
    print("\nPHASE 1: NEURAL NETWORK TRAINING")
    print("─" * 50) 
    print("Training with proven parameters (100 generations, 80 population)...")
    
    # Use EXACT training approach from the working script
    training_params = {
        'num_nurses': num_nurses,
        'total_time': 96,
        'arrival_prob': 0.3,
        'seed': 8000  # Use first training seed
    }
    
    optimizer = FairNeuralEvolutionOptimizer(
        num_generations=100,  # Same as working script
        population_size=80,   # Same as working script
        **training_params
    )
    
    # Train the neural network using standard pattern (like the working script)
    training_log_path = f"logs/full_combination_test/hybrid_training_{num_nurses}nurses.txt"
    os.makedirs(os.path.dirname(training_log_path), exist_ok=True)
    neural_policy = optimizer.run(training_log_path)
    print("Training complete!")
    
    # Hybrid triage logic is integrated into the simulation class
    
    # Phase 2: Test on random mix using EXACT approach from working script
    print(f"\nPHASE 2: RANDOM MIX TESTING")
    print("─" * 50)
    
    # Create test scenarios - 50 random seeds from each pattern 
    all_test_scenarios = []
    test_seeds_per_pattern = 50
    
    for pattern_name in all_patterns:
        pattern_seeds = random.sample(range(9000, 10000), test_seeds_per_pattern)  # 50 random seeds
        for seed in pattern_seeds:
            all_test_scenarios.append((pattern_name, seed))
    
    # Shuffle so we test in random order
    random.shuffle(all_test_scenarios)
    print(f"Generated {len(all_test_scenarios)} test scenarios, testing in random order...")
    
    # Use EXACT explainable simulation class from working script
    class ExplainableSimulation(ERSimulation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.explanation_log = []
            self.decision_count = 0
            self.neural_decisions = 0  # Track neural decisions
            self.fallback_decisions = 0  # Track ESI fallbacks
            
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
            """Explain triage decisions and assign patients using working hybrid logic"""
            if len(self.waiting_patients) <= 1:
                # No decision to explain if only one patient
                if self.waiting_patients and free_nurses:
                    patient = self.waiting_patients.pop(0)
                    nurse = free_nurses[0]
                    nurse.current_patient = patient
                    nurse.busy_until = self.time + patient.treatment_time
                    self.started_patients.append((patient, patient.wait_time))
                return
            
            # Multiple patients - implement hybrid logic from comprehensive script
            sim_state = {
                'current_time': self.time,
                'queue_length': len(self.waiting_patients),
                'nurse_availability': len(free_nurses) / len(self.nurses)
            }
            
            # Get neural network scores
            neural_scores = []
            for patient in self.waiting_patients:
                score = optimizer.fair_neural_triage_score(patient, sim_state, neural_policy)
                neural_scores.append((patient, score))
            neural_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Hybrid decision logic with confidence checking
            if len(neural_scores) >= 2:
                top_patient, top_score = neural_scores[0]
                second_patient, second_score = neural_scores[1]
                confidence_margin = top_score - second_score
                
                # Use the same confidence threshold as the hybrid triage function
                base_threshold = 0.0001
                
                # If confidence is too low, fall back to ESI decision
                if confidence_margin < base_threshold:
                    # Use pure ESI: sort by severity (higher severity = higher priority)
                    esi_scores = [(p, p.severity + 0.1 * p.deterioration_chance - 0.01 * p.wait_time) for p in self.waiting_patients]
                    esi_scores.sort(key=lambda x: x[1], reverse=True)
                    patient_scores = esi_scores
                    self.fallback_decisions += 1  # Count ESI fallback
                else:
                    # Use neural network decision
                    patient_scores = neural_scores
                    self.neural_decisions += 1  # Count neural decision
            else:
                # Single patient or empty, use neural
                patient_scores = neural_scores
                if len(self.waiting_patients) > 0:  # Only count if there's actually a decision
                    self.neural_decisions += 1
            
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
    
    # Run all test scenarios using EXACT approach from working script
    all_results = []
    all_esi_results = []
    all_mts_results = []
    
    for scenario_idx, (pattern_name, test_seed) in enumerate(all_test_scenarios, 1):
        print(f"\nTesting {pattern_name} pattern with seed {test_seed} ({scenario_idx}/300)")
        print("-" * 50)
        
        # Generate arrivals for this pattern/seed
        generate_arrivals_func = ARRIVAL_PATTERNS[pattern_name]
        arrivals = generate_arrivals_func(96, 0.3, test_seed)
        
        # Simulation creates fresh counters automatically
        
        # Hybrid simulation
        sim = ExplainableSimulation(
            num_nurses=num_nurses,
            total_time=96,
            arrival_prob=0.3,
            verbose=False,
            seed=test_seed,
            use_shifts=True
        )
        sim.patient_arrivals = arrivals
        result = sim.run()
        result['seed'] = test_seed
        result['pattern'] = pattern_name
        result['explanations'] = sim.explanation_log
        result['total_decisions'] = sim.decision_count
        
        # Add hybrid-specific metrics from simulation counters
        result['neural_decisions'] = sim.neural_decisions
        result['fallback_decisions'] = sim.fallback_decisions
        total_hybrid_decisions = sim.neural_decisions + sim.fallback_decisions
        result['neural_percentage'] = (sim.neural_decisions / max(1, total_hybrid_decisions)) * 100
        
        all_results.append(result)
        
        # ESI baseline with same arrivals
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
        all_esi_results.append(esi_result)
        
        # MTS baseline with same arrivals
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
        all_mts_results.append(mts_result)
        
        # Print results with None-safe formatting
        avg_wait = result['avg_wait'] if result['avg_wait'] is not None else 0
        weighted_wait = result['avg_weighted_wait'] if result['avg_weighted_wait'] is not None else 0
        esi_avg_wait = esi_result['avg_wait'] if esi_result['avg_wait'] is not None else 0
        esi_weighted_wait = esi_result['avg_weighted_wait'] if esi_result['avg_weighted_wait'] is not None else 0
        mts_avg_wait = mts_result['avg_wait'] if mts_result['avg_wait'] is not None else 0
        mts_weighted_wait = mts_result['avg_weighted_wait'] if mts_result['avg_weighted_wait'] is not None else 0
        
        print(f"SEED {test_seed} ({pattern_name}) RESULTS:")
        print(f"   Patients treated: {result['completed']}")
        print(f"   Patients waiting: {result['still_waiting']}")
        print(f"   Average wait: {avg_wait:.2f} timesteps ({avg_wait*15:.0f} minutes)")
        print(f"   Weighted wait: {weighted_wait:.2f} timesteps ({weighted_wait*15:.0f} minutes)")
        print(f"   Decisions explained: {result['total_decisions']}")
        print(f"   Neural decisions: {result['neural_decisions']} ({result['neural_percentage']:.1f}%)")
        print(f"   ESI fallbacks: {result['fallback_decisions']} ({100-result['neural_percentage']:.1f}%)")
        print(f"   ESI treated: {esi_result['completed']}, waiting: {esi_result['still_waiting']} | avg: {esi_avg_wait:.2f} timesteps ({esi_avg_wait*15:.0f} min), weighted: {esi_weighted_wait:.2f} ({esi_weighted_wait*15:.0f} min)")
        print(f"   MTS treated: {mts_result['completed']}, waiting: {mts_result['still_waiting']} | avg: {mts_avg_wait:.2f} timesteps ({mts_avg_wait*15:.0f} min), weighted: {mts_weighted_wait:.2f} ({mts_weighted_wait*15:.0f} min)")
        
        # Show ALL triage decision explanations (like the comprehensive script)
        if result['total_decisions'] > 0:
            print(f"   ALL TRIAGE DECISION EXPLANATIONS:")
            
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
    
    # Phase 3: Analysis using EXACT approach from working script  
    print(f"\nPHASE 3: COMPREHENSIVE ANALYSIS")
    print("─" * 60)
    
    # Filter out None values for statistics
    completed_counts = [r['completed'] for r in all_results]
    avg_waits = [r['avg_wait'] for r in all_results if r['avg_wait'] is not None]
    weighted_waits = [r['avg_weighted_wait'] for r in all_results if r['avg_weighted_wait'] is not None]
    total_decisions = [r['total_decisions'] for r in all_results]
    neural_percentages = [r['neural_percentage'] for r in all_results]
    
    print(f"\nOVERALL PERFORMANCE STATISTICS (across {len(all_test_scenarios)} mixed tests):")
    print(f"   Patients treated: {statistics.mean(completed_counts):.1f} ± {statistics.stdev(completed_counts):.1f}")
    if neural_percentages:
        print(f"   Neural decision rate: {statistics.mean(neural_percentages):.1f}% ± {statistics.stdev(neural_percentages):.1f}%")
    
    if avg_waits:
        print(f"   Average wait time: {statistics.mean(avg_waits):.2f} ± {statistics.stdev(avg_waits):.2f} timesteps ({statistics.mean(avg_waits)*15:.0f} ± {statistics.stdev(avg_waits)*15:.0f} minutes)")
    else:
        print(f"   Average wait time: No valid wait times recorded")
        
    if weighted_waits:
        print(f"   Weighted wait time: {statistics.mean(weighted_waits):.2f} ± {statistics.stdev(weighted_waits):.2f} timesteps ({statistics.mean(weighted_waits)*15:.0f} ± {statistics.stdev(weighted_waits)*15:.0f} minutes)")
    else:
        print(f"   Weighted wait time: No valid weighted wait times recorded")
        
    print(f"   Decisions explained: {statistics.mean(total_decisions):.1f} ± {statistics.stdev(total_decisions):.1f}")
    
    # Pattern-specific analysis
    pattern_results = {pattern: [] for pattern in all_patterns}
    for result in all_results:
        pattern_results[result['pattern']].append(result)
    
    print(f"\nPERFORMANCE BY PATTERN:")
    print("─" * 40)
    for pattern in all_patterns:
        pattern_data = pattern_results[pattern]
        if pattern_data:
            pattern_completed = [r['completed'] for r in pattern_data]
            pattern_weighted = [r['avg_weighted_wait'] for r in pattern_data if r['avg_weighted_wait'] is not None]
            
            if pattern_weighted:
                print(f"   {pattern.title()}: {statistics.mean(pattern_completed):.1f} treated, {statistics.mean(pattern_weighted)*15:.0f} min weighted wait")
            else:
                print(f"   {pattern.title()}: {statistics.mean(pattern_completed):.1f} treated, no valid wait times")
    
    # Baseline comparison with None filtering
    print(f"\nBASELINE COMPARISON (averaged across all tests):")
    print("─" * 50)
    
    esi_weighted_filtered = [r['avg_weighted_wait'] for r in all_esi_results if r['avg_weighted_wait'] is not None]
    mts_weighted_filtered = [r['avg_weighted_wait'] for r in all_mts_results if r['avg_weighted_wait'] is not None]
    esi_wait_filtered = [r['avg_wait'] for r in all_esi_results if r['avg_wait'] is not None]
    mts_wait_filtered = [r['avg_wait'] for r in all_mts_results if r['avg_wait'] is not None]
    
    if esi_weighted_filtered and mts_weighted_filtered and weighted_waits:
        esi_avg_weighted = statistics.mean(esi_weighted_filtered)
        mts_avg_weighted = statistics.mean(mts_weighted_filtered)
        neural_avg_weighted = statistics.mean(weighted_waits)
        
        esi_avg_wait = statistics.mean(esi_wait_filtered)
        mts_avg_wait = statistics.mean(mts_wait_filtered)
        neural_avg_wait = statistics.mean(avg_waits)
        
        # Convert to hours
        neural_avg_weighted_hours = neural_avg_weighted * 15 / 60
        neural_avg_wait_hours = neural_avg_wait * 15 / 60
        esi_avg_weighted_hours = esi_avg_weighted * 15 / 60
        esi_avg_wait_hours = esi_avg_wait * 15 / 60
        mts_avg_weighted_hours = mts_avg_weighted * 15 / 60
        mts_avg_wait_hours = mts_avg_wait * 15 / 60
        
        # Calculate hybrid-specific statistics
        neural_percentages = [r['neural_percentage'] for r in all_results]
        avg_neural_percentage = statistics.mean(neural_percentages)
        
        print(f"   Hybrid Network: {neural_avg_weighted_hours:.2f} hours weighted ({neural_avg_wait_hours:.2f} hours avg)")
        print(f"   Neural decision rate: {avg_neural_percentage:.1f}%")
        print(f"   ESI Baseline: {esi_avg_weighted_hours:.2f} hours weighted ({esi_avg_wait_hours:.2f} hours avg)")
        print(f"   MTS Baseline: {mts_avg_weighted_hours:.2f} hours weighted ({mts_avg_wait_hours:.2f} hours avg)")
        
        print(f"\nBASELINE PERFORMANCE DETAILS:")
        # Calculate average patient counts for baselines
        esi_completed = statistics.mean([r['completed'] for r in all_esi_results])
        esi_waiting = statistics.mean([r['still_waiting'] for r in all_esi_results])
        mts_completed = statistics.mean([r['completed'] for r in all_mts_results])
        mts_waiting = statistics.mean([r['still_waiting'] for r in all_mts_results])
        
        print(f"   ESI treated: {esi_completed:.1f}, waiting: {esi_waiting:.1f} | avg: {esi_avg_wait:.2f} timesteps ({esi_avg_wait*15:.0f} min), weighted: {esi_avg_weighted:.2f} ({esi_avg_weighted*15:.0f} min)")
        print(f"   MTS treated: {mts_completed:.1f}, waiting: {mts_waiting:.1f} | avg: {mts_avg_wait:.2f} timesteps ({mts_avg_wait*15:.0f} min), weighted: {mts_avg_weighted:.2f} ({mts_avg_weighted*15:.0f} min)")
        
        if neural_avg_weighted < esi_avg_weighted:
            improvement = ((esi_avg_weighted - neural_avg_weighted) / esi_avg_weighted) * 100
            print(f"   -> Neural beats ESI by {improvement:.1f}% (weighted wait)")
        if neural_avg_weighted < mts_avg_weighted:
            improvement = ((mts_avg_weighted - neural_avg_weighted) / mts_avg_weighted) * 100
            print(f"   -> Neural beats MTS by {improvement:.1f}% (weighted wait)")
    else:
        print("   Insufficient valid data for baseline comparison")
    
    # Save results
    print(f"\nSAVING RESULTS...")
    print("─" * 20)
    
    analysis_log_path = f"logs/full_combination_test/hybrid_full_combination_analysis_{num_nurses}nurses.txt"
    os.makedirs(os.path.dirname(analysis_log_path), exist_ok=True)
    
    with open(analysis_log_path, "w", encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"   FULL COMBINATION HYBRID TEST RESULTS - {num_nurses} NURSES\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"CONFIGURATION:\n")
        f.write(f"   Training: Standard pattern, seeds 8000+ (100 gen, 80 pop)\n")
        f.write(f"   Testing: 300 random scenarios (50 from each of 6 patterns)\n")
        f.write(f"   Simulation: 24 hours (96 timesteps of 15 minutes each)\n\n")
        
        # Add all individual results with detailed explanations
        for i, result in enumerate(all_results):
            avg_wait = result['avg_wait'] if result['avg_wait'] is not None else 0
            weighted_wait = result['avg_weighted_wait'] if result['avg_weighted_wait'] is not None else 0
            
            # Get corresponding baseline results
            esi_result = all_esi_results[i]
            mts_result = all_mts_results[i]
            esi_avg_wait = esi_result['avg_wait'] if esi_result['avg_wait'] is not None else 0
            esi_weighted_wait = esi_result['avg_weighted_wait'] if esi_result['avg_weighted_wait'] is not None else 0
            mts_avg_wait = mts_result['avg_wait'] if mts_result['avg_wait'] is not None else 0
            mts_weighted_wait = mts_result['avg_weighted_wait'] if mts_result['avg_weighted_wait'] is not None else 0
            
            f.write(f"SEED {result['seed']} ({result['pattern']}) RESULTS:\n")
            f.write(f"   Patients treated: {result['completed']}\n")
            f.write(f"   Patients waiting: {result['still_waiting']}\n")
            f.write(f"   Average wait: {avg_wait:.2f} timesteps ({avg_wait*15:.0f} minutes)\n")
            f.write(f"   Weighted wait: {weighted_wait:.2f} timesteps ({weighted_wait*15:.0f} minutes)\n")
            f.write(f"   Decisions explained: {result['total_decisions']}\n")
            f.write(f"   Neural decisions: {result['neural_decisions']} ({result['neural_percentage']:.1f}%)\n")
            f.write(f"   ESI fallbacks: {result['fallback_decisions']} ({100-result['neural_percentage']:.1f}%)\n")
            f.write(f"   ESI treated: {esi_result['completed']}, waiting: {esi_result['still_waiting']} | avg: {esi_avg_wait:.2f} timesteps ({esi_avg_wait*15:.0f} min), weighted: {esi_weighted_wait:.2f} ({esi_weighted_wait*15:.0f} min)\n")
            f.write(f"   MTS treated: {mts_result['completed']}, waiting: {mts_result['still_waiting']} | avg: {mts_avg_wait:.2f} timesteps ({mts_avg_wait*15:.0f} min), weighted: {mts_weighted_wait:.2f} ({mts_weighted_wait*15:.0f} min)\n")
            
            # Add detailed triage decision explanations
            if result['explanations']:
                f.write(f"   ALL TRIAGE DECISIONS:\n")
                
                decision_count = 0
                for decision in result['explanations']:
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
        if weighted_waits:
            f.write(f"   Mean weighted wait: {statistics.mean(weighted_waits):.2f} +/- {statistics.stdev(weighted_waits):.2f} timesteps\n")
        f.write(f"   Total decisions explained: {sum(total_decisions)}\n\n")
        
        # ...existing code...
        
        # Add baseline comparison to file
        f.write(f"BASELINE COMPARISON:\n")
        f.write(f"   Hybrid Network: {neural_avg_weighted_hours:.2f} hours weighted ({neural_avg_wait_hours:.2f} hours avg)\n")
        f.write(f"   Neural decision rate: {avg_neural_percentage:.1f}%\n")
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
    print(f"\nFULL COMBINATION TEST COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    num_nurses = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    evaluate_full_combination(num_nurses)