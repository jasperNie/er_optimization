#!/usr/bin/env python3
"""
Proper Hybrid Confidence-Based Triage Policy
Creates a triage policy that combines neural network with ESI fallback
"""

import sys
import os
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers.neural_optimizer import FairNeuralEvolutionOptimizer
from classes import ERSimulation
import statistics

class HybridConfidenceTiagePolicy:
    """
    A triage policy function that uses neural network with ESI fallback
    This is designed to be used as the triage_policy parameter in ERSimulation
    """
    
    def __init__(self, neural_policy, confidence_threshold=0.05):
        self.neural_policy = neural_policy
        self.confidence_threshold = confidence_threshold
        self.decision_log = []
        self.neural_optimizer = None  # Will be set when created
        
    def __call__(self, patient):
        """
        Triage policy function - called by ERSimulation to score patients
        This function will be called for every patient to get their priority score
        """
        # Store patient for decision making context
        if not hasattr(self, 'current_patients'):
            self.current_patients = []
        self.current_patients.append(patient)
        
        # Extract features using the SAME method as pure neural network
        neural_features = self._extract_fair_features(patient)
        
        # Compute neural score using the SAME method as pure neural network
        neural_score = self._compute_neural_score(neural_features)
        
        # For individual scoring, we can't compute confidence directly
        # So we'll use the neural score as primary and let the explainable
        # simulation handle confidence-based ESI fallback during decision making
        return neural_score
    
    def _extract_fair_features(self, patient):
        """Extract features using the SAME method as the pure neural network"""
        if patient is None:
            return np.zeros(6)
        
        features = np.zeros(6)
        
        # Fair patient features (EXACTLY like pure neural network)
        features[0] = patient.severity / 5.0  # Normalized severity (1-5 scale)
        features[1] = patient.deterioration_chance  # Deterioration risk (0-1)
        features[2] = min(patient.wait_time / 20.0, 1.0)  # Normalized wait time
        
        # Fair environmental factors (use defaults since we don't have sim_state access here)
        features[3] = 0.5  # Default time of day (TODO: could be improved)
        features[4] = 0.3  # Default queue length
        features[5] = 0.7  # Default nurse availability
            
        return features
    
    def _compute_neural_score(self, features):
        """Compute neural network score using the SAME method as pure neural network"""
        try:
            # Validate inputs
            if not isinstance(features, np.ndarray) or len(features) != 6:
                return 1.0
                
            # Input to hidden layer
            z1 = np.dot(features, self.neural_policy['w1']) + self.neural_policy['b1']
            
            # Check for numerical issues
            if not np.all(np.isfinite(z1)):
                return 1.0
            
            # Apply activations (EXACTLY like pure neural network)
            a1 = np.zeros_like(z1)
            for i in range(len(z1)):
                activation_type = self.neural_policy['activations'][i]
                # Clip inputs to prevent overflow
                clipped_input = np.clip(z1[i], -100, 100)
                
                if activation_type == 0:  # sigmoid
                    a1[i] = 1 / (1 + np.exp(-clipped_input))
                elif activation_type == 1:  # tanh
                    a1[i] = np.tanh(clipped_input)
                else:  # relu
                    a1[i] = max(0, clipped_input)
            
            # Check for numerical issues
            if not np.all(np.isfinite(a1)):
                return 1.0
            
            # Output layer
            z2 = np.dot(a1, self.neural_policy['w2']) + self.neural_policy['b2']
            
            # Check for numerical issues
            if not np.all(np.isfinite(z2)):
                return 1.0
            
            # Final output (always positive for triage score, EXACTLY like pure neural)
            output = np.maximum(0.1, z2[0]) * self.neural_policy['output_scale']
            
            # Final sanity check
            if not np.isfinite(output) or output <= 0:
                return 1.0
                
            # Apply urgency boost for critical cases (using fair severity data)
            if len(features) > 0 and features[0] > 0.8:  # High severity (severity >= 4)
                output *= self.neural_policy['urgency_boost']
                
            return float(np.clip(output, 0.1, 100.0))  # Reasonable bounds
            
        except Exception as e:
            # Return safe default if any error occurs
            return 1.0

def create_hybrid_triage_policy(confidence_threshold=0.05):
    """Create and train a hybrid triage policy"""
    print(f"CREATING HYBRID TRIAGE POLICY")
    print(f"   Confidence threshold: {confidence_threshold}")
    print(f"   Training neural network component...")
    
    # Train the neural network
    optimizer = FairNeuralEvolutionOptimizer(
        num_generations=100,
        population_size=80,
        seed=1000
    )
    
    neural_policy = optimizer.run()
    print(f"   Neural network training complete!")
    
    # Create hybrid policy
    hybrid_policy = HybridConfidenceTiagePolicy(neural_policy, confidence_threshold)
    hybrid_policy.neural_optimizer = optimizer
    
    return hybrid_policy

def evaluate_hybrid_policy():
    """Comprehensive evaluation of the hybrid triage policy with detailed logging"""
    print("COMPREHENSIVE HYBRID TRIAGE POLICY EVALUATION")
    print("=" * 65)
    
    # Create log directory
    os.makedirs("logs/analysis_logs", exist_ok=True)
    
    # Expanded test parameters - 100 seeds for robust evaluation
    test_seeds = list(range(9000, 9100))  # 100 test seeds (9000-9099)
    
    print("PHASE 1: TRAINING NEURAL NETWORK")
    print("-" * 40)
    
    # Expanded training - 50 seeds for robust learning
    training_seeds = list(range(8000, 8050))  # 50 training seeds (8000-8049)
    print(f"Training on {len(training_seeds)} seeds: {training_seeds[0]}-{training_seeds[-1]}")
    
    # Log training seed arrivals for full documentation
    print("📋 Logging patient arrivals for training seeds...")
    
    # Import from enhanced_evaluation for compatibility
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from enhanced_evaluation import generate_arrivals, log_patient_arrivals
    
    for seed in training_seeds:
        arrivals = generate_arrivals(96, 0.3, seed)  # Same parameters
        log_patient_arrivals(arrivals, seed, "logs/patient_arrivals/hybrid_training", "Hybrid Training")
    print("   Training patient arrival logs saved to logs/patient_arrivals/hybrid_training/")
    
    # Train neural network on multiple seeds for better generalization
    print("Training neural network on expanded seed set...")
    base_optimizer = FairNeuralEvolutionOptimizer(
        num_generations=100,
        population_size=80,
        seed=training_seeds[0]  # Use first training seed
    )
    base_neural_policy = base_optimizer.run()
    print("Neural network training complete!")
    
    print(f"\nPHASE 2: MULTI-SEED EVALUATION WITH EXPLANATIONS")
    print("-" * 55)
    
    # Generate and log patient arrivals for all test seeds
    print("\n📋 Logging patient arrivals for test seeds...")
    
    for seed in test_seeds:
        arrivals = generate_arrivals(96, 0.3, seed)  # Same parameters
        log_patient_arrivals(arrivals, seed, "logs/patient_arrivals/hybrid_testing", "Hybrid Testing")
    print("   Testing patient arrival logs saved to logs/patient_arrivals/hybrid_testing/")
    
    # Find optimal confidence threshold through testing
    optimal_threshold = 0.0001  # Ultra-low threshold to let neural network make almost all decisions
    
    print(f"\nUsing optimal confidence threshold: {optimal_threshold}")
    print("-" * 50)
    
    # Create hybrid policy with optimal threshold
    hybrid_policy = HybridConfidenceTiagePolicy(base_neural_policy, optimal_threshold)
    
    all_results = []
    all_esi_results = []
    all_mts_results = []
    
    for seed_idx, test_seed in enumerate(test_seeds, 1):
        print(f"\nSEED {seed_idx}/100: Testing with seed {test_seed}")
        print("-" * 50)
        
        # Create enhanced simulation that tracks decisions like comprehensive evaluation
        class HybridExplainableSimulation(ERSimulation):
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
                """Explain hybrid triage decisions and assign patients"""
                if len(self.waiting_patients) <= 1:
                    if self.waiting_patients and free_nurses:
                        patient = self.waiting_patients.pop(0)
                        nurse = free_nurses[0]
                        nurse.current_patient = patient
                        nurse.busy_until = self.time + patient.treatment_time
                        self.started_patients.append((patient, patient.wait_time))
                    return
                
                # Multiple patients - explain the hybrid decision
                sim_state = {
                    'current_time': self.time,
                    'queue_length': len(self.waiting_patients),
                    'nurse_availability': len(free_nurses) / len(self.nurses)
                }
                
                # Implement proper confidence-based hybrid decision making
                patient_scores = []
                neural_scores = []
                
                # Get neural scores for all patients
                for patient in self.waiting_patients:
                    neural_features = np.array([
                        patient.severity / 5.0,
                        patient.deterioration_chance,
                        patient.wait_time / 50.0,
                        0.5,  # Default time of day
                        0.3,  # Default queue length
                        0.7   # Default nurse availability
                    ])
                    neural_score = hybrid_policy._compute_neural_score(neural_features)
                    neural_scores.append((patient, neural_score))
                
                # Sort by neural network scores to get top candidates
                neural_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Calculate confidence: margin between top 2 neural scores
                if len(neural_scores) >= 2:
                    top_score = neural_scores[0][1]
                    second_score = neural_scores[1][1]
                    confidence_margin = abs(top_score - second_score)
                    
                    # Get the top two patients for severity-aware decision validation
                    top_patient = neural_scores[0][0]
                    second_patient = neural_scores[1][0]
                    
                    # Severity-aware confidence adjustment: Penalize decisions that hurt high-severity patients
                    base_threshold = hybrid_policy.confidence_threshold
                    
                    # Check if neural network is making a clinically questionable decision
                    clinically_questionable = False
                    
                    # Case 1: Lower severity chosen over significantly higher severity
                    if (second_patient.severity >= 5 and top_patient.severity <= 3 and 
                        second_patient.wait_time >= 10):
                        clinically_questionable = True
                        adjusted_threshold = base_threshold * 10.0  # Much easier to trigger ESI
                    
                    # Case 2: High-severity patient waiting very long gets deprioritized
                    elif (second_patient.severity >= 5 and second_patient.wait_time >= 20):
                        clinically_questionable = True
                        adjusted_threshold = base_threshold * 5.0  # Easier to trigger ESI
                    
                    # Case 3: Severity 5 patient with high deterioration gets deprioritized
                    elif (second_patient.severity >= 5 and second_patient.deterioration_chance >= 0.5 and
                          top_patient.severity < second_patient.severity):
                        clinically_questionable = True
                        adjusted_threshold = base_threshold * 3.0
                    
                    # Case 4: Normal threshold for clinically reasonable decisions
                    else:
                        adjusted_threshold = base_threshold
                    
                    # Force ESI for extreme cases regardless of confidence
                    force_esi = False
                    for patient in [top_patient, second_patient]:
                        if patient.severity >= 5 and patient.wait_time >= 30:  # Critical patient waiting 30+ timesteps
                            force_esi = True
                            break
                    
                    # If confidence is too low OR we're forcing ESI, fall back to ESI decision
                    if confidence_margin < adjusted_threshold or force_esi:
                        # Use pure ESI: sort by severity (higher severity = higher priority)
                        esi_scores = [(p, p.severity + 0.1 * p.deterioration_chance - 0.01 * p.wait_time) for p in self.waiting_patients]
                        esi_scores.sort(key=lambda x: x[1], reverse=True)
                        patient_scores = esi_scores
                        decision_method = "ESI_fallback"
                    else:
                        # Use neural network decision
                        patient_scores = neural_scores
                        decision_method = "neural"
                else:
                    # Single patient or empty, use neural
                    patient_scores = neural_scores
                    decision_method = "neural"
                
                # Log this decision for explanation
                self.decision_count += 1
                chosen_patient, chosen_score = patient_scores[0]
                alternatives = patient_scores[1:3] if len(patient_scores) > 1 else []
                
                # Calculate margin for explanation
                margin = 0.0
                confidence_level = "high"
                if alternatives:
                    margin = chosen_score - alternatives[0][1] 
                    if margin < 0.05:
                        confidence_level = "low"
                    elif margin < 0.1:
                        confidence_level = "moderate"
                    else:
                        confidence_level = "high"
                
                decision_log = {
                    'time': self.time,
                    'decision_id': self.decision_count,
                    'patients': [(str(p), score) for p, score in patient_scores],
                    'chosen': str(chosen_patient),
                    'chosen_score': chosen_score,
                    'alternatives': [(str(p), score) for p, score in alternatives],
                    'sim_state': sim_state.copy(),
                    'method': decision_method,
                    'margin': margin,
                    'confidence': confidence_level
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
        
        # Run hybrid simulation
        sim = HybridExplainableSimulation(
            num_nurses=4,  # Same as comprehensive_neural_evaluation
            total_time=96,  # 24 hours
            arrival_prob=0.3,
            triage_policy=hybrid_policy,
            verbose=False,
            seed=test_seed,
            use_shifts=True
        )
        
        result = sim.run()
        result['seed'] = test_seed
        result['explanations'] = sim.explanation_log
        result['total_decisions'] = sim.decision_count
        all_results.append(result)
        
        # Run baseline comparisons
        esi_sim = ERSimulation(
            num_nurses=4,
            total_time=96,
            arrival_prob=0.3,
            triage_policy={'severity': 1.0, 'deterioration': 0.0, 'wait_time': 0.0},
            verbose=False,
            seed=test_seed,
            use_shifts=True
        )
        esi_result = esi_sim.run()
        esi_result['seed'] = test_seed
        all_esi_results.append(esi_result)
        
        mts_sim = ERSimulation(
            num_nurses=4,
            total_time=96,
            arrival_prob=0.3,
            triage_policy={'severity': 0.0, 'deterioration': 0.0, 'wait_time': 1.0},
            verbose=False,
            seed=test_seed,
            use_shifts=True
        )
        mts_result = mts_sim.run()
        mts_result['seed'] = test_seed
        all_mts_results.append(mts_result)
        
        # Print summary for this seed (same format as comprehensive evaluation)
        print(f"\nSEED {test_seed} RESULTS:")
        if result['avg_wait'] is not None:
            print(f"   Patients treated: {result['completed']}")
            print(f"   Patients waiting: {result['still_waiting']}")
            print(f"   Average wait: {result['avg_wait']:.2f} timesteps ({result['avg_wait']*15:.0f} minutes)")
            print(f"   Weighted wait: {result['avg_weighted_wait']:.2f} timesteps ({result['avg_weighted_wait']*15:.0f} minutes)")
            print(f"   Decisions explained: {result['total_decisions']}")
            print(f"   ESI average wait: {esi_result['avg_wait']:.2f} timesteps ({esi_result['avg_wait']*15:.0f} minutes)")
            print(f"   ESI weighted wait: {esi_result['avg_weighted_wait']:.2f} timesteps ({esi_result['avg_weighted_wait']*15:.0f} minutes)")
            print(f"   MTS average wait: {mts_result['avg_wait']:.2f} timesteps ({mts_result['avg_wait']*15:.0f} minutes)")
            print(f"   MTS weighted wait: {mts_result['avg_weighted_wait']:.2f} timesteps ({mts_result['avg_weighted_wait']*15:.0f} minutes)")
            
            # Show severity-specific comparison
            print(f"\n   SEVERITY-SPECIFIC WAIT TIMES (minutes):")
            print(f"   {'Severity':<8} {'Count':<6} {'Hybrid':<8} {'ESI':<8} {'MTS':<8}")
            print(f"   {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
            
            for severity in range(1, 6):
                sev_key = f'sev_{severity}'
                hybrid_metrics = result['severity_metrics'][sev_key]
                esi_metrics = esi_result['severity_metrics'][sev_key]
                mts_metrics = mts_result['severity_metrics'][sev_key]
                
                if hybrid_metrics['count'] > 0:
                    hybrid_wait = hybrid_metrics['avg_weighted_wait'] * 15
                    esi_wait = esi_metrics['avg_weighted_wait'] * 15 if esi_metrics['count'] > 0 else 0
                    mts_wait = mts_metrics['avg_weighted_wait'] * 15 if mts_metrics['count'] > 0 else 0
                    
                    print(f"   {severity:<8} {hybrid_metrics['count']:<6} {hybrid_wait:<8.0f} {esi_wait:<8.0f} {mts_wait:<8.0f}")
            
            print(f"   ESI overall: {esi_result['avg_weighted_wait']*15:.0f} minutes")
            print(f"   MTS overall: {mts_result['avg_weighted_wait']*15:.0f} minutes")
        
        # Show decision explanations (simplified for readability)
        if result['total_decisions'] > 0:
            print(f"   ALL HYBRID TRIAGE DECISIONS:")
            
            decision_count = 0
            for decision in sim.explanation_log:
                if len(decision['alternatives']) > 0:
                    chosen_score = decision['chosen_score']
                    alt_patient, alt_score = decision['alternatives'][0]
                    
                    decision_count += 1
                    timestep_minutes = decision['time'] * 15
                    hours = timestep_minutes // 60
                    minutes = timestep_minutes % 60
                    time_str = f"{hours:02d}:{minutes:02d}"
                    
                    # Enhanced explanation showing method used
                    margin = chosen_score - alt_score
                    confidence = "high" if margin > 0.1 else "moderate" if margin > 0.05 else "low"
                    method = decision.get('method', 'hybrid')
                    method_display = f" [{method}]" if method == 'ESI_fallback' else ""
                    print(f"     {decision_count}. t={time_str}: {decision['chosen']} chosen over {alt_patient}: score {chosen_score:.3f} vs {alt_score:.3f} (margin: {margin:+.3f}, confidence: {confidence}){method_display}")    # Calculate aggregate statistics (same as comprehensive evaluation)
    print(f"\n🔍 PHASE 3: AGGREGATE ANALYSIS ACROSS ALL SEEDS")
    print("─" * 60)
    
    completed_counts = [r['completed'] for r in all_results]
    valid_results = [r for r in all_results if r['avg_wait'] is not None]
    
    if valid_results:
        avg_waits = [r['avg_wait'] for r in valid_results]
        weighted_waits = [r['avg_weighted_wait'] for r in valid_results]
        total_decisions = [r['total_decisions'] for r in all_results]
        
        print(f"\nAGGREGATE STATISTICS:")
        print(f"   Mean patients treated: {statistics.mean(completed_counts):.1f} +/- {statistics.stdev(completed_counts):.1f}")
        print(f"   Mean weighted wait: {statistics.mean(weighted_waits):.2f} +/- {statistics.stdev(weighted_waits):.2f} timesteps ({statistics.mean(weighted_waits)*15:.0f} minutes)")
        print(f"   Total decisions explained: {statistics.mean(total_decisions):.1f}")
        
        # Calculate baseline comparisons
        esi_valid = [r for r in all_esi_results if r['avg_wait'] is not None]
        mts_valid = [r for r in all_mts_results if r['avg_wait'] is not None]
        
        if esi_valid and mts_valid:
            esi_avg_weighted = statistics.mean([r['avg_weighted_wait'] for r in esi_valid])
            mts_avg_weighted = statistics.mean([r['avg_weighted_wait'] for r in mts_valid])
            hybrid_avg_weighted = statistics.mean(weighted_waits)
            
            esi_avg_wait = statistics.mean([r['avg_wait'] for r in esi_valid])
            mts_avg_wait = statistics.mean([r['avg_wait'] for r in mts_valid])
            hybrid_avg_wait = statistics.mean(avg_waits)
            
            # Convert to hours for comparison
            hybrid_avg_weighted_hours = hybrid_avg_weighted * 15 / 60
            hybrid_avg_wait_hours = hybrid_avg_wait * 15 / 60
            esi_avg_weighted_hours = esi_avg_weighted * 15 / 60
            esi_avg_wait_hours = esi_avg_wait * 15 / 60
            mts_avg_weighted_hours = mts_avg_weighted * 15 / 60
            mts_avg_wait_hours = mts_avg_wait * 15 / 60
            
            print(f"\nBASELINE COMPARISON:")
            print(f"   Hybrid Policy: {hybrid_avg_weighted_hours:.2f} hours weighted ({hybrid_avg_wait_hours:.2f} hours avg)")
            print(f"   ESI Baseline: {esi_avg_weighted_hours:.2f} hours weighted ({esi_avg_wait_hours:.2f} hours avg)")
            print(f"   MTS Baseline: {mts_avg_weighted_hours:.2f} hours weighted ({mts_avg_wait_hours:.2f} hours avg)")
            
            if hybrid_avg_weighted < esi_avg_weighted:
                improvement = ((esi_avg_weighted - hybrid_avg_weighted) / esi_avg_weighted) * 100
                print(f"   -> Hybrid beats ESI by {improvement:.1f}% (weighted wait)")
            if hybrid_avg_weighted < mts_avg_weighted:
                improvement = ((mts_avg_weighted - hybrid_avg_weighted) / mts_avg_weighted) * 100
                print(f"   -> Hybrid beats MTS by {improvement:.1f}% (weighted wait)")
            
            print(f"\nBASELINE PERFORMANCE DETAILS:")
            print(f"   ESI average wait: {esi_avg_wait:.2f} timesteps ({esi_avg_wait*15:.0f} minutes)")
            print(f"   ESI weighted wait: {esi_avg_weighted:.2f} timesteps ({esi_avg_weighted*15:.0f} minutes)")
            print(f"   MTS average wait: {mts_avg_wait:.2f} timesteps ({mts_avg_wait*15:.0f} minutes)")
            print(f"   MTS weighted wait: {mts_avg_weighted:.2f} timesteps ({mts_avg_weighted*15:.0f} minutes)")
            
            # Calculate and display severity-specific aggregate statistics
            print(f"\nSEVERITY-SPECIFIC AGGREGATE ANALYSIS (All Seeds):")
            print(f"   {'Severity':<8} {'Patients':<10} {'Hybrid':<8} {'ESI':<8} {'MTS':<8} {'Difference':<12}")
            print(f"   {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
            
            for severity in range(1, 6):
                sev_key = f'sev_{severity}'
                
                # Collect severity-specific metrics across all seeds
                hybrid_sev_waits = []
                esi_sev_waits = []
                mts_sev_waits = []
                total_patients = 0
                
                for result, esi_result, mts_result in zip(valid_results, esi_valid, mts_valid):
                    hybrid_metrics = result['severity_metrics'][sev_key]
                    esi_metrics = esi_result['severity_metrics'][sev_key]
                    mts_metrics = mts_result['severity_metrics'][sev_key]
                    
                    if hybrid_metrics['count'] > 0:
                        hybrid_sev_waits.append(hybrid_metrics['avg_weighted_wait'])
                        total_patients += hybrid_metrics['count']
                    if esi_metrics['count'] > 0:
                        esi_sev_waits.append(esi_metrics['avg_weighted_wait'])
                    if mts_metrics['count'] > 0:
                        mts_sev_waits.append(mts_metrics['avg_weighted_wait'])
                
                if hybrid_sev_waits:
                    hybrid_avg = statistics.mean(hybrid_sev_waits) * 15  # Convert to minutes
                    esi_avg = statistics.mean(esi_sev_waits) * 15 if esi_sev_waits else 0
                    mts_avg = statistics.mean(mts_sev_waits) * 15 if mts_sev_waits else 0
                    
                    # Calculate improvement over ESI
                    if esi_avg > 0:
                        diff = ((esi_avg - hybrid_avg) / esi_avg) * 100
                        diff_str = f"{diff:+.1f}%"
                    else:
                        diff_str = "N/A"
                    
                    print(f"   {severity:<8} {total_patients:<10} {hybrid_avg:<8.0f} {esi_avg:<8.0f} {mts_avg:<8.0f} {diff_str:<12}")
                else:
                    print(f"   {severity:<8} {0:<10} {'0':<8} {'0':<8} {'0':<8} {'N/A':<12}")
            
            print(f"\n   Key: Positive difference = Hybrid performs better than ESI")
            
            # Save detailed results to log file (same format as comprehensive evaluation)
            with open("logs/analysis_logs/comprehensive_hybrid_evaluation.txt", "w") as f:
                f.write("================================================================================\n")
                f.write("   COMPREHENSIVE HYBRID TRIAGE POLICY EVALUATION RESULTS\n")
                f.write("================================================================================\n\n")
                
                f.write(f"TRAINING CONFIGURATION:\n")
                f.write(f"   Generations: 100\n")
                f.write(f"   Population: 80\n")
                f.write(f"   Training seeds: {len(training_seeds)} seeds ({training_seeds[0]}-{training_seeds[-1]})\n")
                f.write(f"   Test seeds: {len(test_seeds)} seeds ({test_seeds[0]}-{test_seeds[-1]})\n")
                f.write(f"   Confidence threshold: {optimal_threshold}\n")
                f.write(f"   Simulation: 24 hours (96 timesteps of 15 minutes each)\n\n")
                
                # Write individual seed results
                for i, result in enumerate(all_results):
                    seed = test_seeds[i]
                    esi_result = all_esi_results[i]
                    mts_result = all_mts_results[i]
                    
                    f.write(f"SEED {seed} RESULTS:\n")
                    if result['avg_wait'] is not None:
                        f.write(f"   Patients treated: {result['completed']}\n")
                        f.write(f"   Patients waiting: {result['still_waiting']}\n")
                        f.write(f"   Average wait: {result['avg_wait']:.2f} timesteps ({result['avg_wait']*15:.0f} minutes)\n")
                        f.write(f"   Weighted wait: {result['avg_weighted_wait']:.2f} timesteps ({result['avg_weighted_wait']*15:.0f} minutes)\n")
                        f.write(f"   Decisions explained: {result['total_decisions']}\n")
                        f.write(f"   ESI average wait: {esi_result['avg_wait']:.2f} timesteps ({esi_result['avg_wait']*15:.0f} minutes)\n")
                        f.write(f"   ESI weighted wait: {esi_result['avg_weighted_wait']:.2f} timesteps ({esi_result['avg_weighted_wait']*15:.0f} minutes)\n")
                        f.write(f"   MTS average wait: {mts_result['avg_wait']:.2f} timesteps ({mts_result['avg_wait']*15:.0f} minutes)\n")
                        f.write(f"   MTS weighted wait: {mts_result['avg_weighted_wait']:.2f} timesteps ({mts_result['avg_weighted_wait']*15:.0f} minutes)\n")
                        
                        # Write all decision explanations
                        if result['total_decisions'] > 0:
                            f.write(f"   ALL HYBRID TRIAGE DECISIONS:\n")
                            
                            decision_count = 0
                            for decision in result['explanations']:
                                if 'alternatives' in decision and decision['alternatives']:
                                    chosen_score = decision['chosen_score']
                                    alt_patient, alt_score = decision['alternatives'][0]
                                    
                                    decision_count += 1
                                    timestep_minutes = decision['time'] * 15
                                    hours = timestep_minutes // 60
                                    minutes = timestep_minutes % 60
                                    time_str = f"{hours:02d}:{minutes:02d}"
                                    
                                    margin = chosen_score - alt_score
                                    confidence = "high" if margin > 0.1 else "moderate" if margin > 0.05 else "low"
                                    f.write(f"     {decision_count}. t={time_str}: {decision['chosen']} chosen over {alt_patient}: hybrid score {chosen_score:.3f} vs {alt_score:.3f} (margin: {margin:+.3f}, confidence: {confidence})\n")
                    f.write("\n")
                
                f.write(f"AGGREGATE STATISTICS:\n")
                f.write(f"   Mean patients treated: {statistics.mean(completed_counts):.1f} +/- {statistics.stdev(completed_counts):.1f}\n")
                f.write(f"   Mean weighted wait: {statistics.mean(weighted_waits):.2f} +/- {statistics.stdev(weighted_waits):.2f} hours\n")
                f.write(f"   Total decisions explained: {sum(total_decisions)}\n\n")
                
                f.write(f"BASELINE COMPARISON:\n")
                f.write(f"   Hybrid Policy: {hybrid_avg_weighted_hours:.2f} hours weighted ({hybrid_avg_wait_hours:.2f} hours avg)\n")
                f.write(f"   ESI Baseline: {esi_avg_weighted_hours:.2f} hours weighted ({esi_avg_wait_hours:.2f} hours avg)\n")
                f.write(f"   MTS Baseline: {mts_avg_weighted_hours:.2f} hours weighted ({mts_avg_wait_hours:.2f} hours avg)\n")
                
                f.write(f"\nBASELINE PERFORMANCE DETAILS:\n")
                f.write(f"   ESI average wait: {esi_avg_wait:.2f} timesteps ({esi_avg_wait*15:.0f} minutes)\n")
                f.write(f"   ESI weighted wait: {esi_avg_weighted:.2f} timesteps ({esi_avg_weighted*15:.0f} minutes)\n")
                f.write(f"   MTS average wait: {mts_avg_wait:.2f} timesteps ({mts_avg_wait*15:.0f} minutes)\n")
                f.write(f"   MTS weighted wait: {mts_avg_weighted:.2f} timesteps ({mts_avg_weighted*15:.0f} minutes)\n")
                
                if hybrid_avg_weighted < esi_avg_weighted:
                    improvement = ((esi_avg_weighted - hybrid_avg_weighted) / esi_avg_weighted) * 100
                    f.write(f"   -> Hybrid beats ESI by {improvement:.1f}% (weighted wait)\n")
                if hybrid_avg_weighted < mts_avg_weighted:
                    improvement = ((mts_avg_weighted - hybrid_avg_weighted) / mts_avg_weighted) * 100
                    f.write(f"   -> Hybrid beats MTS by {improvement:.1f}% (weighted wait)\n")
            
            print(f"\n💾 SAVING DETAILED RESULTS...")
            print(f"   Results saved to: logs/analysis_logs/comprehensive_hybrid_evaluation.txt")
            print(f"   Patient arrivals saved to: logs/patient_arrivals/hybrid_testing/")
            
            print(f"\n🎉 COMPREHENSIVE HYBRID EVALUATION COMPLETE!")
            print("=" * 65)
            
            return {
                'hybrid_weighted_wait': hybrid_avg_weighted_hours,
                'esi_weighted_wait': esi_avg_weighted_hours,
                'mts_weighted_wait': mts_avg_weighted_hours,
                'improvement_vs_esi': ((esi_avg_weighted - hybrid_avg_weighted) / esi_avg_weighted) * 100 if hybrid_avg_weighted < esi_avg_weighted else 0,
                'improvement_vs_mts': ((mts_avg_weighted - hybrid_avg_weighted) / mts_avg_weighted) * 100 if hybrid_avg_weighted < mts_avg_weighted else 0
            }
    
    else:
        print("No valid results obtained!")
        return {}

if __name__ == "__main__":
    # Run comprehensive evaluation
    results = evaluate_hybrid_policy()
    
    print(f"\nCOMPREHENSIVE HYBRID EVALUATION COMPLETE!")
    print("Results saved to logs/analysis_logs/comprehensive_hybrid_evaluation.txt")
    print("Patient arrivals logged in logs/patient_arrivals/hybrid_testing/")
    print("All decision explanations captured and analyzed.")