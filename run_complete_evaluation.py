#!/usr/bin/env python3
"""
Complete ER Optimization Evaluation
Runs all permutations of neural and hybrid evaluations across patterns and nurse counts
"""

import os
import sys
import subprocess
import time
import re
from arrival_patterns import ARRIVAL_PATTERNS

def run_evaluation(script_type, pattern, num_nurses):
    """Run a single evaluation and return success status"""
    script_name = f"comprehensive_{script_type}_pattern_evaluation.py"
    script_path = f"analysis/{script_name}"
    
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_type.upper()} - {pattern} pattern - {num_nurses} nurses")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script_path, pattern, str(num_nurses)],
            cwd=os.getcwd(),
            # Remove capture_output=True to show real-time progress
            timeout=1800  # 30 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"SUCCESS in {duration:.1f}s")
            return True, duration, ""
        else:
            print(f"FAILED in {duration:.1f}s")
            return False, duration, "Process failed"
            
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after 30 minutes")
        return False, 1800, "Timeout after 30 minutes"
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
        return False, 0, str(e)

def extract_results_from_log(script_type, pattern, num_nurses):
    """Extract key metrics from the analysis log file"""
    log_path = f"logs/complete_evaluation/{script_type}/{pattern}_{num_nurses}nurses_analysis.txt"
    
    if not os.path.exists(log_path):
        return None
        
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract key metrics using regex
        results = {}
        
        # Extract baseline comparison section
        baseline_section = re.search(r'BASELINE COMPARISON:(.*?)(?=\n\w|\Z)', content, re.DOTALL)
        if baseline_section:
            baseline_text = baseline_section.group(1)
            
            # Extract neural/hybrid performance
            if script_type == 'neural':
                neural_match = re.search(r'Neural Network: ([\d.]+) hours weighted \(([\d.]+) hours avg\)', baseline_text)
            else:
                neural_match = re.search(r'Hybrid Network: ([\d.]+) hours weighted \(([\d.]+) hours avg\)', baseline_text)
                # Also extract neural decision rate for hybrid
                decision_rate_match = re.search(r'Neural decision rate: ([\d.]+)%', baseline_text)
                results['neural_decision_rate'] = float(decision_rate_match.group(1)) if decision_rate_match else 0.0
            
            if neural_match:
                results['weighted_wait_hours'] = float(neural_match.group(1))
                results['avg_wait_hours'] = float(neural_match.group(2))
            
            # Extract ESI performance
            esi_match = re.search(r'ESI Baseline:\s+([\d.]+) hours weighted \(([\d.]+) hours avg\)', baseline_text)
            if esi_match:
                results['esi_weighted_hours'] = float(esi_match.group(1))
                results['esi_avg_hours'] = float(esi_match.group(2))
            
            # Extract MTS performance
            mts_match = re.search(r'MTS Baseline: ([\d.]+) hours weighted \(([\d.]+) hours avg\)', baseline_text)
            if mts_match:
                results['mts_weighted_hours'] = float(mts_match.group(1))
                results['mts_avg_hours'] = float(mts_match.group(2))
            
            # Extract improvement percentages
            esi_improvement = re.search(r'-> Neural beats ESI by ([\d.]+)% \(weighted wait\)', baseline_text)
            if esi_improvement:
                results['esi_improvement'] = float(esi_improvement.group(1))
                
            mts_improvement = re.search(r'-> Neural beats MTS by ([\d.]+)% \(weighted wait\)', baseline_text)
            if mts_improvement:
                results['mts_improvement'] = float(mts_improvement.group(1))
        
        # Extract performance statistics
        perf_section = re.search(r'PERFORMANCE STATISTICS.*?PATTERN:(.*?)DECISION PATTERN ANALYSIS:', content, re.DOTALL)
        if perf_section:
            perf_text = perf_section.group(1)
            
            patients_match = re.search(r'Patients treated: ([\d.]+) ± ([\d.]+)', perf_text)
            if patients_match:
                results['patients_treated_mean'] = float(patients_match.group(1))
                results['patients_treated_std'] = float(patients_match.group(2))
        
        return results
        
    except Exception as e:
        print(f"Error extracting results from {log_path}: {str(e)}")
        return None

def main():
    """Main evaluation runner"""
    print("STARTING COMPLETE ER OPTIMIZATION EVALUATION")
    print("=" * 80)
    
    # Configuration
    patterns = list(ARRIVAL_PATTERNS.keys())  # All 6 patterns
    nurse_counts = [2, 3, 4, 5, 6]  # 5 different nurse counts
    script_types = ['neural', 'hybrid']  # 2 algorithms
    
    total_evaluations = len(patterns) * len(nurse_counts) * len(script_types)
    print(f"Patterns: {patterns}")
    print(f"Nurse counts: {nurse_counts}")
    print(f"Algorithms: {script_types}")
    print(f"Total evaluations to run: {total_evaluations}")
    print(f"Estimated time: ~{total_evaluations * 3} minutes")
    
    # Create output directory
    output_dir = "logs/complete_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Track all results
    all_results = {}
    completed = 0
    
    # Run all evaluations
    for script_type in script_types:
        for pattern in patterns:
            for num_nurses in nurse_counts:
                completed += 1
                key = (script_type, pattern, num_nurses)
                
                print(f"\n[{completed}/{total_evaluations}] Running {script_type} - {pattern} - {num_nurses} nurses...")
                
                success, duration, error = run_evaluation(script_type, pattern, num_nurses)
                
                # Extract results if successful
                metrics = None
                if success:
                    metrics = extract_results_from_log(script_type, pattern, num_nurses)
                    if metrics:
                        print(f"   📊 Extracted metrics: {metrics.get('weighted_wait_hours', 0):.1f}h weighted wait")
                
                all_results[key] = {
                    'success': success,
                    'duration': duration,
                    'error': error,
                    'metrics': metrics
                }
    
    # Print summary to console
    print(f"\n🏆 EVALUATION COMPLETE!")
    print("=" * 80)
    successful_runs = sum(1 for result in all_results.values() if result['success'])
    print(f"Successful evaluations: {successful_runs}/{total_evaluations}")
    print(f"Failed evaluations: {total_evaluations - successful_runs}/{total_evaluations}")
    
    print(f"\n📁 Individual results saved to: logs/complete_evaluation/")

if __name__ == "__main__":
    main()