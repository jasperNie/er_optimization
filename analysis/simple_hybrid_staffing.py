#!/usr/bin/env python3
"""
Simple Hybrid Staffing Test - Just run proper_hybrid_evaluation.py 5 times with different nurse counts
"""

import subprocess
import sys
import os

def run_hybrid_with_nurse_count(nurse_count, scenario_name):
    """Run the original proper_hybrid_evaluation.py with a specific nurse count"""
    
    print(f"="*80)
    print(f"RUNNING HYBRID EVALUATION - {scenario_name} ({nurse_count} nurses)")
    print(f"="*80)
    
    # Read the original proper_hybrid_evaluation.py
    script_path = "analysis/proper_hybrid_evaluation.py"
    with open(script_path, 'r') as f:
        original_script = f.read()
    
    # Replace the nurse count (default is 4) - handle all variants
    modified_script = original_script.replace('num_nurses=4', f'num_nurses={nurse_count}')
    modified_script = modified_script.replace('num_nurses = 4', f'num_nurses = {nurse_count}')
    # Handle the dictionary entry format with spaces  
    modified_script = modified_script.replace("'num_nurses': 4", f"'num_nurses': {nurse_count}")
    # CRITICAL: Replace hardcoded ERSimulation call for nurse schedule generation
    modified_script = modified_script.replace('ERSimulation(4, 96, 0.3, 9000)', f'ERSimulation({nurse_count}, 96, 0.3, 9000)')
    
    # Change the output filename to include scenario
    modified_script = modified_script.replace(
        '"logs/analysis_logs/comprehensive_hybrid_evaluation.txt"',
        f'"logs/analysis_logs/hybrid_staffing_{scenario_name.lower()}_{nurse_count}nurses.txt"'
    )
    
    # Also replace any print statements mentioning the filename
    modified_script = modified_script.replace(
        'logs/analysis_logs/comprehensive_hybrid_evaluation.txt',
        f'logs/analysis_logs/hybrid_staffing_{scenario_name.lower()}_{nurse_count}nurses.txt'
    )
    
    # Disable patient arrival logging to avoid duplicates - handle ALL patterns
    modified_script = modified_script.replace(
        'log_patient_arrivals(arrivals, seed, "logs/patient_arrivals"',
        '# log_patient_arrivals(arrivals, seed, "logs/patient_arrivals"'
    )
    # Disable all print statements about patient arrival logging
    modified_script = modified_script.replace(
        'print("   Training patient arrival logs saved',
        '# print("   Training patient arrival logs saved'
    )
    modified_script = modified_script.replace(
        'print("   Testing patient arrival logs saved',
        '# print("   Testing patient arrival logs saved'
    )
    modified_script = modified_script.replace(
        'print("Patient arrivals logged in logs/patient_arrivals',
        '# print("Patient arrivals logged in logs/patient_arrivals'
    )
    
    # Update nurse schedule filename to be unique for this scenario
    modified_script = modified_script.replace(
        '"logs/nurse_schedules/nurse_schedule_base_4_nurses_original.txt"',
        f'"logs/nurse_schedules/nurse_schedule_base_{nurse_count}_nurses_{scenario_name.lower()}.txt"'
    )
    # Also replace the print statement for nurse schedule path
    modified_script = modified_script.replace(
        'logs/nurse_schedules/nurse_schedule_base_4_nurses_original.txt',
        f'logs/nurse_schedules/nurse_schedule_base_{nurse_count}_nurses_{scenario_name.lower()}.txt'
    )
    
    # Write temporary script
    temp_script = f"temp_hybrid_{scenario_name.lower()}.py"
    with open(temp_script, 'w') as f:
        f.write(modified_script)
    
    try:
        # Run the modified script with real-time output
        result = subprocess.run([sys.executable, temp_script], cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"✅ {scenario_name} completed successfully!")
        else:
            print(f"❌ {scenario_name} failed with exit code {result.returncode}")
            
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script):
            os.remove(temp_script)
    
    return result.returncode == 0

def main():
    """Run hybrid evaluation with different staffing levels"""
    
    print("SIMPLE HYBRID NEURAL ALGORITHM STAFFING VARIATION TEST")
    print("Running proper_hybrid_evaluation.py with different nurse counts")
    print("="*80)
    
    staffing_tests = [
        (2, "Understaffed"),
        (3, "Standard"),      
        (4, "Original"),      
        (5, "Well_Staffed"),
        (6, "Extra_Staffed")
    ]
    
    successful_runs = 0
    total_runs = len(staffing_tests)
    
    for nurses, scenario in staffing_tests:
        success = run_hybrid_with_nurse_count(nurses, scenario)
        if success:
            successful_runs += 1
        print()
    
    print("="*80)
    print(f"SUMMARY: {successful_runs}/{total_runs} scenarios completed successfully")
    print("="*80)
    
    # List the generated log files
    import glob
    log_files = glob.glob("logs/analysis_logs/hybrid_staffing_*.txt")
    if log_files:
        print("Generated analysis logs:")
        for log_file in sorted(log_files):
            print(f"  {log_file}")

if __name__ == "__main__":
    main()