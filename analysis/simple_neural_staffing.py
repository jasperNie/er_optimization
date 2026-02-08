#!/usr/bin/env python3
"""
Simple Neural Staffing Evaluation
Runs comprehensive_neural_evaluation.py with different nurse counts to test staffing variations.
"""

import subprocess
import sys
from pathlib import Path

def run_neural_with_nurses(nurse_count, scenario_name):
    """Run comprehensive_neural_evaluation.py with specified nurse count"""
    
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name} ({nurse_count} nurses)")
    print(f"{'='*60}")
    
    # Read the original script
    script_path = Path(__file__).parent / "comprehensive_neural_evaluation.py"
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Replace nurse count in the script content - handle all variants
    modified_content = script_content.replace('num_nurses=4', f'num_nurses={nurse_count}')
    modified_content = modified_content.replace('num_nurses = 4', f'num_nurses = {nurse_count}')
    # Handle the dictionary entry format with spaces
    modified_content = modified_content.replace("'num_nurses': 4", f"'num_nurses': {nurse_count}")
    # CRITICAL: Replace hardcoded ERSimulation call for nurse schedule generation
    modified_content = modified_content.replace('ERSimulation(4, 96, 0.3, 9000)', f'ERSimulation({nurse_count}, 96, 0.3, 9000)')
    
    # Also update the output filename to be unique for this scenario  
    original_filename = 'logs/analysis_logs/comprehensive_neural_evaluation.txt'
    new_filename = f'logs/analysis_logs/neural_staffing_{scenario_name.lower()}_{nurse_count}nurses.txt'
    # Replace both quoted and unquoted versions
    modified_content = modified_content.replace(f'"{original_filename}"', f'"{new_filename}"')
    modified_content = modified_content.replace(original_filename, new_filename)
    
    # Disable patient arrival logging to avoid duplicates - handle ALL patterns
    modified_content = modified_content.replace(
        'log_patient_arrivals(arrivals, seed, "logs/patient_arrivals"',
        '# log_patient_arrivals(arrivals, seed, "logs/patient_arrivals"'
    )
    modified_content = modified_content.replace(
        'log_patient_arrivals(arrivals, test_seed, "logs/patient_arrivals"',
        '# log_patient_arrivals(arrivals, test_seed, "logs/patient_arrivals"'
    )
    # Disable all print statements about patient arrival logging
    modified_content = modified_content.replace(
        'print("   Training patient arrival logs saved',
        '# print("   Training patient arrival logs saved'
    )
    modified_content = modified_content.replace(
        'print("   Testing patient arrival logs saved',
        '# print("   Testing patient arrival logs saved'
    )
    modified_content = modified_content.replace(
        'print("   Patient arrival logs saved',
        '# print("   Patient arrival logs saved'
    )
    modified_content = modified_content.replace(
        'print(f"   📋 Patient arrivals saved',
        '# print(f"   📋 Patient arrivals saved'
    )
    
    # Update nurse schedule filename to be unique for this scenario
    modified_content = modified_content.replace(
        '"logs/nurse_schedules/nurse_schedule_base_4_nurses_original.txt"',
        f'"logs/nurse_schedules/nurse_schedule_base_{nurse_count}_nurses_{scenario_name.lower()}.txt"'
    )
    # Also replace the print statement for nurse schedule path
    modified_content = modified_content.replace(
        'logs/nurse_schedules/nurse_schedule_base_4_nurses_original.txt',
        f'logs/nurse_schedules/nurse_schedule_base_{nurse_count}_nurses_{scenario_name.lower()}.txt'
    )
    
    # Update training log filename to be unique for this scenario
    modified_content = modified_content.replace(
        '"logs/full_training_comprehensive.txt"',
        f'"logs/neural_train_{scenario_name.lower()}.txt"'
    )
    
    # Write to temporary file
    temp_script = Path(__file__).parent / f"temp_neural_{nurse_count}nurses.py"
    with open(temp_script, 'w') as f:
        f.write(modified_content)
    
    try:
        # Run the modified script
        result = subprocess.run([
            sys.executable, str(temp_script)
        ], cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print(f"✅ {scenario_name} completed successfully!")
        else:
            print(f"❌ {scenario_name} failed with return code {result.returncode}")
            
    except Exception as e:
        print(f"❌ Error running {scenario_name}: {e}")
    finally:
        # Clean up temporary file
        if temp_script.exists():
            temp_script.unlink()

def main():
    """Run neural staffing evaluation with different nurse counts"""
    
    print("Starting Neural Staffing Evaluation...")
    print("Testing neural network algorithm with 5 different staffing levels")
    
    # Define scenarios: (nurse_count, scenario_name)
    scenarios = [
        (2, "Understaffed"),
        (3, "Standard"), 
        (4, "Original"),
        (5, "Well_Staffed"),
        (6, "Extra_Staffed")
    ]
    
    completed = 0
    failed = 0
    
    for nurse_count, scenario_name in scenarios:
        try:
            run_neural_with_nurses(nurse_count, scenario_name)
            completed += 1
        except Exception as e:
            print(f"❌ Failed {scenario_name}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {completed}/{len(scenarios)} scenarios completed successfully")
    if failed > 0:
        print(f"❌ {failed} scenarios failed")
    print(f"{'='*60}")
    
    print("\nGenerated analysis logs:")
    logs_dir = Path(__file__).parent.parent / "logs" / "analysis_logs"
    for nurse_count, scenario_name in scenarios:
        log_file = logs_dir / f"neural_staffing_{scenario_name.lower()}_{nurse_count}nurses.txt"
        if log_file.exists():
            print(f"  {log_file}")

if __name__ == "__main__":
    main()