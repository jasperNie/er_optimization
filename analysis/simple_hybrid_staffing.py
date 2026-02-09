#!/usr/bin/env python3
"""
Simple Hybrid Staffing Test - Just run proper_hybrid_evaluation.py 5 times with different nurse counts
"""

import subprocess
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

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
    
    # Update chart output directory to be unique for this scenario
    modified_script = modified_script.replace(
        '"report_visualizations/hybrid_evaluation"',
        f'"report_visualizations/hybrid_evaluation_{nurse_count}nurses"'
    )
    
    # Write temporary script
    temp_script = f"temp_hybrid_{scenario_name.lower()}.py"
    with open(temp_script, 'w') as f:
        f.write(modified_script)
    
    try:
        # Run the modified script with real-time output
        result = subprocess.run([sys.executable, temp_script], cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"{scenario_name} completed successfully!")
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
    generated_logs = []
    if log_files:
        print("Generated analysis logs:")
        for log_file in sorted(log_files):
            print(f"  {log_file}")
            generated_logs.append(log_file)
    
    # Generate hybrid staffing comparison visualization
    print("\nGENERATING HYBRID STAFFING VISUALIZATION...")
    
    # Parse results from log files
    staffing_data = []
    fallback_data = []
    scenario_names = ['Understaffed (2)', 'Standard (3)', 'Original (4)', 'Well Staffed (5)', 'Extra Staffed (6)']
    
    for i, log_file in enumerate(generated_logs):
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                
                nurses = i + 2  # Extract nurse count from index
                
                # Extract hybrid performance and fallback data
                for line in lines:
                    if 'Hybrid Policy:' in line and 'hours weighted' in line:
                        try:
                            hours_str = line.split('hours weighted')[0].split(':')[-1].strip()
                            hours = float(hours_str)
                            staffing_data.append({'scenario': scenario_names[i], 'nurses': nurses, 'wait_time': hours})
                        except:
                            pass
                    elif 'Neural decisions:' in line and '(' in line and '%' in line:
                        try:
                            # Extract percentage from "Neural decisions: X/Y (Z.Z%)"
                            pct_str = line.split('(')[1].split('%')[0]
                            neural_pct = float(pct_str)
                            fallback_pct = 100 - neural_pct
                            fallback_data.append({'nurses': nurses, 'neural_pct': neural_pct, 'fallback_pct': fallback_pct})
                        except:
                            pass
        except:
            pass
    
    if staffing_data and fallback_data:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Hybrid performance by staffing
        nurses_list = [d['nurses'] for d in staffing_data]
        wait_times = [d['wait_time'] for d in staffing_data]
        colors = plt.cm.plasma(np.linspace(0, 1, len(nurses_list)))
        
        bars1 = ax1.bar(nurses_list, wait_times, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Hybrid System Performance\nby Staffing Level', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Nurses')
        ax1.set_ylabel('Weighted Wait Time (Hours)')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_xticks(nurses_list)
        
        for bar, time in zip(bars1, wait_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{time:.1f}h', ha='center', fontweight='bold')
        
        # 2. ESI fallback usage by staffing
        fallback_nurses = [d['nurses'] for d in fallback_data]
        fallback_pcts = [d['fallback_pct'] for d in fallback_data]
        
        ax2.plot(fallback_nurses, fallback_pcts, 'o-', linewidth=3, markersize=8, color='#E67E22')
        ax2.set_title('ESI Fallback Usage\nby Staffing Level', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Nurses')
        ax2.set_ylabel('ESI Fallback Percentage (%)')
        ax2.grid(alpha=0.3)
        ax2.set_xticks(fallback_nurses)
        
        for x, y in zip(fallback_nurses, fallback_pcts):
            ax2.text(x, y + 0.2, f'{y:.1f}%', ha='center', fontweight='bold')
        
        # 3. Decision method stacked bar
        neural_pcts = [d['neural_pct'] for d in fallback_data]
        width = 0.6
        
        bars3_neural = ax3.bar(fallback_nurses, neural_pcts, width, label='Neural Network', 
                              color='#27AE60', alpha=0.8)
        bars3_esi = ax3.bar(fallback_nurses, fallback_pcts, width, bottom=neural_pcts, 
                           label='ESI Fallback', color='#E74C3C', alpha=0.8)
        
        ax3.set_title('Decision Method Distribution\nby Staffing Level', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Number of Nurses')
        ax3.set_ylabel('Decision Percentage (%)')
        ax3.set_xticks(fallback_nurses)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Efficiency comparison
        if len(wait_times) > 1:
            efficiency = [(wait_times[0] - wt) / wait_times[0] * 100 for wt in wait_times]
            ax4.plot(nurses_list, efficiency, 'o-', linewidth=3, markersize=8, color='#8E44AD')
            ax4.set_title('Staffing Efficiency\n(% Improvement from 2 Nurses)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Number of Nurses')
            ax4.set_ylabel('Wait Time Reduction (%)')
            ax4.grid(alpha=0.3)
            ax4.set_xticks(nurses_list)
            
            for i, (x, y) in enumerate(zip(nurses_list, efficiency)):
                ax4.text(x, y + 2, f'{y:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        # Chart generation removed
        plt.close()
    else:
        print("Could not parse hybrid staffing data for visualization")

if __name__ == "__main__":
    main()