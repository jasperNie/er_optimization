import re
import matplotlib.pyplot as plt
import numpy as np

# Path to the analysis log file
log_path = 'logs/complete_evaluation/neural/standard_4nurses_analysis.txt'

with open(log_path, 'r') as f:
    log = f.read()

# Find the first SEED block
seed_block = re.search(r'SEED \d+ RESULTS:(.*?)(?:SEED \d+ RESULTS:|\Z)', log, re.DOTALL)
if not seed_block:
    raise ValueError('No SEED block found in log file.')
block = seed_block.group(1)

# Extract Neural (main) average wait and weighted wait
neural_avg_wait = None
neural_weighted_wait = None
m = re.search(r'Average wait: [\d.]+ timesteps \((\d+) minutes\)', block)
if m:
    neural_avg_wait = float(m.group(1))
m = re.search(r'Weighted wait: [\d.]+ timesteps \((\d+) minutes\)', block)
if m:
    neural_weighted_wait = float(m.group(1))

# Extract ESI and MTS average wait and weighted wait
esi_avg_wait = esi_weighted_wait = None
mts_avg_wait = mts_weighted_wait = None
m = re.search(r'ESI treated:.*?avg: [\d.]+ timesteps \((\d+) min\), weighted: [\d.]+ \((\d+) min\)', block)
if m:
    esi_avg_wait = float(m.group(1))
    esi_weighted_wait = float(m.group(2))
m = re.search(r'MTS treated:.*?avg: [\d.]+ timesteps \((\d+) min\), weighted: [\d.]+ \((\d+) min\)', block)
if m:
    mts_avg_wait = float(m.group(1))
    mts_weighted_wait = float(m.group(2))

plot_labels = ['Neural Network', 'ESI (Severity)', 'MTS (Wait Time)']
avg_waits = [neural_avg_wait, esi_avg_wait, mts_avg_wait]
avg_weighted_waits = [neural_weighted_wait, esi_weighted_wait, mts_weighted_wait]

x = np.arange(len(plot_labels))
width = 0.35

plt.figure(figsize=(8, 6))
bar1 = plt.bar(x - width/2, avg_waits, width, label='Avg Wait (min)', color='#3CB371', alpha=0.85)
bar2 = plt.bar(x + width/2, avg_weighted_waits, width, label='Avg Weighted Wait (min)', color='#DC143C', alpha=0.85)

plt.title('Policy Comparison: Wait Times\n(Standard Pattern, 4 Nurses)', fontsize=15)
plt.xlabel('Triage Policy', fontsize=12)
plt.xticks(x, plot_labels, fontsize=11)
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bar1, bar2]:
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
output_path = 'policy_comparison_from_log.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
plt.show()
