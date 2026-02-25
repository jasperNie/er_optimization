# Helper to track cumulative patients treated over time
def get_cumulative_patients_treated(sim, total_time):
    sim.time = 0
    sim.waiting_patients = []
    sim.completed_patients = []
    sim.started_patients = []
    cumulative = []
    treated_set = set()
    for t in range(total_time):
        # Add any patients just completed this step
        for nurse in getattr(sim, 'base_nurses', []):
            if nurse.current_patient is None and hasattr(nurse, 'last_completed') and nurse.last_completed is not None:
                treated_set.add(nurse.last_completed)
        # Count all completed patients so far
        cumulative.append(len(sim.completed_patients))
        sim.step()
    return cumulative
#!/usr/bin/env python3
"""
Train a neural network on standard pattern, test on a single seed, and plot queue lengths over time.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimizers.neural_optimizer import FairNeuralEvolutionOptimizer, create_fair_neural_triage_function
from classes import ERSimulation
from arrival_patterns import ARRIVAL_PATTERNS
from triage_policies import esi_policy, mts_policy


TRAINING_SEEDS = list(range(8000, 8050))  # 50 seeds for training
TEST_SEED = 9000  # Single test seed
NUM_NURSES = 4

# Helper to track cumulative weighted wait over time
def get_cumulative_weighted_wait(sim, total_time):
    sim.time = 0
    sim.waiting_patients = []
    sim.completed_patients = []
    sim.started_patients = []
    cumulative = []
    for t in range(total_time):
        # Update wait time for all waiting patients
        for p in sim.waiting_patients:
            p.wait_time += 1
        # Calculate weighted wait for all patients (treated + waiting)
        all_patients = []
        if sim.started_patients:
            all_patients.extend(sim.started_patients)
        for patient in sim.waiting_patients:
            all_patients.append((patient, patient.wait_time))
        weighted_wait = sum([wait * p.severity * (1 + p.deterioration_chance) for p, wait in all_patients])
        cumulative.append(weighted_wait * 15)  # convert to minutes
        sim.step()
    return cumulative

TOTAL_TIME = 96
ARRIVAL_PROB = 0.3
PATTERN = 'standard'

# Train neural network
print(f"Training neural network on {PATTERN} pattern...")

optimizer = FairNeuralEvolutionOptimizer(
    num_generations=100,
    population_size=80,
    num_nurses=NUM_NURSES,
    total_time=TOTAL_TIME,
    arrival_prob=ARRIVAL_PROB,
    seed=TRAINING_SEEDS[0]
)
training_log_path = f"logs/complete_evaluation/neural/quick_train_{PATTERN}.txt"
neural_policy = optimizer.run(training_log_path)
triage_func = create_fair_neural_triage_function(neural_policy, optimizer)


# Generate arrivals for test seed
arrivals = ARRIVAL_PATTERNS[PATTERN](TOTAL_TIME, ARRIVAL_PROB, TEST_SEED)

# Run neural net policy
# Run neural net policy
test_sim = ERSimulation(
    num_nurses=NUM_NURSES,
    total_time=TOTAL_TIME,
    arrival_prob=ARRIVAL_PROB,
    triage_policy=triage_func,
    verbose=False,
    seed=TEST_SEED,
    use_shifts=True
)
test_sim.patient_arrivals = arrivals
results_neural = test_sim.run()
weighted_wait_neural = results_neural['avg_weighted_wait'] * 15 if results_neural['avg_weighted_wait'] is not None else 0  # convert to minutes
cum_weighted_neural = get_cumulative_weighted_wait(test_sim, TOTAL_TIME)

# Cumulative patients treated for neural
test_sim2 = ERSimulation(
    num_nurses=NUM_NURSES,
    total_time=TOTAL_TIME,
    arrival_prob=ARRIVAL_PROB,
    triage_policy=triage_func,
    verbose=False,
    seed=TEST_SEED,
    use_shifts=True
)
test_sim2.patient_arrivals = arrivals
cum_treated_neural = get_cumulative_patients_treated(test_sim2, TOTAL_TIME)

# Run ESI policy
# Run ESI policy
esi_sim = ERSimulation(
    num_nurses=NUM_NURSES,
    total_time=TOTAL_TIME,
    arrival_prob=ARRIVAL_PROB,
    triage_policy=esi_policy,
    verbose=False,
    seed=TEST_SEED,
    use_shifts=True
)
esi_sim.patient_arrivals = arrivals.copy()
results_esi = esi_sim.run()
weighted_wait_esi = results_esi['avg_weighted_wait'] * 15 if results_esi['avg_weighted_wait'] is not None else 0  # convert to minutes
cum_weighted_esi = get_cumulative_weighted_wait(esi_sim, TOTAL_TIME)

# Cumulative patients treated for ESI
esi_sim2 = ERSimulation(
    num_nurses=NUM_NURSES,
    total_time=TOTAL_TIME,
    arrival_prob=ARRIVAL_PROB,
    triage_policy=esi_policy,
    verbose=False,
    seed=TEST_SEED,
    use_shifts=True
)
esi_sim2.patient_arrivals = arrivals.copy()
cum_treated_esi = get_cumulative_patients_treated(esi_sim2, TOTAL_TIME)

# Run MTS policy
# Run MTS policy
mts_sim = ERSimulation(
    num_nurses=NUM_NURSES,
    total_time=TOTAL_TIME,
    arrival_prob=ARRIVAL_PROB,
    triage_policy=mts_policy,
    verbose=False,
    seed=TEST_SEED,
    use_shifts=True
)
mts_sim.patient_arrivals = arrivals.copy()
results_mts = mts_sim.run()
weighted_wait_mts = results_mts['avg_weighted_wait'] * 15 if results_mts['avg_weighted_wait'] is not None else 0  # convert to minutes
cum_weighted_mts = get_cumulative_weighted_wait(mts_sim, TOTAL_TIME)

# Cumulative patients treated for MTS
mts_sim2 = ERSimulation(
    num_nurses=NUM_NURSES,
    total_time=TOTAL_TIME,
    arrival_prob=ARRIVAL_PROB,
    triage_policy=mts_policy,
    verbose=False,
    seed=TEST_SEED,
    use_shifts=True
)
mts_sim2.patient_arrivals = arrivals.copy()
cum_treated_mts = get_cumulative_patients_treated(mts_sim2, TOTAL_TIME)


# Plot cumulative weighted wait time over time
plt.figure(figsize=(12, 7))
timesteps = range(1, TOTAL_TIME+1)
plt.plot(timesteps, cum_weighted_neural, label='Neural Network', color='blue', linewidth=2)
plt.plot(timesteps, cum_weighted_esi, label='ESI (Severity)', color='orange', linewidth=2)
plt.plot(timesteps, cum_weighted_mts, label='MTS (Wait Time)', color='green', linewidth=2)
plt.title(f'Cumulative Weighted Wait Time Over Time\n({PATTERN.replace("_", " ").title()} Pattern, Seed {TEST_SEED}, {NUM_NURSES} Nurses)', fontsize=15)
plt.xlabel('Timestep (15 min each)', fontsize=12)
plt.ylabel('Cumulative Weighted Wait (minutes)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot to the main directory
output_path = os.path.join(os.path.dirname(__file__), 'cumulative_weighted_wait.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
plt.show()


