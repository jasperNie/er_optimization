# Main script that runs the same exact functionality as the original script
import random
import math
from classes import ERSimulation, Patient
from evolutionary_optimizer import EvolutionaryTriageOptimizer

# Run evolutionary optimizer (same as original)
optimizer = EvolutionaryTriageOptimizer(num_generations=100, population_size=100, num_nurses=3, total_time=96, arrival_prob=0.5, seed=2025)
best_policy = optimizer.run(gen_log_path="generation_log.txt")

# --- MTS and ESI comparison (same as original) ---
def mts_policy(patient):
    if patient is None:
        return 0
    v = getattr(patient, 'vitals', {})
    if v.get('spo2', 100) < 85 or v.get('hr', 0) > 140 or v.get('bp_sys', 0) < 70:
        return 5
    if hasattr(patient, 'presenting') and patient.presenting in ('shortness_of_breath', 'chest_pain'):
        if v.get('spo2', 100) < 92 or v.get('hr', 0) > 110:
            return 4
        return 3
    if hasattr(patient, 'presenting') and patient.presenting in ('abdominal_pain',):
        return 3
    if hasattr(patient, 'presenting') and patient.presenting in ('minor_injury', 'headache'):
        return 2
    return 1

def esi_policy(patient):
    if patient is None:
        return 0
    v = getattr(patient, 'vitals', {})
    if v.get('spo2', 100) < 85 or v.get('hr', 0) > 140 or v.get('bp_sys', 0) < 70:
        return 5
    if v.get('spo2', 100) < 92 or v.get('hr', 0) > 120 or v.get('rr', 0) > 25:
        return 4
    resources = getattr(patient, 'expected_resources', 1)
    if resources == 0:
        return 1
    if resources == 1:
        return 2
    if resources >= 2:
        if v.get('hr', 0) > 120 or v.get('rr', 0) > 24 or v.get('spo2', 100) < 92:
            return 4
        return 3

# Generate arrivals ONCE for all policies (same as original)
random.seed(2025)
arrivals = []
patient_id = 0
total_time = optimizer.total_time if 'optimizer' in locals() else 96
timesteps_per_day = 96  # 24 hours * 4 timesteps per hour
for t in range(total_time):
    base_prob = 0.3
    time_factor = 0.15 * (1 + math.sin(2 * math.pi * t / timesteps_per_day))
    burst = 0.15 if random.random() < 0.05 else 0
    prob = min(1.0, max(0.0, base_prob + time_factor + burst))
    if random.random() < prob:
        severity = random.choices([1,2,3,4,5], weights=[0.1,0.15,0.25,0.25,0.25])[0]
        det_base = 0.1 + 0.15 * (severity-1)
        deterioration = round(random.uniform(det_base, min(0.6, det_base+0.15)), 2)
        treat_base = 5 + 2 * severity
        treatment_time = random.randint(treat_base, treat_base+5)
        presenting = random.choice(['chest_pain', 'abdominal_pain', 'shortness_of_breath', 'minor_injury', 'headache'])
        vitals = {
            'hr': random.randint(50, 130),
            'rr': random.randint(12, 30),
            'spo2': random.randint(88, 100),
            'bp_sys': random.randint(90, 160)
        }
        expected_resources = random.choice([0, 1, 2, 3])
        p = Patient(patient_id, severity, deterioration, treatment_time)
        patient_id += 1
        p.presenting = presenting
        p.vitals = vitals
        p.expected_resources = expected_resources
        arrivals.append(p)
    else:
        arrivals.append(None)

# Run all three policies on the same arrivals (same as original)
for name, policy, verbose in [
    ('Optimized', best_policy, True),
    ('MTS', mts_policy, False),
    ('ESI', esi_policy, False)
]:
    sim = ERSimulation(num_nurses=4, total_time=total_time, arrival_prob=0, triage_policy=policy, verbose=False, seed=2025, use_shifts=True)
    sim.patient_arrivals = arrivals.copy()
    if verbose:
        sim.print_arrivals(file_path="arrivals_log.txt")
        sim.print_nurse_schedule(file_path="nurse_schedule_log.txt")
    metrics = sim.run()
    print(f"\n{name} policy metrics:")
    if name == 'Optimized':
        print("Triage policy weights used in cost function:")
        if isinstance(best_policy, dict):
            for k, v in best_policy.items():
                print(f"{k}: {v}")
    print("--------------------------")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    # Print unattended patients and their remaining treatment time
    if sim.waiting_patients:
        print("Unattended patients (ID, severity, treatment_time left):")
        for p in sim.waiting_patients:
            print(f"  ID: {p.id}, severity: {p.severity}, treatment_time: {p.treatment_time}")
    else:
        print("All patients attended.")