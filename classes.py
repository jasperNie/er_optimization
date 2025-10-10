import random

# -------------------------
# Patient & Nurse Classes
# -------------------------

class Patient:
    def __init__(self, id, severity, deterioration_chance, treatment_time):
        self.id = id
        self.severity = severity  # e.g., 1–5 (5 = most severe)
        self.deterioration_chance = deterioration_chance  # float [0,1]
        self.treatment_time = treatment_time  # minutes needed
        self.wait_time = 0

    def __repr__(self):
        return (f"Patient({self.id}, sev={self.severity}, "
                f"det={self.deterioration_chance:.2f}, "
                f"treat={self.treatment_time}, wait={self.wait_time})")


class Nurse:
    def __init__(self, id):
        self.id = id
        self.busy_until = 0  # time until nurse is free
        self.current_patient = None

    def __repr__(self):
        return f"Nurse({self.id}, busy_until={self.busy_until})"


# -------------------------
# Simulation
# -------------------------

import statistics


# -------------------------
# Evolutionary Algorithm for Triage Policy
# -------------------------


class ERSimulation:
    def __init__(self, num_nurses=3, total_time=50, arrival_prob=0.3, triage_policy=None, verbose=False, seed=None):
        self.time = 0
        self.total_time = total_time
        self.arrival_prob = arrival_prob
        self.nurses = [Nurse(i) for i in range(num_nurses)]
        self.waiting_patients = []
        self.completed_patients = []
        self.next_patient_id = 0
        self.started_patients = []  # track when patients actually start treatment
        self.triage_policy = triage_policy or {'severity': 1.0, 'deterioration': 1.0, 'wait_time': 1.0}
        self.verbose = verbose
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        # Pre-generate patient arrivals and attributes for each timestep
        import math
        self.patient_arrivals = []  # List of (timestep, Patient) or None
        for t in range(self.total_time):
            # More realistic, time-varying and bursty arrival probability
            base_prob = self.arrival_prob
            time_factor = 0.15 * (1 + math.sin(2 * math.pi * t / 24))  # daily cycle
            burst = 0.15 if random.random() < 0.05 else 0  # occasional burst
            prob = min(1.0, max(0.0, base_prob + time_factor + burst))
            if random.random() < prob:
                # Correlate severity, deterioration, and treatment time
                severity = random.choices([1,2,3,4,5], weights=[0.1,0.15,0.25,0.25,0.25])[0]
                det_base = 0.1 + 0.15 * (severity-1)
                deterioration = round(random.uniform(det_base, min(0.6, det_base+0.15)), 2)
                treat_base = 5 + 2 * severity
                treatment_time = random.randint(treat_base, treat_base+5)
                patient = Patient(self.next_patient_id, severity, deterioration, treatment_time)
                self.next_patient_id += 1
                self.patient_arrivals.append(patient)
            else:
                self.patient_arrivals.append(None)
        if self.verbose:
            self.print_arrivals()

    def print_arrivals(self, file_path="arrivals_log.txt"):
        with open(file_path, "w") as f:
            f.write("[ERSimulation] Pre-generated patients (None = no arrival):\n")
            for t, p in enumerate(self.patient_arrivals, 1):
                f.write(f"t={t}: {p}\n")

    def step(self):
        self.time += 1

        # 1. New patient arrivals (use pre-generated list)
        if self.time <= len(self.patient_arrivals):
            new_patient = self.patient_arrivals[self.time - 1]
            if new_patient is not None:
                self.waiting_patients.append(new_patient)
                if self.verbose:
                    print(f"[t={self.time}] New arrival: {new_patient}")

        # 2. Update nurse status
        for nurse in self.nurses:
            if nurse.current_patient and self.time >= nurse.busy_until:
                self.completed_patients.append(nurse.current_patient)
                if self.verbose:
                    print(f"[t={self.time}] Nurse {nurse.id} finished treating {nurse.current_patient}")
                nurse.current_patient = None

        # 3. Update waiting patients
        for patient in self.waiting_patients:
            patient.wait_time += 1
            # Deterioration: probabilistically increase severity
            if patient.severity < 5:
                if random.random() < patient.deterioration_chance:
                    patient.severity += 1
                    # Lower deterioration chance after upgrade to avoid rapid jumps
                    patient.deterioration_chance = max(0.01, patient.deterioration_chance * 0.5)
                    # Optionally, log this event (uncomment if needed):
                    # print(f"[t={self.time}] Patient {patient.id} deteriorated to severity {patient.severity}")

        # 4. Assign free nurses (using triage policy)
        free_nurses = [n for n in self.nurses if n.current_patient is None]
        if free_nurses and self.waiting_patients:
            def triage_score(p):
                if callable(self.triage_policy):
                    return self.triage_policy(p)
                else:
                    return (
                        self.triage_policy['severity'] * p.severity +
                        self.triage_policy['deterioration'] * p.deterioration_chance +
                        self.triage_policy['wait_time'] * (p.wait_time + 1)
                    )
            self.waiting_patients.sort(key=triage_score, reverse=True)
            for nurse in free_nurses:
                if self.waiting_patients:
                    patient = self.waiting_patients.pop(0)
                    nurse.current_patient = patient
                    nurse.busy_until = self.time + patient.treatment_time
                    self.started_patients.append((patient, patient.wait_time))
                    if self.verbose:
                        print(f"[t={self.time}] Nurse {nurse.id} starts treating {patient}")

    def run(self):
        while self.time < self.total_time:
            self.step()

        # Collect metrics
        if self.started_patients:
            waits = [wait for _, wait in self.started_patients]
            weighted_waits = [
                wait * p.severity * p.deterioration_chance
                for p, wait in self.started_patients
            ]
            avg_wait = statistics.mean(waits)
            avg_weighted_wait = statistics.mean(weighted_waits)
            return {
                'completed': len(self.completed_patients),
                'still_waiting': len(self.waiting_patients),
                'avg_wait': avg_wait,
                'max_wait': max(waits),
                'avg_weighted_wait': avg_weighted_wait
            }
        else:
            return {
                'completed': 0,
                'still_waiting': len(self.waiting_patients),
                'avg_wait': None,
                'max_wait': None,
                'avg_weighted_wait': None
            }



class EvolutionaryTriageOptimizer:
    def __init__(self, num_generations=20, population_size=100, num_nurses=3, total_time=30, arrival_prob=0.5, seed=123):
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_nurses = num_nurses
        self.total_time = total_time
        self.arrival_prob = arrival_prob
        self.seed = seed

    def random_policy(self):
        # Random weights for severity, deterioration, wait_time
        return {
            'severity': random.uniform(0.5, 2.0),
            'deterioration': random.uniform(0.5, 2.0),
            'wait_time': random.uniform(0.5, 2.0)
        }

    def mutate(self, policy):
        # Slightly change one of the weights
        key = random.choice(list(policy.keys()))
        new_policy = policy.copy()
        new_policy[key] += random.uniform(-0.2, 0.2)
        new_policy[key] = max(0.1, new_policy[key])
        return new_policy

    def crossover(self, p1, p2):
        # Average weights
        return {k: (p1[k] + p2[k]) / 2 for k in p1}

    def evaluate(self, policy):
        # Run simulation and return fitness (lower avg_weighted_wait is better)
        sim = ERSimulation(
            num_nurses=self.num_nurses,
            total_time=self.total_time,
            arrival_prob=self.arrival_prob,
            triage_policy=policy,
            verbose=False,
            seed=self.seed
        )
        result = sim.run()
        # If no patients treated, penalize
        if result['avg_weighted_wait'] is None:
            return float('inf')
        return result['avg_weighted_wait']

    def run(self, gen_log_path="generation_log.txt"):
        import sys
        # Initialize population
        population = [self.random_policy() for _ in range(self.population_size)]
        print("Running simulation (evolutionary optimization)...")
        elite_size = 5
        with open(gen_log_path, "w") as gen_log:
            for gen in range(self.num_generations):
                # Loading bar: print every 5 generations or last gen
                if (gen+1) % 5 == 0 or gen == self.num_generations-1:
                    bar_len = 40
                    progress = int(bar_len * (gen+1) / self.num_generations)
                    bar = '[' + '#' * progress + '-' * (bar_len - progress) + ']'
                    print(f"Gen {gen+1:3d}/{self.num_generations} {bar}", end='\r', flush=True)
                # Evaluate fitness
                fitnesses = [self.evaluate(p) for p in population]
                # Select elites
                elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:elite_size]
                elites = [population[i] for i in elite_indices]
                best_policy = elites[0]
                best_fitness = fitnesses[elite_indices[0]]
                gen_log.write(f"Generation {gen+1}: Best avg_weighted_wait = {best_fitness:.2f}\n")
                gen_log.write(f"  Best policy weights: {{'severity': {best_policy['severity']:.4f}, 'deterioration': {best_policy['deterioration']:.4f}, 'wait_time': {best_policy['wait_time']:.4f}}}\n")
                # Generate new population: keep elites, fill rest with mutated/crossover children from elites
                new_population = elites[:]
                while len(new_population) < self.population_size:
                    parent1 = random.choice(elites)
                    parent2 = random.choice(elites)
                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child)
                    new_population.append(child)
                population = new_population
            # Print final bar
            print(f"Gen {self.num_generations:3d}/{self.num_generations} [{'#'*40}] Done.{' '*10}")
            gen_log.write(f"Best triage policy: {best_policy}\n")
        return best_policy


# Example usage: Run evolutionary optimizer
if __name__ == "__main__":
    optimizer = EvolutionaryTriageOptimizer(num_generations=100, population_size=100, num_nurses=3, total_time=120, arrival_prob=0.5, seed=2025)
    best_policy = optimizer.run(gen_log_path="generation_log.txt")

    # --- MTS and ESI comparison ---
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

    # Generate arrivals ONCE for all policies
    import math
    random.seed(2025)
    arrivals = []
    patient_id = 0
    total_time = optimizer.total_time if 'optimizer' in locals() else 120
    for t in range(total_time):
        base_prob = 0.3
        time_factor = 0.15 * (1 + math.sin(2 * math.pi * t / 24))
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

    # Run all three policies on the same arrivals
    for name, policy, verbose in [
        ('Optimized', best_policy, True),
        ('MTS', mts_policy, False),
        ('ESI', esi_policy, False)
    ]:
        sim = ERSimulation(num_nurses=5, total_time=120, arrival_prob=0, triage_policy=policy, verbose=False, seed=2025)
        sim.patient_arrivals = arrivals.copy()
        if verbose:
            sim.print_arrivals(file_path="arrivals_log.txt")
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
