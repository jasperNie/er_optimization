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
    def __init__(self, num_nurses=3, total_time=60, arrival_prob=0.3, triage_policy=None, verbose=False, explain_triage=False):
        # Each timestep = 15 minutes, 60 timesteps = 12 hours
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
        self.explain_triage = explain_triage

    def generate_patient(self):
        severity = random.randint(1, 5)
        deterioration = round(random.uniform(0.1, 0.5), 2)
        treatment_time = random.randint(5, 15)
        p = Patient(self.next_patient_id, severity, deterioration, treatment_time)
        self.next_patient_id += 1
        return p

    def step(self):
        self.time += 1

        # 1. New patient arrivals
        if random.random() < self.arrival_prob:
            new_patient = self.generate_patient()
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

        # 4. Assign free nurses (using triage policy)
        free_nurses = [n for n in self.nurses if n.current_patient is None]
        if free_nurses and self.waiting_patients:
            def triage_score(p):
                score = (
                    self.triage_policy['severity'] * p.severity +
                    self.triage_policy['deterioration'] * p.deterioration_chance +
                    self.triage_policy['wait_time'] * (p.wait_time + 1)
                )
                if self.explain_triage:
                    print(f"[t={self.time}] Triage score for Patient {p.id}: "
                          f"severity({self.triage_policy['severity']}*{p.severity}) + "
                          f"deterioration({self.triage_policy['deterioration']}*{p.deterioration_chance}) + "
                          f"wait_time({self.triage_policy['wait_time']}*{p.wait_time + 1}) = {score:.2f}")
                return score
            self.waiting_patients.sort(key=triage_score, reverse=True)
            if self.explain_triage:
                print(f"[t={self.time}] Triage policy weights: {self.triage_policy}")
                print(f"[t={self.time}] Sorted patients by triage score: {[p.id for p in self.waiting_patients]}")
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
    def __init__(self, num_generations, population_size, num_nurses, total_time, arrival_prob):
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_nurses = num_nurses
        self.total_time = total_time
        self.arrival_prob = arrival_prob

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
            verbose=False
        )
        result = sim.run()
        # If no patients treated, penalize
        if result['avg_weighted_wait'] is None:
            return float('inf')
        return result['avg_weighted_wait']

    def run(self):
        # Initialize population
        population = [self.random_policy() for _ in range(self.population_size)]
        for gen in range(self.num_generations):
            # Evaluate fitness
            fitnesses = [self.evaluate(p) for p in population]
            # Select best policies
            sorted_pop = [p for _, p in sorted(zip(fitnesses, population), key=lambda x: x[0])]
            population = sorted_pop[:self.population_size//2]
            # Generate new population
            while len(population) < self.population_size:
                if random.random() < 0.5:
                    # Mutation
                    parent = random.choice(population)
                    child = self.mutate(parent)
                else:
                    # Crossover
                    p1, p2 = random.sample(population, 2)
                    child = self.crossover(p1, p2)
                population.append(child)
            print(f"Generation {gen+1}: Best avg_weighted_wait = {fitnesses[0]:.2f}")
        # Final best policy
        best_policy = population[0]
        print("Best triage policy:", best_policy)
        return best_policy


# Example usage: Run evolutionary optimizer
if __name__ == "__main__":
    optimizer = EvolutionaryTriageOptimizer(num_generations=100, population_size=100, num_nurses=2, total_time=60, arrival_prob=0.5)
    best_policy = optimizer.run()
    # Run final simulation with best policy and print results
    print("\n--- Running final simulation with explainable triage ---")
    sim = ERSimulation(num_nurses=3, total_time=60, arrival_prob=0.3, triage_policy=best_policy, verbose=True, explain_triage=True)
    metrics = sim.run()
    print("\nFinal simulation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nTriage policy weights explanation:")
    print("The triage policy determines which patient is prioritized for treatment based on a weighted sum of three factors:")
    print(f"  Severity weight: {best_policy['severity']:.2f} (multiplied by patient severity, 1-5)")
    print(f"  Deterioration weight: {best_policy['deterioration']:.2f} (multiplied by patient deterioration chance, 0.1-0.5)")
    print(f"  Wait time weight: {best_policy['wait_time']:.2f} (multiplied by patient wait time in timesteps)")
    print("\nAt each assignment, the triage score is computed as:")
    print("  triage_score = severity_weight * severity + deterioration_weight * deterioration_chance + wait_time_weight * (wait_time + 1)")
    print("The patient with the highest triage score is selected for treatment next.")
