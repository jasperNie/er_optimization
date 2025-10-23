import random
from classes import ERSimulation


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