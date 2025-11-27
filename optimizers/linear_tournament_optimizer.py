import random
from classes import ERSimulation


class ImprovedEvolutionaryTriageOptimizer:
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

    def tournament_selection(self, population, fitnesses, tournament_size=10, num_winners=5):
        """
        Improved selection method: Tournament Selection
        - Randomly pick 'tournament_size' individuals
        - Select best 'num_winners' from tournament
        - This maintains diversity while still favoring good solutions
        """
        # Randomly select tournament participants
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        
        # Sort tournament participants by fitness (lower is better)
        tournament_sorted = sorted(tournament_indices, key=lambda i: fitnesses[i])
        
        # Return the best num_winners from tournament
        winner_indices = tournament_sorted[:num_winners]
        return [population[i] for i in winner_indices]

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
        # Initialize population
        population = [self.random_policy() for _ in range(self.population_size)]
        
        with open(gen_log_path, "w") as gen_log:
            for gen in range(self.num_generations):
                # Progress bar: show training progress
                if (gen+1) % 5 == 0 or gen == self.num_generations-1:
                    bar_len = 40
                    progress = int(bar_len * (gen+1) / self.num_generations)
                    bar = '[' + '#' * progress + '-' * (bar_len - progress) + ']'
                    print(f"Gen {gen+1:3d}/{self.num_generations} {bar}", end='\r', flush=True)
                
                # Evaluate fitness for entire population
                fitnesses = [self.evaluate(p) for p in population]
                
                # Find best policy for logging
                best_index = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
                best_policy = population[best_index]
                best_fitness = fitnesses[best_index]
                
                gen_log.write(f"Generation {gen+1}: Best avg_weighted_wait = {best_fitness:.2f}\n")
                gen_log.write(f"  Best policy weights: {{'severity': {best_policy['severity']:.4f}, 'deterioration': {best_policy['deterioration']:.4f}, 'wait_time': {best_policy['wait_time']:.4f}}}\n")
                
                # Tournament Selection instead of pure elitism
                selected_parents = self.tournament_selection(
                    population, fitnesses, 
                    tournament_size=10,  # Randomly pick 10 candidates
                    num_winners=5        # Select best 5 from those 10
                )
                
                # Generate new population
                new_population = []
                
                # Keep the selected parents (ensures good solutions aren't lost)
                new_population.extend(selected_parents)
                
                # Fill rest with offspring from selected parents
                while len(new_population) < self.population_size:
                    parent1 = random.choice(selected_parents)
                    parent2 = random.choice(selected_parents)
                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child)
                    new_population.append(child)
                
                population = new_population
            
            print(f"Gen {self.num_generations:3d}/{self.num_generations} [{'#'*40}] Done.{' '*10}")
            gen_log.write(f"Best triage policy: {best_policy}\n")
        
        return best_policy


# Comparison function to test both algorithms
def compare_algorithms(num_runs=3):
    """Compare original vs improved algorithm"""
    print("=== ALGORITHM COMPARISON ===\n")
    
    original_results = []
    improved_results = []
    
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        
        # Test original algorithm
        print("  Testing original algorithm...")
        from evolutionary_optimizer import EvolutionaryTriageOptimizer
        original = EvolutionaryTriageOptimizer(num_generations=50, population_size=100, seed=run*1000)
        original_best = original.run(f"original_run_{run+1}.txt")
        original_fitness = original.evaluate(original_best)
        original_results.append(original_fitness)
        
        # Test improved algorithm  
        print("  Testing improved algorithm...")
        improved = ImprovedEvolutionaryTriageOptimizer(num_generations=50, population_size=100, seed=run*1000)
        improved_best = improved.run(f"improved_run_{run+1}.txt")
        improved_fitness = improved.evaluate(improved_best)
        improved_results.append(improved_fitness)
        
        print(f"    Original: {original_fitness:.2f}, Improved: {improved_fitness:.2f}")
        print()
    
    print("=== FINAL RESULTS ===")
    print(f"Original Algorithm - Average: {sum(original_results)/len(original_results):.2f}")
    print(f"Improved Algorithm - Average: {sum(improved_results)/len(improved_results):.2f}")
    
    improvement = (sum(original_results) - sum(improved_results)) / sum(original_results) * 100
    print(f"Improvement: {improvement:.1f}%")


if __name__ == "__main__":
    # Test the improved algorithm
    optimizer = ImprovedEvolutionaryTriageOptimizer(
        num_generations=100, 
        population_size=100, 
        num_nurses=3, 
        total_time=96, 
        arrival_prob=0.5, 
        seed=2025
    )
    best_policy = optimizer.run("improved_generation_log.txt")
    print(f"\nBest policy found: {best_policy}")