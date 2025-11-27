import random
import numpy as np
import math
from classes import ERSimulation


class FairAdvancedEvolutionaryOptimizer:
    """Fair advanced algorithm using only severity, deterioration, and wait time data"""
    
    def __init__(self, num_generations=100, population_size=100, num_nurses=3, total_time=96, arrival_prob=0.3, seed=123):
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_nurses = num_nurses
        self.total_time = total_time
        self.arrival_prob = arrival_prob
        self.seed = seed
        
        # Advanced EA parameters
        self.elite_size = max(5, population_size // 20)  # Top 5%
        self.mutation_rate = 0.15  # Higher initial mutation
        self.use_adaptive_mutation = True
        self.crowding_distance_threshold = 0.1
        
        # Fitness tracking for adaptive parameters
        self.fitness_history = []
        self.diversity_history = []
        
        random.seed(seed)
        np.random.seed(seed)

    def create_fair_advanced_policy(self):
        """Create policy using only fair parameters (severity, deterioration, wait time)"""
        return {
            # Core weights (same as original but wider ranges)
            'severity': random.uniform(0.1, 5.0),
            'deterioration': random.uniform(0.1, 5.0), 
            'wait_time': random.uniform(0.1, 3.0),
            
            # Advanced factors using only available data
            'urgency_multiplier': random.uniform(1.0, 4.0),    # Boost for high severity
            'deterioration_threshold': random.uniform(0.2, 0.8), # When to apply deterioration boost
            'deterioration_boost': random.uniform(1.2, 3.0),   # How much to boost deteriorating patients
            'wait_threshold': random.uniform(3, 15),           # When wait becomes critical
            'wait_penalty_factor': random.uniform(1.1, 2.5),  # Exponential wait penalty
            'severity_threshold': random.uniform(3.5, 4.5),   # When severity becomes critical
            
            # Queue dynamics (can be estimated from simulation state)
            'queue_pressure': random.uniform(0.0, 2.0),       # Queue length influence
            'efficiency_factor': random.uniform(0.8, 1.5),    # Balance efficiency vs thoroughness
            
            # Time-based adjustments (shift patterns are observable)
            'day_shift_bonus': random.uniform(0.8, 1.2),      # 6AM-2PM adjustment
            'evening_shift_bonus': random.uniform(0.9, 1.1),  # 2PM-10PM adjustment  
            'night_shift_penalty': random.uniform(1.0, 1.3),  # 10PM-6AM adjustment
            
            # Advanced scoring methods
            'severity_power': random.uniform(0.8, 2.0),       # Non-linear severity scaling
            'deterioration_power': random.uniform(0.8, 2.0),  # Non-linear deterioration scaling
        }

    def fair_advanced_triage_score(self, patient, current_time, queue_length, policy):
        """Advanced triage scoring using only fair parameters"""
        if patient is None:
            return 0
            
        # Non-linear base scoring (vs simple linear multiplication)
        severity_score = policy['severity'] * (patient.severity ** policy['severity_power'])
        deterioration_score = policy['deterioration'] * (patient.deterioration_chance ** policy['deterioration_power'])
        wait_score = policy['wait_time'] * (patient.wait_time + 1)
        
        base_score = severity_score + deterioration_score + wait_score
        
        # Critical severity boost
        if patient.severity >= policy['severity_threshold']:
            base_score *= policy['urgency_multiplier']
            
        # High deterioration risk boost
        if patient.deterioration_chance >= policy['deterioration_threshold']:
            base_score *= policy['deterioration_boost']
            
        # Progressive wait penalty (exponential growth for long waits)
        if patient.wait_time > policy['wait_threshold']:
            wait_multiplier = policy['wait_penalty_factor'] ** (patient.wait_time - policy['wait_threshold'])
            base_score *= wait_multiplier
            
        # Queue pressure adjustment (observable from simulation)
        if queue_length > 5:
            queue_factor = 1 + (queue_length - 5) * policy['queue_pressure'] / 10
            base_score *= queue_factor
            
        # Efficiency considerations
        base_score *= policy['efficiency_factor']
            
        # Time-of-day adjustments (shift patterns)
        timestep_in_day = current_time % 96
        if 0 <= timestep_in_day < 32:  # Day shift
            base_score *= policy['day_shift_bonus']
        elif 32 <= timestep_in_day < 64:  # Evening shift  
            base_score *= policy['evening_shift_bonus']
        else:  # Night shift
            base_score *= policy['night_shift_penalty']
            
        return base_score

    def adaptive_mutation_method(self, policy, generation, fitness_stagnation):
        """Adaptive mutation that increases when fitness stagnates"""
        base_mutation = self.mutation_rate
        
        # Increase mutation if fitness has stagnated
        if fitness_stagnation > 10:
            base_mutation *= (1 + fitness_stagnation / 20)
            
        # Late-generation exploration boost
        if generation > self.num_generations * 0.7:
            base_mutation *= 1.5
            
        new_policy = policy.copy()
        
        # Mutate each parameter with probability
        for key in new_policy:
            if random.random() < base_mutation:
                if key in ['severity', 'deterioration', 'wait_time']:
                    # Core parameters - careful mutations
                    new_policy[key] += random.gauss(0, 0.2)
                    new_policy[key] = max(0.05, min(5.0, new_policy[key]))
                elif key in ['urgency_multiplier', 'deterioration_boost', 'wait_penalty_factor']:
                    # Multiplier parameters
                    new_policy[key] += random.gauss(0, 0.1)
                    new_policy[key] = max(1.0, min(4.0, new_policy[key]))
                elif key in ['deterioration_threshold']:
                    # Threshold parameters (0-1 range)
                    new_policy[key] += random.gauss(0, 0.05)
                    new_policy[key] = max(0.0, min(1.0, new_policy[key]))
                elif key in ['severity_threshold']:
                    # Severity threshold (1-5 range)
                    new_policy[key] += random.gauss(0, 0.2)
                    new_policy[key] = max(1.0, min(5.0, new_policy[key]))
                elif key in ['wait_threshold']:
                    # Wait threshold (time steps)
                    new_policy[key] += random.gauss(0, 2.0)
                    new_policy[key] = max(1.0, min(30.0, new_policy[key]))
                elif key in ['severity_power', 'deterioration_power']:
                    # Power parameters
                    new_policy[key] += random.gauss(0, 0.1)
                    new_policy[key] = max(0.5, min(3.0, new_policy[key]))
                else:
                    # Other parameters - general mutation
                    new_policy[key] += random.gauss(0, abs(new_policy[key]) * 0.1)
                    new_policy[key] = max(0.1, new_policy[key])
                    
        return new_policy

    def multi_point_crossover(self, p1, p2):
        """Multi-point crossover for better recombination"""
        child = {}
        keys = list(p1.keys())
        random.shuffle(keys)
        
        # Random number of crossover points (1-3)
        num_points = random.randint(1, 3)
        crossover_points = sorted(random.sample(range(len(keys)), num_points))
        
        parent_source = 0  # Start with parent 1
        key_idx = 0
        
        for i, key in enumerate(keys):
            # Switch parent at crossover points
            if key_idx in crossover_points:
                parent_source = 1 - parent_source
            
            if parent_source == 0:
                child[key] = p1[key]
            else:
                child[key] = p2[key]
                
            key_idx += 1
            
        return child

    def calculate_diversity(self, population):
        """Calculate population diversity to monitor convergence"""
        if len(population) < 2:
            return 0
            
        total_distance = 0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                distance = 0
                for key in population[i]:
                    distance += abs(population[i][key] - population[j][key])
                total_distance += distance
                comparisons += 1
                
        return total_distance / comparisons if comparisons > 0 else 0

    def evaluate_with_multiple_seeds(self, policy, num_seeds=2):
        """Evaluate policy on multiple random seeds for robustness"""
        fitnesses = []
        
        for i in range(num_seeds):
            eval_seed = self.seed + i * 1000
            sim = ERSimulation(
                num_nurses=self.num_nurses,
                total_time=self.total_time,
                arrival_prob=self.arrival_prob,
                triage_policy=lambda p: self.fair_advanced_triage_score(
                    p, sim.time if hasattr(sim, 'time') else 0,
                    len(sim.waiting_patients) if hasattr(sim, 'waiting_patients') else 0,
                    policy
                ),
                verbose=False,
                seed=eval_seed,
                use_shifts=True
            )
            
            result = sim.run()
            
            if result['avg_weighted_wait'] is None:
                fitness = 1000  # Heavy penalty
            else:
                # Multi-objective fitness: weighted wait + completion penalty
                completion_rate = result['completed'] / max(1, result['completed'] + result['still_waiting'])
                fitness = result['avg_weighted_wait'] + (1 - completion_rate) * 50
                
            fitnesses.append(fitness)
            
        # Return average fitness across seeds for robustness
        return sum(fitnesses) / len(fitnesses)

    def run(self, gen_log_path="fair_advanced_generation_log.txt"):
        """Run fair advanced evolutionary optimization"""
        # Training started
        
        # Initialize population with fair advanced policies
        population = [self.create_fair_advanced_policy() for _ in range(self.population_size)]
        
        best_fitness_ever = float('inf')
        fitness_stagnation = 0
        
        with open(gen_log_path, "w") as gen_log:
            gen_log.write("Fair Advanced Evolutionary Triage Optimization Log\n")
            gen_log.write("=" * 50 + "\n")
            gen_log.write("Using only fair parameters: severity, deterioration, wait_time + advanced math\n\n")
            
            for gen in range(self.num_generations):
                # Progress bar: show training progress
                if (gen + 1) % 5 == 0 or gen == self.num_generations - 1:
                    bar_len = 40
                    progress = int(bar_len * (gen + 1) / self.num_generations)
                    bar = '[' + '#' * progress + '-' * (bar_len - progress) + ']'
                    print(f"Gen {gen+1:3d}/{self.num_generations} {bar}", end='\r', flush=True)
                
                # Evaluate fitness for all individuals
                fitnesses = []
                for policy in population:
                    fitness = self.evaluate_with_multiple_seeds(policy, 1 if gen < 50 else 2)
                    fitnesses.append(fitness)
                
                # Track diversity and fitness
                diversity = self.calculate_diversity(population)
                self.diversity_history.append(diversity)
                
                # Find best individual
                best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
                best_fitness = fitnesses[best_idx]
                best_policy = population[best_idx]
                
                # Check for improvement
                if best_fitness < best_fitness_ever:
                    best_fitness_ever = best_fitness
                    fitness_stagnation = 0
                else:
                    fitness_stagnation += 1
                
                self.fitness_history.append(best_fitness)
                
                # Logging
                if gen % 20 == 0 or gen == self.num_generations - 1:
                    gen_log.write(f"Generation {gen+1:3d}: Best fitness = {best_fitness:.3f}\n")
                    gen_log.write(f"  Population diversity: {diversity:.3f}\n")
                    gen_log.write(f"  Fitness stagnation: {fitness_stagnation} generations\n")
                    gen_log.write(f"  Best policy snippet: severity={best_policy['severity']:.3f}, "
                                f"deterioration={best_policy['deterioration']:.3f}, "
                                f"urgency_multiplier={best_policy['urgency_multiplier']:.3f}\n\n")
                
                # Selection and reproduction
                # Elite preservation
                elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:self.elite_size]
                elites = [population[i] for i in elite_indices]
                
                # Tournament selection for diversity
                new_population = elites[:]
                
                while len(new_population) < self.population_size:
                    # Parent selection with tournament
                    tournament_size = min(7, len(population))
                    tournament_indices = random.sample(range(len(population)), tournament_size)
                    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
                    
                    # Select two parents
                    parent1_idx = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
                    remaining_indices = [i for i in tournament_indices if i != parent1_idx]
                    parent2_idx = random.choice(remaining_indices)
                    
                    parent1 = population[parent1_idx]
                    parent2 = population[parent2_idx]
                    
                    # Crossover
                    child = self.multi_point_crossover(parent1, parent2)
                    
                    # Adaptive mutation
                    child = self.adaptive_mutation_method(child, gen, fitness_stagnation)
                    
                    new_population.append(child)
                
                population = new_population
                
                # Diversity injection if population converges
                if diversity < self.crowding_distance_threshold and gen < self.num_generations * 0.8:
                    # Replace worst 10% with random individuals
                    num_replacements = max(1, self.population_size // 10)
                    worst_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:num_replacements]
                    
                    for idx in worst_indices:
                        population[idx] = self.create_fair_advanced_policy()
            
            print(f"Gen {self.num_generations:3d}/{self.num_generations} [{'#'*40}] Done.{' '*10}")
            
            # Final evaluation and logging
            final_fitness = self.evaluate_with_multiple_seeds(best_policy, 5)
            gen_log.write(f"\n{'='*50}\n")
            gen_log.write(f"FAIR OPTIMIZATION COMPLETE\n")
            gen_log.write(f"Best fitness achieved: {best_fitness_ever:.3f}\n")
            gen_log.write(f"Final robust evaluation: {final_fitness:.3f}\n")
            gen_log.write(f"Total generations with stagnation: {fitness_stagnation}\n")
            gen_log.write(f"\nBest policy parameters:\n")
            for key, value in best_policy.items():
                gen_log.write(f"  {key}: {value:.4f}\n")
        
        return best_policy