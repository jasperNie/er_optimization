import random
import numpy as np
from classes import ERSimulation


class HybridOptimizer:
    """Hybrid optimizer combining multiple optimization strategies"""
    
    def __init__(self, num_generations=150, population_size=100, num_nurses=3, total_time=96, arrival_prob=0.3, seed=123):
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_nurses = num_nurses
        self.total_time = total_time
        self.arrival_prob = arrival_prob
        self.seed = seed
        
        # Multi-strategy parameters
        self.strategies = ['severity_first', 'deterioration_first', 'balanced', 'wait_focused', 'smart_hybrid']
        self.strategy_weights = {strategy: 1.0 for strategy in self.strategies}
        
        random.seed(seed)
        np.random.seed(seed)

    def create_strategy_policy(self, strategy_type):
        """Create policies based on different strategic approaches"""
        base_policy = {
            'severity': random.uniform(0.5, 3.0),
            'deterioration': random.uniform(0.5, 3.0),
            'wait_time': random.uniform(0.1, 2.0),
            'strategy_type': strategy_type
        }
        
        if strategy_type == 'severity_first':
            # Prioritize severity above all else
            base_policy.update({
                'severity': random.uniform(3.0, 5.0),
                'deterioration': random.uniform(1.0, 2.5),
                'wait_time': random.uniform(0.1, 1.0),
                'severity_threshold': random.uniform(3.5, 4.5),
                'emergency_multiplier': random.uniform(2.0, 4.0)
            })
            
        elif strategy_type == 'deterioration_first':
            # Focus on preventing deterioration
            base_policy.update({
                'severity': random.uniform(1.5, 3.0),
                'deterioration': random.uniform(3.0, 5.0),
                'wait_time': random.uniform(0.5, 1.5),
                'deterioration_threshold': random.uniform(0.3, 0.7),
                'prevention_bonus': random.uniform(1.5, 3.0)
            })
            
        elif strategy_type == 'balanced':
            # Balanced approach
            base_policy.update({
                'severity': random.uniform(1.5, 2.5),
                'deterioration': random.uniform(1.5, 2.5),
                'wait_time': random.uniform(0.8, 1.8),
                'balance_factor': random.uniform(0.8, 1.2)
            })
            
        elif strategy_type == 'wait_focused':
            # Minimize overall waiting time
            base_policy.update({
                'severity': random.uniform(1.0, 2.0),
                'deterioration': random.uniform(1.0, 2.0),
                'wait_time': random.uniform(2.0, 4.0),
                'wait_threshold': random.uniform(5, 15),
                'efficiency_bonus': random.uniform(1.2, 2.0)
            })
            
        elif strategy_type == 'smart_hybrid':
            # Adaptive strategy that changes based on conditions
            base_policy.update({
                'severity': random.uniform(1.8, 3.2),
                'deterioration': random.uniform(1.8, 3.2),
                'wait_time': random.uniform(0.8, 2.2),
                'adaptive_factor': random.uniform(0.5, 1.5),
                'queue_sensitivity': random.uniform(0.3, 1.0),
                'time_sensitivity': random.uniform(0.2, 0.8)
            })
            
        return base_policy

    def strategy_triage_score(self, patient, simulation_state, policy):
        """Calculate triage score based on strategy type"""
        if patient is None:
            return 0
            
        strategy_type = policy['strategy_type']
        current_time = simulation_state.get('current_time', 0)
        queue_length = simulation_state.get('queue_length', 0)
        nurses_busy = simulation_state.get('nurses_busy', 0)
        
        if strategy_type == 'severity_first':
            score = policy['severity'] * patient.severity
            
            # Emergency cases get massive priority
            severity_threshold = policy.get('severity_threshold', 4.0)
            if patient.severity >= severity_threshold:
                emergency_multiplier = policy.get('emergency_multiplier', 3.0)
                score *= emergency_multiplier
                
            score += policy['deterioration'] * patient.deterioration_chance
            score += policy['wait_time'] * (patient.wait_time + 1)
            
        elif strategy_type == 'deterioration_first':
            score = policy['deterioration'] * patient.deterioration_chance
            
            # High deterioration risk gets prevention bonus
            deterioration_threshold = policy.get('deterioration_threshold', 0.5)
            if patient.deterioration_chance >= deterioration_threshold:
                prevention_bonus = policy.get('prevention_bonus', 2.0)
                score *= prevention_bonus
                
            score += policy['severity'] * patient.severity
            score += policy['wait_time'] * (patient.wait_time + 1)
            
        elif strategy_type == 'balanced':
            balance_factor = policy.get('balance_factor', 1.0)
            score = (
                policy['severity'] * patient.severity * balance_factor +
                policy['deterioration'] * patient.deterioration_chance * balance_factor +
                policy['wait_time'] * (patient.wait_time + 1)
            )
            
        elif strategy_type == 'wait_focused':
            score = policy['wait_time'] * (patient.wait_time + 1)
            
            # Heavy penalty for long waits
            wait_threshold = policy.get('wait_threshold', 10)
            if patient.wait_time > wait_threshold:
                efficiency_bonus = policy.get('efficiency_bonus', 1.5)
                score *= efficiency_bonus
                
            score += policy['severity'] * patient.severity
            score += policy['deterioration'] * patient.deterioration_chance
            
        elif strategy_type == 'smart_hybrid':
            # Base score
            score = (
                policy['severity'] * patient.severity +
                policy['deterioration'] * patient.deterioration_chance +
                policy['wait_time'] * (patient.wait_time + 1)
            )
            
            # Adaptive adjustments based on current conditions
            # If queue is long, prioritize efficiency
            queue_sensitivity = policy.get('queue_sensitivity', 0.5)
            if queue_length > 5:
                queue_factor = 1 + (queue_length - 5) * queue_sensitivity
                if patient.severity <= 2:  # Quick cases
                    score *= queue_factor
                    
            # If many nurses busy, prioritize critical cases
            adaptive_factor = policy.get('adaptive_factor', 1.0)
            if nurses_busy > 0.7:  # More than 70% busy
                if patient.severity >= 4:
                    score *= (1 + adaptive_factor)
                    
            # Time-based adjustments
            time_sensitivity = policy.get('time_sensitivity', 0.5)
            time_factor = 1 + (current_time / 100) * time_sensitivity
            score *= time_factor
            
        return score

    def evaluate_strategy_policy(self, policy):
        """Evaluate a strategy-based policy"""
        def triage_function(patient):
            # Get simulation state (this is a simplified approximation)
            sim_state = {
                'current_time': getattr(triage_function, 'current_time', 0),
                'queue_length': getattr(triage_function, 'queue_length', 0),
                'nurses_busy': getattr(triage_function, 'nurses_busy', 0)
            }
            return self.strategy_triage_score(patient, sim_state, policy)
        
        sim = ERSimulation(
            num_nurses=self.num_nurses,
            total_time=self.total_time,
            arrival_prob=self.arrival_prob,
            triage_policy=triage_function,
            verbose=False,
            seed=self.seed,
            use_shifts=True
        )
        
        result = sim.run()
        
        if result['avg_weighted_wait'] is None:
            return 1000  # Heavy penalty for no completions
            
        # Multi-objective fitness
        completion_rate = result['completed'] / max(1, result['completed'] + result['still_waiting'])
        fitness = result['avg_weighted_wait'] + (1 - completion_rate) * 30
        
        return fitness

    def adaptive_crossover(self, p1, p2):
        """Adaptive crossover that preserves good strategies"""
        child = {}
        
        # Inherit strategy type from better parent (determined randomly with bias)
        if random.random() < 0.7:  # 70% chance to inherit from first parent
            child['strategy_type'] = p1['strategy_type']
            template = p1
        else:
            child['strategy_type'] = p2['strategy_type']
            template = p2
            
        # Core parameters: blend
        for param in ['severity', 'deterioration', 'wait_time']:
            alpha = random.uniform(0.3, 0.7)
            child[param] = alpha * p1[param] + (1 - alpha) * p2[param]
            
        # Strategy-specific parameters: inherit from template with mutation
        for key, value in template.items():
            if key not in child and key != 'strategy_type':
                child[key] = value + random.gauss(0, abs(value) * 0.1)
                child[key] = max(0.1, child[key])  # Keep positive
                
        return child

    def smart_mutation(self, policy, generation_ratio):
        """Smart mutation that adapts based on generation and strategy"""
        new_policy = policy.copy()
        
        # Mutation rate decreases over generations but has periodic spikes
        base_rate = 0.2 * (1 - generation_ratio * 0.7)
        if generation_ratio > 0.8:  # Late-generation exploration spike
            base_rate *= 2
            
        for key, value in new_policy.items():
            if key == 'strategy_type':
                # Occasionally change strategy type completely
                if random.random() < 0.05:  # 5% chance
                    new_policy[key] = random.choice(self.strategies)
                continue
                
            if random.random() < base_rate:
                if key in ['severity', 'deterioration', 'wait_time']:
                    # Core parameters
                    mutation_strength = 0.3 if generation_ratio < 0.5 else 0.15
                    new_policy[key] += random.gauss(0, mutation_strength)
                    new_policy[key] = max(0.1, min(5.0, new_policy[key]))
                else:
                    # Strategy-specific parameters
                    mutation_strength = abs(value) * 0.2
                    new_policy[key] += random.gauss(0, mutation_strength)
                    new_policy[key] = max(0.1, new_policy[key])
                    
        return new_policy

    def run(self, gen_log_path="hybrid_generation_log.txt"):
        """Run hybrid multi-strategy optimization"""
        # Training started
        
        # Initialize diverse population with different strategies
        population = []
        strategies_per_strategy = self.population_size // len(self.strategies)
        
        for strategy in self.strategies:
            for _ in range(strategies_per_strategy):
                population.append(self.create_strategy_policy(strategy))
                
        # Fill remainder with random strategies
        while len(population) < self.population_size:
            strategy = random.choice(self.strategies)
            population.append(self.create_strategy_policy(strategy))
            
        best_fitness_ever = float('inf')
        strategy_performance = {strategy: [] for strategy in self.strategies}
        
        with open(gen_log_path, "w") as gen_log:
            gen_log.write("Hybrid Multi-Strategy Optimization Log\n")
            gen_log.write("=" * 50 + "\n\n")
            
            for gen in range(self.num_generations):
                generation_ratio = gen / self.num_generations
                
                # Progress bar: show training progress
                if (gen + 1) % 5 == 0 or gen == self.num_generations - 1:
                    bar_len = 40
                    progress = int(bar_len * (gen + 1) / self.num_generations)
                    bar = '[' + '#' * progress + '-' * (bar_len - progress) + ']'
                    print(f"Gen {gen+1:3d}/{self.num_generations} {bar}", end='\r', flush=True)
                
                # Evaluate all individuals
                fitnesses = [self.evaluate_strategy_policy(policy) for policy in population]
                
                # Track strategy performance
                for i, policy in enumerate(population):
                    strategy_type = policy['strategy_type']
                    strategy_performance[strategy_type].append(fitnesses[i])
                
                # Find best individual
                best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
                best_fitness = fitnesses[best_idx]
                best_policy = population[best_idx]
                
                if best_fitness < best_fitness_ever:
                    best_fitness_ever = best_fitness
                
                # Logging every 15 generations
                if gen % 15 == 0 or gen == self.num_generations - 1:
                    gen_log.write(f"Generation {gen+1:3d}: Best fitness = {best_fitness:.3f}\n")
                    gen_log.write(f"  Best strategy: {best_policy['strategy_type']}\n")
                    
                    # Strategy performance analysis
                    gen_log.write("  Strategy performance (avg fitness last 10 gen):\n")
                    for strategy in self.strategies:
                        recent_performance = strategy_performance[strategy][-10:] if strategy_performance[strategy] else [1000]
                        avg_performance = sum(recent_performance) / len(recent_performance)
                        gen_log.write(f"    {strategy}: {avg_performance:.3f}\n")
                    gen_log.write("\n")
                
                # Selection and reproduction
                # Elite preservation
                elite_size = max(3, self.population_size // 30)
                elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:elite_size]
                elites = [population[i] for i in elite_indices]
                
                # Strategy-aware selection
                new_population = elites[:]
                
                while len(new_population) < self.population_size:
                    # Select parents with strategy diversity in mind
                    if random.random() < 0.3:  # 30% chance for cross-strategy breeding
                        # Select parents from different strategies
                        strategies_present = list(set(p['strategy_type'] for p in population))
                        if len(strategies_present) > 1:
                            strategy1, strategy2 = random.sample(strategies_present, 2)
                            candidates1 = [i for i, p in enumerate(population) if p['strategy_type'] == strategy1]
                            candidates2 = [i for i, p in enumerate(population) if p['strategy_type'] == strategy2]
                            parent1_idx = random.choice(candidates1)
                            parent2_idx = random.choice(candidates2)
                        else:
                            parent1_idx, parent2_idx = random.sample(range(len(population)), 2)
                    else:
                        # Tournament selection
                        tournament_size = min(5, len(population))
                        tournament_indices = random.sample(range(len(population)), tournament_size)
                        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
                        
                        parent1_idx = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
                        remaining = [i for i in tournament_indices if i != parent1_idx]
                        parent2_idx = random.choice(remaining)
                    
                    parent1 = population[parent1_idx]
                    parent2 = population[parent2_idx]
                    
                    # Crossover and mutation
                    child = self.adaptive_crossover(parent1, parent2)
                    child = self.smart_mutation(child, generation_ratio)
                    
                    new_population.append(child)
                
                population = new_population
            
            print(f"Gen {self.num_generations:3d}/{self.num_generations} [{'#'*40}] Done.{' '*10}")
            
            # Final analysis
            gen_log.write(f"\n{'='*50}\n")
            gen_log.write(f"OPTIMIZATION COMPLETE\n")
            gen_log.write(f"Best fitness achieved: {best_fitness_ever:.3f}\n")
            gen_log.write(f"Winning strategy: {best_policy['strategy_type']}\n")
            gen_log.write(f"\nFinal strategy performance summary:\n")
            
            for strategy in self.strategies:
                if strategy_performance[strategy]:
                    avg_perf = sum(strategy_performance[strategy]) / len(strategy_performance[strategy])
                    best_perf = min(strategy_performance[strategy])
                    gen_log.write(f"  {strategy:15}: avg={avg_perf:.3f}, best={best_perf:.3f}\n")
            
            gen_log.write(f"\nBest policy parameters:\n")
            for key, value in best_policy.items():
                if isinstance(value, str):
                    gen_log.write(f"  {key}: {value}\n")
                else:
                    gen_log.write(f"  {key}: {value:.4f}\n")
        
        return best_policy