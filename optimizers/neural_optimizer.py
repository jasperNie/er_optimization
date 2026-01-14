import random
import numpy as np
import math
from classes import ERSimulation


class FairNeuralEvolutionOptimizer:
    """Fair neural network evolution using only severity, deterioration, and wait time data"""
    
    def __init__(self, num_generations=100, population_size=80, num_nurses=3, total_time=96, arrival_prob=0.3, seed=123):
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_nurses = num_nurses
        self.total_time = total_time
        self.arrival_prob = arrival_prob
        self.seed = seed
        
        # Neural network parameters - FAIR VERSION
        self.input_size = 6  # Only fair input features
        self.hidden_size = 6  # Smaller hidden layer for fair comparison
        self.output_size = 1  # Single triage score output
        
        random.seed(seed)
        np.random.seed(seed)

    def create_fair_neural_policy(self):
        """Create a neural network policy using only fair parameters"""
        # Only fair input features: severity, deterioration, wait_time, 
        # time_of_day, queue_length, nurse_availability (observable simulation states)
        
        policy = {
            # Input to hidden layer weights (input_size x hidden_size)
            'w1': np.random.normal(0, 0.5, (self.input_size, self.hidden_size)),
            
            # Hidden layer biases
            'b1': np.random.normal(0, 0.2, self.hidden_size),
            
            # Hidden to output weights (hidden_size x output_size)
            'w2': np.random.normal(0, 0.5, (self.hidden_size, self.output_size)),
            
            # Output bias
            'b2': np.random.normal(0, 0.1, self.output_size),
            
            # Activation function types for each hidden neuron (0=sigmoid, 1=tanh, 2=relu)
            'activations': np.random.randint(0, 3, self.hidden_size),
            
            # Global scaling factors
            'output_scale': random.uniform(0.5, 2.0),
            'urgency_boost': random.uniform(1.0, 3.0)
        }
        
        return policy

    def extract_fair_features(self, patient, sim_state):
        """Extract only fair numerical features from patient and simulation state"""
        if patient is None:
            return np.zeros(self.input_size)
        
        features = np.zeros(self.input_size)
        
        # Fair patient features (available to ESI/MTS)
        features[0] = patient.severity / 5.0  # Normalized severity (1-5 scale)
        features[1] = patient.deterioration_chance  # Deterioration risk (0-1)
        features[2] = min(patient.wait_time / 20.0, 1.0)  # Normalized wait time
        
        # Fair environmental factors (observable from simulation)
        features[3] = (sim_state.get('current_time', 0) % 96) / 96.0  # Time of day cycle
        features[4] = min(sim_state.get('queue_length', 0) / 10.0, 1.0)  # Queue pressure
        features[5] = sim_state.get('nurse_availability', 0.5)  # Nurse availability ratio
            
        return features

    def activate(self, x, activation_type):
        """Apply activation function with numerical stability"""
        # Clip inputs to prevent overflow
        x = np.clip(x, -100, 100)
        
        if activation_type == 0:  # Sigmoid
            return 1 / (1 + np.exp(-x))
        elif activation_type == 1:  # Tanh
            return np.tanh(x)
        elif activation_type == 2:  # ReLU
            return np.maximum(0, x)
        else:
            return x

    def forward_pass(self, features, policy):
        """Forward pass through the neural network with error handling"""
        try:
            # Validate inputs
            if not isinstance(features, np.ndarray) or len(features) != self.input_size:
                return 1.0
                
            # Input to hidden layer
            hidden_input = np.dot(features, policy['w1']) + policy['b1']
            
            # Check for numerical issues with safer approach
            try:
                if not np.all(np.isfinite(hidden_input)):
                    return 1.0
            except (ValueError, TypeError, KeyboardInterrupt):
                return 1.0
            
            # Apply activations
            hidden_output = np.zeros_like(hidden_input)
            for i in range(len(hidden_input)):
                hidden_output[i] = self.activate(hidden_input[i], policy['activations'][i])
            
            # Check for numerical issues with safer approach
            try:
                if not np.all(np.isfinite(hidden_output)):
                    return 1.0
            except (ValueError, TypeError, KeyboardInterrupt):
                return 1.0
            
            # Hidden to output layer
            output_input = np.dot(hidden_output, policy['w2']) + policy['b2']
            
            # Check for numerical issues with safer approach
            try:
                if not np.all(np.isfinite(output_input)):
                    return 1.0
            except (ValueError, TypeError, KeyboardInterrupt):
                return 1.0
            
            # Final output (always positive for triage score)
            output = np.maximum(0.1, output_input[0]) * policy['output_scale']
            
            # Final sanity check with safer approach
            try:
                if not np.isfinite(output) or output <= 0:
                    return 1.0
            except (ValueError, TypeError, KeyboardInterrupt):
                return 1.0
                
            # Apply urgency boost for critical cases (using fair severity data)
            if len(features) > 0 and features[0] > 0.8:  # High severity (severity >= 4)
                output *= policy['urgency_boost']
                
            return float(np.clip(output, 0.1, 100.0))  # Reasonable bounds
            
        except Exception as e:
            # Return safe default if any error occurs
            return 1.0

    def fair_neural_triage_score(self, patient, sim_state, policy):
        """Calculate triage score using fair neural network"""
        if patient is None:
            return 0
            
        features = self.extract_fair_features(patient, sim_state)
        score = self.forward_pass(features, policy)
        
        return float(score)

    def mutate_fair_neural_policy(self, policy, mutation_rate=0.1):
        """Mutate neural network weights with different strategies"""
        new_policy = {}
        
        for key, value in policy.items():
            if key in ['w1', 'w2']:
                # Weight matrix mutation
                new_value = value.copy()
                mutation_mask = np.random.random(value.shape) < mutation_rate
                new_value[mutation_mask] += np.random.normal(0, 0.1, np.sum(mutation_mask))
                new_policy[key] = new_value
                
            elif key in ['b1', 'b2']:
                # Bias vector mutation
                new_value = value.copy()
                mutation_mask = np.random.random(value.shape) < mutation_rate
                new_value[mutation_mask] += np.random.normal(0, 0.05, np.sum(mutation_mask))
                new_policy[key] = new_value
                
            elif key == 'activations':
                # Activation function mutation
                new_value = value.copy()
                if random.random() < mutation_rate:
                    idx = random.randint(0, len(value) - 1)
                    new_value[idx] = random.randint(0, 2)
                new_policy[key] = new_value
                
            elif key in ['output_scale', 'urgency_boost']:
                # Scalar parameter mutation
                if random.random() < mutation_rate:
                    new_value = value + random.gauss(0, value * 0.1)
                    new_policy[key] = max(0.1, new_value)
                else:
                    new_policy[key] = value
            else:
                new_policy[key] = value
                
        return new_policy

    def crossover_fair_neural_policies(self, p1, p2):
        """Crossover two neural network policies"""
        child = {}
        
        for key in p1.keys():
            if key in ['w1', 'w2', 'b1', 'b2']:
                # Matrix/vector crossover
                if random.random() < 0.5:
                    # Uniform crossover
                    mask = np.random.random(p1[key].shape) < 0.5
                    child[key] = np.where(mask, p1[key], p2[key])
                else:
                    # Average crossover
                    alpha = random.uniform(0.3, 0.7)
                    child[key] = alpha * p1[key] + (1 - alpha) * p2[key]
                    
            elif key == 'activations':
                # Activation function crossover
                child[key] = np.array([
                    p1[key][i] if random.random() < 0.5 else p2[key][i]
                    for i in range(len(p1[key]))
                ])
                
            else:
                # Scalar parameter crossover
                child[key] = p1[key] if random.random() < 0.5 else p2[key]
                
        return child

    def evaluate_fair_neural_policy(self, policy):
        """Evaluate fair neural network policy"""
        def fair_neural_triage_function(patient):
            # Simple simulation state approximation using only observable data
            sim_state = {
                'current_time': getattr(fair_neural_triage_function, 'time', 0),
                'queue_length': getattr(fair_neural_triage_function, 'queue_len', 0),
                'nurse_availability': 0.5  # Reasonable default
            }
            return self.fair_neural_triage_score(patient, sim_state, policy)
        
        sim = ERSimulation(
            num_nurses=self.num_nurses,
            total_time=self.total_time,
            arrival_prob=self.arrival_prob,
            triage_policy=fair_neural_triage_function,
            verbose=False,
            seed=self.seed,
            use_shifts=True
        )
        
        try:
            result = sim.run()
            
            if result['avg_weighted_wait'] is None or not isinstance(result['avg_weighted_wait'], (int, float)):
                return 1000
                
            # Enhanced fitness function
            completion_rate = result['completed'] / max(1, result['completed'] + result['still_waiting'])
            base_fitness = result['avg_weighted_wait']
            
            # Sanity check for numerical values
            if not isinstance(base_fitness, (int, float)) or base_fitness < 0 or base_fitness > 10000:
                return 1000
                
            # Penalty for low completion
            completion_penalty = (1 - completion_rate) ** 2 * 40
            
            # Bonus for high completion with low wait
            efficiency_bonus = 0
            if completion_rate > 0.8 and base_fitness < 15:
                efficiency_bonus = -2
                
            final_fitness = base_fitness + completion_penalty + efficiency_bonus
            
            # Final sanity check
            if not isinstance(final_fitness, (int, float)) or final_fitness < 0:
                return 1000
                
            return final_fitness
            
        except Exception as e:
            # If any error occurs during evaluation, return high penalty
            return 1000

    def run(self, gen_log_path="fair_neural_evolution_log.txt"):
        """Run fair neural evolution optimization"""
        # Training started
        
        # Initialize population of fair neural networks
        population = [self.create_fair_neural_policy() for _ in range(self.population_size)]
        
        best_fitness_ever = float('inf')
        
        with open(gen_log_path, "w") as gen_log:
            gen_log.write("Fair Neural Evolution Optimization Log\n")
            gen_log.write("=" * 50 + "\n")
            gen_log.write(f"Fair network architecture: {self.input_size} -> {self.hidden_size} -> {self.output_size}\n")
            gen_log.write("Input features: severity, deterioration, wait_time, time_of_day, queue_length, nurse_availability\n\n")
            
            for gen in range(self.num_generations):
                # Progress bar: show training progress
                if (gen + 1) % 5 == 0 or gen == self.num_generations - 1:
                    bar_len = 40
                    progress = int(bar_len * (gen + 1) / self.num_generations)
                    bar = '[' + '#' * progress + '-' * (bar_len - progress) + ']'
                    print(f"Gen {gen+1:3d}/{self.num_generations} {bar}", end='\r', flush=True)
                
                # Evaluate all individuals
                fitnesses = [self.evaluate_fair_neural_policy(policy) for policy in population]
                
                # Find best individual
                best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
                best_fitness = fitnesses[best_idx]
                best_policy = population[best_idx]
                
                if best_fitness < best_fitness_ever:
                    best_fitness_ever = best_fitness
                
                # Adaptive mutation rate
                generation_ratio = gen / self.num_generations
                base_mutation_rate = 0.15 * (1 - generation_ratio * 0.5)
                
                # Logging
                if gen % 20 == 0 or gen == self.num_generations - 1:
                    avg_fitness = sum(fitnesses) / len(fitnesses)
                    gen_log.write(f"Generation {gen+1:3d}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}\n")
                    gen_log.write(f"  Mutation rate: {base_mutation_rate:.3f}\n")
                    gen_log.write(f"  Best policy output_scale: {best_policy['output_scale']:.3f}\n\n")
                
                # Selection and reproduction
                # Elite preservation (top 10%)
                elite_size = max(2, self.population_size // 10)
                elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:elite_size]
                elites = [population[i] for i in elite_indices]
                
                new_population = elites[:]
                
                while len(new_population) < self.population_size:
                    # Tournament selection
                    tournament_size = min(5, len(population))
                    tournament_indices = random.sample(range(len(population)), tournament_size)
                    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
                    
                    # Select parents
                    parent1_idx = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
                    remaining = [i for i in tournament_indices if i != parent1_idx]
                    parent2_idx = random.choice(remaining)
                    
                    parent1 = population[parent1_idx]
                    parent2 = population[parent2_idx]
                    
                    # Crossover
                    child = self.crossover_fair_neural_policies(parent1, parent2)
                    
                    # Mutation
                    child = self.mutate_fair_neural_policy(child, base_mutation_rate)
                    
                    new_population.append(child)
                
                population = new_population
            
            print(f"Gen {self.num_generations:3d}/{self.num_generations} [{'#'*40}] Done.{' '*10}")
            
            # Final evaluation with multiple seeds for robustness
            final_fitnesses = []
            for seed_offset in range(3):
                test_seed = self.seed + seed_offset * 100
                
                def final_fair_triage_function(patient):
                    sim_state = {
                        'current_time': getattr(final_fair_triage_function, 'time', 0),
                        'queue_length': getattr(final_fair_triage_function, 'queue_len', 0),
                        'nurse_availability': 0.5
                    }
                    return self.fair_neural_triage_score(patient, sim_state, best_policy)
                
                sim = ERSimulation(
                    num_nurses=self.num_nurses,
                    total_time=self.total_time,
                    arrival_prob=self.arrival_prob,
                    triage_policy=final_fair_triage_function,
                    verbose=False,
                    seed=test_seed,
                    use_shifts=True
                )
                
                result = sim.run()
                if result['avg_weighted_wait'] is not None:
                    final_fitnesses.append(result['avg_weighted_wait'])
            
            avg_final_fitness = sum(final_fitnesses) / len(final_fitnesses) if final_fitnesses else best_fitness_ever
            
            gen_log.write(f"\n{'='*50}\n")
            gen_log.write(f"FAIR NEURAL EVOLUTION COMPLETE\n")
            gen_log.write(f"Best training fitness: {best_fitness_ever:.3f}\n")
            gen_log.write(f"Average final test fitness: {avg_final_fitness:.3f}\n")
            gen_log.write(f"\nBest fair network architecture summary:\n")
            gen_log.write(f"  Hidden layer activations: {best_policy['activations']}\n")
            gen_log.write(f"  Output scale: {best_policy['output_scale']:.4f}\n")
            gen_log.write(f"  Urgency boost: {best_policy['urgency_boost']:.4f}\n")
            gen_log.write(f"  Weight matrix shapes: w1={best_policy['w1'].shape}, w2={best_policy['w2'].shape}\n")
        
        return best_policy


def create_fair_neural_triage_function(neural_policy, optimizer):
    """Create a fair triage function from a trained neural policy"""
    def triage_function(patient):
        if patient is None:
            return 0
        
        sim_state = {
            'current_time': getattr(triage_function, 'time', 0),
            'queue_length': getattr(triage_function, 'queue_len', 0),
            'nurse_availability': 0.5
        }
        
        return optimizer.fair_neural_triage_score(patient, sim_state, neural_policy)
    
    return triage_function