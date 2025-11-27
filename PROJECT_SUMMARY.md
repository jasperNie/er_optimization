# ER Optimization Project Summary

## Project Overview
This project develops and evaluates evolutionary algorithms for optimizing Emergency Room (ER) triage policies. The goal is to create automated decision-making systems that outperform standard medical triage protocols (ESI and MTS) in reducing patient wait times while maintaining fair treatment.

## Problem Statement
Traditional triage systems like ESI (Emergency Severity Index) and MTS (Manchester Triage System) rely on fixed rules that may not adapt optimally to varying patient loads and staffing conditions. Our evolutionary algorithms learn optimal triage strategies that minimize weighted wait times across different severity levels.

## Core Architecture

### Simulation Engine (classes.py)
- **ERSimulation**: Main simulation environment modeling 24-hour ER operations
- **Patient**: Individual patient objects with severity, deterioration rate, treatment time
- **Nurse**: Staff members with shift-based scheduling and availability tracking
- **Key Features**:
  - 96 timesteps (15-minute intervals over 24 hours)
  - Dynamic patient arrivals with realistic patterns (time-of-day, burst events)
  - Shift-based nursing staff (7am-7pm day shift, 7pm-7am night shift)
  - Patient deterioration over time
  - Resource allocation and treatment tracking

### Optimization Algorithms (optimizers/)

#### 1. Linear Elite Optimizer (linear_elite_optimizer.py)
- **Parameters**: 3 total
  - `severity`: Weight for patient severity level (0.5-2.0)
  - `deterioration`: Weight for deterioration chance (0.5-2.0)  
  - `wait_time`: Weight for current wait time (0.5-2.0)
- **Evolution Strategy**: Elite selection (top 20% survive)
- **Population**: 100 individuals, 100 generations
- **Triage Function**: Linear combination: `severity*w1 + deterioration*w2 + wait_time*w3`

#### 2. Linear Tournament Optimizer (linear_tournament_optimizer.py)
- **Parameters**: 3 total (identical to Linear Elite)
  - `severity`: Weight for patient severity level (0.5-2.0)
  - `deterioration`: Weight for deterioration chance (0.5-2.0)
  - `wait_time`: Weight for current wait time (0.5-2.0)
- **Evolution Strategy**: Tournament selection (random competition)
- **Population**: 100 individuals, 100 generations
- **Improvement**: Better diversity preservation vs pure elitism

#### 3. Advanced Optimizer (advanced_optimizer.py)
- **Parameters**: 13 total
  - Core weights:
    - `severity`: Patient severity weight (0.1-5.0)
    - `deterioration`: Deterioration chance weight (0.1-5.0)
    - `wait_time`: Wait time weight (0.1-3.0)
  - Advanced factors:
    - `urgency_multiplier`: Boost for high severity patients (1.0-4.0)
    - `deterioration_threshold`: When to apply deterioration boost (0.2-0.8)
    - `deterioration_boost`: Deterioration penalty multiplier (1.2-3.0)
    - `wait_threshold`: Critical wait time threshold (3-15 timesteps)
    - `wait_penalty_factor`: Exponential wait penalty (1.1-2.5)
    - `severity_threshold`: Critical severity threshold (3.5-4.5)
  - Queue dynamics:
    - `queue_pressure`: Queue length influence (0.0-2.0)
    - `efficiency_factor`: Efficiency vs thoroughness balance (0.8-1.5)
  - Time-based adjustments:
    - `day_shift_bonus`: 6AM-2PM adjustment (0.8-1.2)
    - `evening_shift_bonus`: 2PM-10PM adjustment (0.9-1.1)
    - `night_shift_penalty`: 10PM-6AM adjustment (1.0-1.3)
  - Non-linear scaling:
    - `severity_power`: Non-linear severity scaling (0.8-2.0)
    - `deterioration_power`: Non-linear deterioration scaling (0.8-2.0)
- **Population**: 100 individuals, 100 generations

#### 4. Hybrid Optimizer (hybrid_optimizer.py)
- **Parameters**: 80-120 total (varies by strategy mix)
- **Strategy Components** (5 total strategies):
  1. **Severity-First Strategy** (7 parameters):
     - `severity`: High priority weight (3.0-5.0)
     - `deterioration`: Moderate weight (1.0-2.5)
     - `wait_time`: Low priority (0.1-1.0)
     - `severity_threshold`: Critical severity point (3.5-4.5)
     - `emergency_multiplier`: Emergency boost factor (2.0-4.0)
  2. **Deterioration-First Strategy** (7 parameters):
     - `severity`: Moderate weight (1.5-3.0)
     - `deterioration`: High priority weight (3.0-5.0)
     - `wait_time`: Moderate weight (0.5-1.5)
     - `deterioration_threshold`: Prevention threshold (0.3-0.7)
     - `prevention_bonus`: Prevention bonus multiplier (1.5-3.0)
  3. **Balanced Strategy** (5 parameters):
     - `severity`: Balanced weight (1.5-2.5)
     - `deterioration`: Balanced weight (1.5-2.5)
     - `wait_time`: Balanced weight (0.8-1.8)
     - `balance_factor`: Overall balance adjustment (0.8-1.2)
  4. **Wait-Focused Strategy** (7 parameters):
     - `severity`: Lower weight (1.0-2.0)
     - `deterioration`: Lower weight (1.0-2.0)
     - `wait_time`: High priority weight (2.0-4.0)
     - `wait_threshold`: Wait time threshold (5-15)
     - `efficiency_bonus`: Efficiency multiplier (1.2-2.0)
  5. **Smart-Hybrid Strategy** (8 parameters):
     - `severity`: Adaptive weight (1.8-3.2)
     - `deterioration`: Adaptive weight (1.8-3.2)
     - `wait_time`: Adaptive weight (0.8-2.2)
     - `adaptive_factor`: Adaptation strength (0.5-1.5)
     - `queue_sensitivity`: Queue responsiveness (0.3-1.0)
     - `time_sensitivity`: Time-of-day sensitivity (0.2-0.8)
- **Co-evolution**: Multiple strategies compete and adapt simultaneously
- **Population**: 100 individuals, 150 generations

#### 5. Neural Optimizer (neural_optimizer.py) 🏆 **WINNER**
- **Parameters**: 43 total
  - **Neural Network Architecture**: 6 inputs → 6 hidden neurons → 1 output
  - Weight matrices:
    - `w1`: Input-to-hidden weights (6×6 = 36 parameters)
    - `w2`: Hidden-to-output weights (6×1 = 6 parameters)
  - Bias vectors:
    - `b1`: Hidden layer biases (6 parameters)
    - `b2`: Output bias (1 parameter)
  - Activation functions:
    - `activations`: Per-neuron activation types (6 parameters: sigmoid/tanh/ReLU)
  - Scaling factors:
    - `output_scale`: Global output scaling (0.5-2.0)
    - `urgency_boost`: Urgency amplification (1.0-3.0)
- **Evolutionary Mechanisms**:
  - **Population**: 80 neural networks compete simultaneously
  - **Selection**: Tournament selection (best of 5 random individuals)
  - **Crossover**: Two methods for neural networks:
    - Uniform crossover: Random neuron-by-neuron mixing of weights/biases
    - Average crossover: Weighted averaging of parent networks (α=0.3-0.7)
  - **Mutation**: Multi-strategy mutation:
    - Weight matrices: Gaussian noise added to random weights (σ=0.1)
    - Bias vectors: Gaussian noise added to random biases (σ=0.05) 
    - Activation functions: Random switching between sigmoid/tanh/ReLU
    - Scaling factors: Gaussian perturbation (σ=10% of current value)
  - **Elite Preservation**: Top 10% survive each generation unchanged
  - **Adaptive Mutation**: Rate decreases from 15% to 7.5% over 100 generations
- **Inputs**: severity, deterioration, wait_time, time_of_day, queue_length, nurse_availability
- **Features**:
  - **Evolutionary Neural Networks**: Uses genetic algorithms to evolve neural network weights rather than gradient descent
  - Non-linear pattern recognition through evolved neural architectures
  - Multiple activation functions evolved per neuron
  - Adaptive feature importance through evolved weights
  - Dynamic output scaling and urgency boosting
- **Population**: 80 individuals, 100 generations  
- **Achievement**: First algorithm to significantly outperform both medical baselines

### Fair Data Access Policy
All algorithms are restricted to the same information available to human triage nurses:
- Patient severity level (1-5)
- Deterioration rate (observable decline)
- Current wait time
- Basic vital signs (HR, RR, SpO2, BP)
- Observable simulation state (nurse availability, queue length)

**Prohibited Data**: Future patient arrivals, internal simulation parameters, treatment outcomes

## Performance Results

### Final Rankings (Weighted Wait Time - Lower is Better)
1. **Neural**: 7.50 🏆 **(+10.1% improvement over ESI, +29.7% over MTS)**
2. **ESI** (Medical Baseline): 8.34
3. **Advanced**: 8.66 (-3.8% vs ESI, +18.8% vs MTS)
4. **MTS** (Medical Baseline): 10.67
5. **Hybrid**: 11.58 (-38.8% vs ESI, -8.5% vs MTS)
6. **Linear Tournament**: 12.48 (-49.7% vs ESI, -17.0% vs MTS)
7. **Linear Elite**: 12.53 (-50.2% vs ESI, -17.4% vs MTS)

### Key Insights
- **Neural Evolution** successfully learned non-linear patterns that outperform both medical baselines
- **Critical Bug Fix**: Neural algorithm performance improved dramatically after fixing input validation bug
- **Baseline Comparison**: Now includes comprehensive analysis against both ESI and MTS standards
- Traditional linear approaches (Elite, Tournament) performed poorly vs complex triage decisions
- Advanced mathematical features improved performance but couldn't match neural adaptability
- Hybrid multi-strategy approach showed promise but suffered from parameter complexity
- **Unattended Patient Tracking**: Enhanced evaluation now monitors and penalizes patients left untreated

## Training Methodology
- **Training Seeds**: [1000, 1001, 1002] - Used during algorithm evolution
- **Testing Seeds**: [2000, 2001, 2002, 2003, 2004] - Completely separate evaluation
- **Scenarios**: Each seed generates unique patient arrival patterns, severity distributions
- **Validation**: Strict separation ensures no overfitting to training scenarios

## Technical Implementation
- **Language**: Python 3
- **Dependencies**: numpy (neural networks), random, math
- **Parallel Processing**: Each generation evaluates population members independently
- **Logging**: Detailed generation-by-generation progress tracking
- **Reproducibility**: Seed-based randomization for consistent results

## File Structure
```
er_optimization/
├── classes.py                 # Core simulation engine
├── triage_policies.py         # ESI/MTS baseline implementations
├── run_evaluation.py          # Main execution script
├── optimizers/
│   ├── linear_elite_optimizer.py      # Basic evolutionary approach
│   ├── linear_tournament_optimizer.py # Tournament selection variant
│   ├── advanced_optimizer.py          # Non-linear mathematical features
│   ├── hybrid_optimizer.py            # Multi-strategy co-evolution
│   └── neural_optimizer.py            # Neural network evolution (WINNER)
├── evaluations/
│   └── enhanced_evaluation.py         # Comprehensive testing framework
├── logs/                              # Training progress logs
└── old_files/                         # Deprecated implementations
```

## Success Metrics
✅ **Primary Goal Achieved**: Created algorithms that outperform ESI and MTS
✅ **Fairness Maintained**: All algorithms use only ethically available data
✅ **Reproducible Results**: Consistent performance across multiple test scenarios
✅ **Practical Implementation**: Real-time triage decisions under 96-timestep simulation

## Future Enhancements
- Extended training (200+ generations) for further optimization
- Multi-objective optimization (wait time + safety + fairness)
- Real-world validation with actual ER data
- Integration with hospital information systems
- Adaptive algorithms that retrain based on local patient patterns