# ER Triage Optimization System

A modular Python system for optimizing Emergency Room triage policies using evolutionary algorithms and comparing different triage strategies with realistic nursing shift schedules.

## Project Structure

```
er_optimization/
├── classes.py                  # Core classes (Patient, Nurse, ERSimulation)
├── evolutionary_optimizer.py   # Evolutionary algorithm for policy optimization  
├── triage_policies.py          # MTS and ESI triage policy implementations
├── simulation_runner.py        # High-level simulation running functions (unused)
├── logging_utils.py            # Utilities for logging and reporting (unused)
├── main.py                     # Main execution script - runs original functionality
├── .gitignore                  # Git ignore file for Python cache and temp files
└── README.md                   # This file
```

## Module Overview

### `classes.py`
Contains the core simulation classes:
- **Patient**: Represents ER patients with severity, deterioration, treatment time, vitals, and presenting symptoms
- **Nurse**: Represents nursing staff with availability tracking and patient assignments
- **ERSimulation**: Main simulation engine with realistic 8-hour nursing shifts, handoff protocols, and patient flow

### `evolutionary_optimizer.py`
- **EvolutionaryTriageOptimizer**: Genetic algorithm to evolve optimal triage policies
- Uses population-based optimization to find best weights for severity, deterioration, and wait time
- Generates `generation_log.txt` with optimization progress

### `triage_policies.py`
Implements standard triage policies:
- **MTS Policy**: Modified Triage Score based on severity and wait time
- **ESI Policy**: Emergency Severity Index focusing on urgency levels with vital signs
- **Policy Comparison**: Function to compare different policies side-by-side

### `main.py` 
Main execution script that runs the complete optimization workflow:
1. Runs evolutionary algorithm (100 generations, population 100)
2. Tests evolved policy against MTS and ESI policies
3. Generates comprehensive logs and comparisons
4. Uses realistic patient arrivals with vital signs and presenting symptoms

## Quick Start

1. **Run the complete optimization system:**
   ```bash
   python main.py
   ```
   This will:
   - Run evolutionary optimization for 100 generations
   - Compare evolved policy with MTS and ESI policies
   - Generate detailed logs and nurse schedules
   - Print comprehensive results to console

2. **Use individual modules (optional):**
   ```python
   from evolutionary_optimizer import EvolutionaryTriageOptimizer
   from classes import ERSimulation
   from triage_policies import mts_policy, esi_policy
   
   # Run evolutionary optimization
   optimizer = EvolutionaryTriageOptimizer(
       num_generations=100,
       population_size=100,
       num_nurses=5,
       total_time=96,  # 24 hours in 15-minute timesteps
       arrival_prob=0.5,
       seed=2025
   )
   
   best_policy = optimizer.run()
   
   # Run single simulation with nursing shifts
   sim = ERSimulation(
       num_nurses=5,
       total_time=96,
       arrival_prob=0.5,
       triage_policy=best_policy,
       verbose=True,
       use_shifts=True
   )
   
   result = sim.run()
   ```

## Key Features

### Realistic Nursing Shifts
- **8-hour shifts**: Day (7 nurses), Evening (5 nurses), Night (3 nurses)
- **Variable staffing levels**: Day shift has 1.5x staff, Night shift has 0.7x staff
- **Handoff protocols**: Nurses stay to complete current patient treatments during shift changes
- **Shift scheduling**: Based on 15-minute timesteps (96 timesteps = 24 hours)

### Advanced Patient Modeling  
- **Realistic arrivals**: Time-varying patterns with daily cycles and random bursts
- **Patient deterioration**: Severity increases over time based on deterioration rates
- **Vital signs**: Heart rate, respiratory rate, oxygen saturation, blood pressure
- **Presenting symptoms**: Chest pain, abdominal pain, shortness of breath, etc.
- **Treatment variability**: Treatment time correlates with severity level

### Policy Optimization
- **Evolutionary algorithm**: 100 generations with population of 100 policies
- **Three-way comparison**: Evolved policy vs MTS vs ESI triage protocols
- **Weight optimization**: Finds optimal weights for severity, deterioration, and wait time
- **Performance metrics**: Average wait time, weighted wait time, completion rates

### Comprehensive Logging
- **generation_log.txt**: Evolutionary algorithm progress and best policies per generation
- **arrivals_log.txt**: Complete patient arrival log with attributes
- **nurse_schedule_log.txt**: Detailed nursing shift schedule with timestamps
- **Console output**: Policy comparison results and unattended patient statistics

## Configuration

Current simulation parameters (in `main.py`):
- **num_nurses**: 5 (base staffing level, scaled by shift multipliers)
- **total_time**: 96 timesteps (24 hours in 15-minute intervals)
- **arrival_prob**: 0.3 (base probability, varies with time and bursts)
- **seed**: 2025 (for reproducible results)
- **generations**: 100 (evolutionary algorithm iterations)
- **population_size**: 100 (number of policies per generation)

### Shift Staffing:
- **Day shift (00:00-08:00)**: 7 nurses (5 × 1.5 = 7.5 → 7)
- **Evening shift (08:00-16:00)**: 5 nurses (5 × 1.0 = 5)
- **Night shift (16:00-24:00)**: 3 nurses (5 × 0.7 = 3.5 → 3)

## Output Files

The system generates three main log files:
- **`generation_log.txt`**: Evolutionary optimization progress, best fitness per generation
- **`arrivals_log.txt`**: Complete patient arrival log with vitals and symptoms  
- **`nurse_schedule_log.txt`**: 24-hour nursing schedule with shift changes marked

## Example Run

```bash
python main.py
```

**Output:**
1. **Evolutionary Progress**: Loading bar showing 100 generations of optimization
2. **Policy Comparison**: Results for Optimized, MTS, and ESI policies showing:
   - Number of completed patients
   - Patients still waiting  
   - Average wait time
   - Maximum wait time
   - Average weighted wait time (accounts for severity)
3. **Unattended Patients**: List of patients not treated within 24 hours
4. **Log Files**: Detailed logs for analysis and debugging

## Understanding the Results

### Policy Weights (Evolved Policy)
The evolutionary algorithm optimizes three weights:
- **severity**: How much to prioritize high-severity patients
- **deterioration**: How much to consider deterioration risk
- **wait_time**: How much to factor in waiting time

### Performance Metrics
- **completed**: Number of patients fully treated
- **still_waiting**: Patients in queue at simulation end
- **avg_wait**: Average time patients waited before treatment
- **max_wait**: Longest wait time experienced
- **avg_weighted_wait**: Wait time weighted by severity × deterioration (fitness metric)

### Typical Results
- **Evolved Policy**: Usually performs best on weighted wait time
- **MTS Policy**: Balances severity and wait time effectively  
- **ESI Policy**: Focuses heavily on vital signs and urgency levels

## Extending the System

### Adding New Triage Policies
1. Define your policy function in `main.py` (like `mts_policy` and `esi_policy`)
2. Add it to the policy comparison loop
3. Policy should take a patient and return a priority score

### Modifying Parameters
1. Edit values in `main.py` for different scenarios:
   - Change `num_nurses` for different hospital sizes
   - Adjust `arrival_prob` for busier/quieter ERs
   - Modify shift multipliers in `classes.py` for different staffing patterns

### Advanced Analysis
- Export log files to Excel/CSV for statistical analysis
- Plot arrival patterns and wait times over 24-hour cycles
- Analyze shift performance and handoff effectiveness