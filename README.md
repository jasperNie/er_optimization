# ER Optimization - Evolutionary Triage Algorithms

## Quick Start

### Running the Complete Evaluation
```bash
# Run all algorithms and compare against ESI/MTS baselines
python run_evaluation.py
```

This will:
1. Train all 5 evolutionary algorithms (100 generations each)
2. Test them on unseen scenarios 
3. Compare performance against medical baselines (ESI and MTS)
4. Show nurse schedules and detailed metrics

### Expected Output
```
=== ENHANCED ALGORITHM EVALUATION ===
Training and testing multiple advanced algorithms against ESI and MTS

Training on seeds: [1000, 1001, 1002]
Testing on seeds: [2000, 2001, 2002, 2003, 2004]

=== TRAINING PHASE ===
1. Training Linear Elite Algorithm...      ✓ Complete
2. Training Linear Tournament Algorithm... ✓ Complete  
3. Training Advanced Algorithm...          ✓ Complete
4. Training Hybrid Algorithm...            ✓ Complete
5. Training Neural Algorithm...            ✓ Complete

=== TESTING PHASE ===
[Individual algorithm results]

=== FINAL PERFORMANCE RANKING ===
Rank Algorithm       Weighted Wait Completed  Unattended  Combined Score
1    Neural          7.50          34.6       5.2         11.45
2    ESI             8.34          33.4       6.4         13.19
3    Advanced        8.66          32.4       7.4         14.26
4    MTS             10.67         34.4       5.4         14.77
...

=== IMPROVEMENT ANALYSIS ===
Comparison vs baselines:
  Neural         : ESI +10.1%, MTS +29.7% (score: 7.50)
  Advanced       : ESI  -3.8%, MTS +18.8% (score: 8.66)

🏆 WINNER: Neural
   Best weighted wait score: 7.50 (combined: 11.45)
   🎉 Successfully outperformed baseline algorithms!
```

## Understanding the Results

### Performance Metrics
- **Weighted Wait Time**: Primary metric - lower is better
  - Accounts for patient severity (critical patients weighted more heavily)
  - Includes deterioration penalties for delayed treatment
- **Completed Patients**: Number of patients successfully treated in 24 hours
- **Unattended Patients**: Patients left without treatment (penalty applied)
- **Combined Score**: Overall performance measure including unattended patient penalties
- **Improvement Analysis**: Shows percentage improvement/decline vs both ESI and MTS baselines

### Algorithm Performance
1. **Neural (WINNER)**: 7.50 score (+10.1% better than ESI, +29.7% better than MTS)
   - Uses neural network to learn non-linear patterns
   - Best at adapting to complex patient mix scenarios
   - Revolutionary breakthrough after critical bug fix
   
2. **ESI (Medical Baseline)**: 8.34 score
   - Standard Emergency Severity Index protocol
   - Used in most US hospitals
   
3. **Advanced**: 8.66 score (-3.8% vs ESI, +18.8% vs MTS)
   - Mathematical optimization with 13 parameters
   - Smart thresholds and non-linear scaling
   - Second-best evolutionary algorithm
   
4. **MTS (Medical Baseline)**: 10.67 score
   - Manchester Triage System protocol
   - Used in UK and European hospitals

### Reading Training Logs
Training progress is saved in `logs/` directory:

```bash
# View training progress for the winning algorithm
cat logs/train_neural.txt
```

Example log output:
```
Gen 1: Best=12.45, Avg=15.23, Worst=18.76
Gen 2: Best=11.89, Avg=14.56, Worst=17.23
...
Gen 100: Best=7.48, Avg=8.95, Worst=12.34
```

- **Best**: Best performing individual in the population
- **Avg**: Average performance across all individuals  
- **Worst**: Worst performing individual
- **Trend**: Should show improvement over generations

## Nurse Scheduling System

### Shift Structure
The simulation uses realistic shift-based nursing:

```
Day Shift:   07:00 - 19:00 (12 hours)
Night Shift: 19:00 - 07:00 (12 hours)
```

### Nurse Schedule Output
During evaluation, nurse schedules are printed showing:
- Shift assignments
- Patient assignments  
- Workload distribution
- Break times and availability

Example nurse schedule:
```
--- Nurse Schedule for Seed 2000 ---
Nurse 0 (Day Shift): 
  T0-T48: Available, T49-T65: Treating Patient 12, T66-T96: Available
Nurse 1 (Day Shift):
  T0-T25: Available, T26-T45: Treating Patient 8, T46-T96: Available  
Nurse 2 (Night Shift):
  T0-T48: Off Duty, T49-T72: Available, T73-T89: Treating Patient 23
Nurse 3 (Night Shift):
  T0-T48: Off Duty, T49-T96: Available
```

### Understanding Timesteps
- **96 timesteps total** = 24 hours (15 minutes per timestep)
- **T0-T47**: 00:00-11:45 (first 12 hours)
- **T48-T95**: 12:00-23:45 (second 12 hours)
- **Day shift active**: T28-T75 (07:00-19:00)
- **Night shift active**: T76-T27 (19:00-07:00, wrapping around)

## File Organization

### Core Files
- `run_evaluation.py` - Main script to run everything
- `classes.py` - Simulation engine (Patient, Nurse, ERSimulation)
- `triage_policies.py` - Baseline algorithms (ESI, MTS)

### Optimizers (5 algorithms)
- `optimizers/linear_elite_optimizer.py` - Simple linear approach
- `optimizers/linear_tournament_optimizer.py` - Better selection method
- `optimizers/advanced_optimizer.py` - Non-linear mathematical features
- `optimizers/hybrid_optimizer.py` - Multi-strategy evolution
- `optimizers/neural_optimizer.py` - **Winner** - Neural network evolution

### Evaluation & Logs
- `evaluations/enhanced_evaluation.py` - Comprehensive testing framework
- `logs/evaluation_results.txt` - Latest performance rankings and analysis
- `logs/train_*.txt` - Training progress for each algorithm
- `logs/patient_arrivals/` - Detailed patient arrival logs (training/testing)
- `logs/nurse_schedule_esi_example.txt` - Example nurse shift schedules

### Data Access Ethics
All algorithms use only **fair data** available to human nurses:
- ✅ Patient severity level (1-5)
- ✅ Observable deterioration rate  
- ✅ Current wait time
- ✅ Basic vital signs (HR, RR, SpO2, BP)
- ✅ Nurse availability and queue length
- ❌ Future patient arrivals
- ❌ Internal simulation parameters
- ❌ Treatment outcomes

## Troubleshooting

### Common Issues
1. **Import Errors**: Make sure you're running from the project root directory
2. **Missing numpy**: Install with `pip install numpy`
3. **Long Runtime**: Full evaluation takes 10-15 minutes for all algorithms

### Performance Expectations
- **Excellent Run**: Neural algorithm achieves 7.5 or better (current best: 7.50)
- **Good Run**: Any algorithm scoring below 8.5 weighted wait
- **Poor Run**: Score above 10.0 indicates potential issues
- **Training Convergence**: Best score should improve over 100 generations
- **Bug-Free**: Neural algorithm should outperform both medical baselines

### Customization Options
Edit `run_evaluation.py` to modify:
- Number of training generations (default: 100)
- Population sizes (default: 80-100)
- Test scenarios (default: 5 different seeds)
- Nurse staffing levels (default: 3-4 nurses)

## Next Steps
1. Run `python run_evaluation.py` to see the algorithms in action
2. Examine `logs/train_neural.txt` to see how the winner evolved
3. Check `PROJECT_SUMMARY.md` for detailed technical analysis
4. Consider extending training to 200+ generations for even better results