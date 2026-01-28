# ER Optimization Project

An evolutionary algorithm approach to optimizing Emergency Room triage policies with neural network explainability.

## 🏗️ Project Structure

```
er_optimization/
├── 📁 Core Files
│   ├── classes.py                 # Core simulation classes (Patient, Nurse, ERSimulation)
│   ├── triage_policies.py         # Baseline triage policy implementations  
│   ├── logging_utils.py           # Logging and utility functions
│   ├── enhanced_evaluation.py     # Main evaluation framework
│   └── run_evaluation.py          # Main evaluation script
│
├── 📁 optimizers/                 # Optimization algorithms
│   ├── linear_elite_optimizer.py      # Basic evolutionary algorithm
│   ├── linear_tournament_optimizer.py # Tournament selection variant
│   ├── advanced_optimizer.py          # Advanced feature engineering
│   ├── hybrid_optimizer.py            # Multi-strategy hybrid approach
│   └── neural_optimizer.py            # Neural network evolution (WINNER)
│
├── 📁 explainability/            # Neural network explainability tools
│   ├── neural_explainer.py           # Comprehensive decision analysis
│   └── full_run_explainer.py         # Complete simulation explanations
│
├── 📁 analysis/                  # Performance analysis scripts
│   └── comprehensive_neural_evaluation.py # Multi-seed evaluation
│
├── 📁 logs/                      # Training and evaluation logs
│   ├── evaluation_results.txt        # Main benchmark results
│   ├── comprehensive_neural_evaluation.txt # Multi-seed results
│   └── [various training logs]
│
└── 📁 Documentation
    ├── README.md                      # This file
    └── PROJECT_SUMMARY.md             # Detailed project summary
```

## 🚀 Quick Start

### Run Main Evaluation
```bash
python run_evaluation.py
```
Trains and evaluates all algorithms against ESI and MTS baselines.

### Explain Neural Network Decisions
```bash
# Comprehensive decision analysis
python explainability/neural_explainer.py

# Full simulation with explanations
python explainability/full_run_explainer.py
```

### Performance Analysis
```bash
# Multi-seed comprehensive evaluation
python analysis/comprehensive_neural_evaluation.py
```

## 🏆 Key Results

**Neural Network vs Traditional Methods:**
- **Neural beats ESI by 11.8%** in weighted wait time
- **Neural beats MTS by 3.0%** in weighted wait time  
- **Consistent performance** across multiple random seeds
- **Explainable decisions** with confidence scoring

**Performance Metrics (24-hour simulation):**
- Patients treated: 24.0 ± 0.9
- Average weighted wait: 4.00 ± 0.70
- Decision explanations: ~22 per simulation

## 🧠 Neural Network Insights

The winning neural network learned to:
1. **Balance medical urgency with wait time fairness**
2. **Prioritize patients with longer waits** when severity is similar
3. **Adapt to time-of-day patterns** (day/evening/night shifts)
4. **Make explainable decisions** with clear scoring rationale

**Example Decision Pattern:**
- Patient A: severity=5, wait=5min → Score: 0.45
- Patient B: severity=2, wait=20min → Score: 0.52 ✅ **Chosen**

The network learned that system-wide efficiency comes from balancing individual medical needs with overall fairness.

## 📊 Algorithm Comparison

| Algorithm | Weighted Wait | Performance vs ESI |
|-----------|---------------|-------------------|
| Neural Network | **4.00** | **+11.8%** |
| ESI (severity) | 4.52 | baseline |
| MTS (wait time) | 4.12 | +8.8% |
| Hybrid | 4.25 | +6.0% |
| Advanced | 4.35 | +3.8% |

## 🔬 Technical Details

**Neural Network Architecture:**
- Input: 6 features (severity, deterioration, wait time, time cycle, queue pressure, nurse availability)
- Hidden layer: Variable size with evolutionary activation functions
- Output: Triage priority score (0.1-100.0)
- Training: 100 generations, 80 population size

**Evaluation Methodology:**
- 24-hour simulations (96 timesteps)
- Multiple random seeds for robustness
- Real shift patterns (day/evening/night)
- Fair feature engineering (no privileged information)

## 📈 Future Work

1. **Real-world validation** with hospital data
2. **Multi-objective optimization** (wait time + patient outcomes)
3. **Online learning** for adaptation to changing conditions
4. **Integration with existing EHR systems**

---

*This project demonstrates how evolutionary algorithms can optimize complex healthcare systems while maintaining explainability and fairness.*