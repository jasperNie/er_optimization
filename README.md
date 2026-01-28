# Emergency Room Optimization with Hybrid Triage System

A sophisticated emergency room simulation and optimization framework that combines neural network intelligence with clinical decision validation to improve patient triage while maintaining clinical safety.

## 🏥 Project Overview

This project implements a **Hybrid Confidence Triage Policy** that:
- Uses neural networks to make intelligent triage decisions
- Falls back to Emergency Severity Index (ESI) when clinical safety is at risk
- Provides comprehensive performance analysis across severity levels
- Maintains detailed decision logging for clinical audit

## 🎯 Key Features

### Core Components
- **ERSimulation**: Complete emergency room environment with configurable resources
- **FairNeuralEvolutionOptimizer**: Evolutionary training for neural triage networks
- **HybridConfidenceTiagePolicy**: Clinical validation with confidence-based ESI fallback
- **Comprehensive Evaluation**: Multi-seed testing with severity-specific metrics

### Clinical Safety Features
- **Confidence-based ESI Fallback**: Automatically triggers safer ESI decisions when neural confidence is low
- **Clinical Decision Validation**: Detects and prevents dangerous neural network choices
- **Severity-aware Performance**: Tracks wait times by patient severity (1-5 scale)
- **Decision Audit Trail**: Complete logging of all triage decisions with explanations

## 🚀 Quick Start

### Running the Hybrid Evaluation
```bash
python analysis/proper_hybrid_evaluation.py
```

This will:
1. Train a neural network using evolutionary optimization
2. Test the hybrid policy across multiple random seeds
3. Generate comprehensive performance reports
4. Save detailed logs for clinical analysis

### Key Results
- **6.5% better** than ESI baseline (weighted wait times)
- **29.6% better** than Manchester Triage System
- **Clinical safety preserved** through validation system
- **-30.1% performance** for Severity 5 patients vs ESI (acceptable trade-off for overall gains)

## 📊 Performance Analysis

The system tracks:
- **Overall Wait Times**: Average and weighted by severity
- **Severity-Specific Performance**: Individual analysis for each severity level
- **Decision Confidence**: Neural network certainty in each choice
- **ESI Fallback Rate**: Frequency of safety-triggered overrides

## 🏗️ Project Structure

```
er_optimization/
├── 📁 Core System
│   ├── classes.py                 # Core simulation engine (Patient, Nurse, ERSimulation)
│   ├── triage_policies.py         # Hybrid triage implementation with clinical validation
│   └── logging_utils.py           # Logging and utility functions
│
├── 📁 optimizers/                 # Optimization algorithm evolution
│   ├── linear_elite_optimizer.py      # Basic evolutionary algorithm
│   ├── linear_tournament_optimizer.py # Tournament selection variant  
│   ├── advanced_optimizer.py          # Enhanced feature engineering
│   ├── hybrid_optimizer.py            # Multi-strategy approach
│   └── neural_optimizer.py            # Neural network evolution (FINAL)
│
├── 📁 analysis/                   # Evaluation frameworks
│   ├── comprehensive_neural_evaluation.py # Neural network testing
│   └── proper_hybrid_evaluation.py       # Final hybrid system (CURRENT)
│
├── 📁 docs/                       # Documentation and research history
│   └── development_history/            # Complete development evolution
│
├── 📁 logs/                       # Performance results and training logs
│   ├── analysis_logs/                  # Final evaluation results
│   └── patient_arrivals/               # Detailed simulation data
│
└── 📁 Legacy Scripts               # Development and testing scripts
    ├── enhanced_evaluation.py          # Previous evaluation framework
    └── run_evaluation.py               # Original evaluation script
```

## 📈 Clinical Impact

The hybrid system successfully balances:
- **Efficiency**: Better overall performance than traditional systems
- **Safety**: Clinical validation prevents dangerous AI decisions
- **Transparency**: Full audit trail of all triage decisions
- **Adaptability**: Confidence thresholds adjust based on clinical context

## 🔬 Research Applications

This framework enables research in:
- AI-assisted medical decision making
- Clinical safety validation systems
- Emergency department optimization
- Human-AI collaboration in healthcare

## 📚 Development Documentation

The `docs/development_history/` folder contains complete documentation of:
- **Algorithm Evolution**: From linear to neural network approaches
- **Performance Comparisons**: Detailed analysis of each optimization strategy
- **Clinical Safety Development**: Journey toward medically-safe AI systems
- **Research Insights**: Key learnings and failed approaches

This documentation is valuable for understanding the complete research process and can be referenced in academic reports and papers.

## 📝 License

This project is designed for research and educational purposes in healthcare AI and emergency medicine optimization.