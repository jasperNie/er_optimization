# ER Triage Optimization

Final thesis repository for a simulation-based study of emergency department triage optimization. The codebase models ER operations, evolves learned triage policies, compares them with rule-based baselines, and stores the raw and processed outputs used for analysis.

## What the final project contains

This repository is the final working snapshot of the project, centered on four pieces:

1. **ER simulation** with 24-hour runs, 15-minute timesteps, shift-based nurse staffing, patient deterioration, and queue dynamics.
2. **Baseline triage policies** implementing ESI- and MTS-style rules for comparison.
3. **Learned triage policies** produced by evolutionary optimization, with final experiments focused on:
   - a pure neural triage policy
   - a hybrid neural policy with ESI fallback when confidence is low
4. **Evaluation and reporting pipeline** for batch experiments, log scraping, CSV summaries, and figure generation.

## Research scope

The final experiment set evaluates learned triage policies across:

- **6 arrival scenarios**: `standard`, `peak_hours`, `weekend`, `disaster`, `flu_season`, `steady_state`
- **5 staffing levels**: 2, 3, 4, 5, and 6 nurses
- **2 learned-policy families**: neural and hybrid

Training runs use seeds `8000-8049`, and the held-out evaluation runs use seeds `9000-9099`.

The completed batch outputs are stored in `logs/complete_evaluation/`, and the scraped summary tables are stored in `logs/scraped_analysis/`.

## Core files and folders

### Simulation and policies

- `classes.py` - core simulation objects: `Patient`, `Nurse`, and `ERSimulation`
- `arrival_patterns.py` - six scenario generators used in the final evaluation sweep
- `triage_policies.py` - rule-based comparison policies (`esi_policy` and `mts_policy`)

### Optimizers

- `optimizers/linear_elite_optimizer.py` - early linear evolutionary baseline
- `optimizers/linear_tournament_optimizer.py` - linear baseline with tournament selection
- `optimizers/advanced_optimizer.py` - larger hand-engineered feature model
- `optimizers/hybrid_optimizer.py` - multi-strategy evolutionary approach
- `optimizers/neural_optimizer.py` - evolutionary neural network optimizer used in the final experiments

### Final evaluation scripts

- `analysis/comprehensive_neural_pattern_evaluation.py` - detailed neural evaluation for one arrival pattern and staffing level
- `analysis/comprehensive_hybrid_pattern_evaluation.py` - detailed hybrid evaluation for one arrival pattern and staffing level
- `run_complete_evaluation.py` - runs the full thesis sweep across all patterns, staffing levels, and both policy families
- `analysis/simple_full_combination_neural_test.py` - broader generalization test on mixed-pattern scenarios for the neural policy
- `analysis/simple_full_combination_hybrid_test.py` - broader generalization test on mixed-pattern scenarios for the hybrid policy

### Result processing and outputs

- `analysis_log_scraper.py` - converts raw analysis logs into structured CSV files
- `logs/complete_evaluation/` - raw output logs from the full evaluation sweep
- `logs/scraped_analysis/` - consolidated CSV tables derived from raw logs
- `logs/scraped_analysis/comprehensive_analysis_table.csv` - main thesis summary table covering all 60 final configurations
- `logs/analysis_logs/` - detailed explanation logs for individual decisions, confidence levels, and hybrid fallback behavior
- `report_visualizations/` - exported figures organized by scenario and method

## Key thesis artifacts

If starting from the finished repository state, the most useful outputs are:

- `logs/scraped_analysis/comprehensive_analysis_table.csv` for the final cross-configuration comparison table
- `logs/analysis_logs/` for interpretable decision traces used to explain why one patient was prioritized over another
- `report_visualizations/evaluation_results/` for summary plots of weighted wait, policy comparisons, and scenario-specific results

The `comprehensive_analysis_table.csv` file summarizes the full `6 x 5 x 2 = 60` final evaluation grid and is the best single file to reference for thesis tables and appendix-level summaries.

## Core reported metrics

The final analysis primarily reports:

- **Weighted wait time**: the main optimization objective used to compare triage policies
- **Improvement vs ESI**: percentage reduction in weighted wait relative to the ESI-style baseline
- **Improvement vs MTS**: percentage reduction in weighted wait relative to the MTS-style baseline
- **Hybrid neural decision rate**: the share of hybrid decisions made by the learned model rather than the ESI fallback

These metrics appear throughout the scraped CSV summaries, raw evaluation logs, and generated figures.

### Archived development material

- `old_analysis/` - earlier experiment scripts retained for provenance after being moved out of the main analysis path

## Recommended reproduction workflow

Run all commands from the repository root.

### 1. Run the full thesis experiment sweep

```bash
python run_complete_evaluation.py
```

This launches the full grid of evaluations across all arrival patterns, staffing levels, and both learned-policy families.

### 2. Convert raw logs into analysis tables

```bash
python analysis_log_scraper.py --log-dir logs/complete_evaluation --output-dir logs/scraped_analysis
```

This produces CSV summaries that are easier to inspect, analyze, and plot.

### 3. Optional: run the smaller legacy benchmark harness

```bash
python run_evaluation.py
```

This older script compares the full optimizer set on a smaller evaluation setup. It is useful for sanity checks, but it is **not** the main thesis batch runner.

## Notes on repository organization

- This is a **script-first research repository**, not a packaged library.
- The most important files for thesis submission are the simulation code, the optimizer implementations, the final evaluation scripts, and the generated logs/figures.
- Historical duplicates and legacy analysis scripts have been moved or removed so that the root documentation reflects the final project state.

## Minimal environment expectations

The codebase is written in Python and uses common scientific packages used throughout the analysis scripts, including:

- `numpy`
- `matplotlib`
- `pandas` for some table-inspection utilities

## Submission summary

In short, this repository contains:

- the ER simulator,
- the baseline and learned triage policies,
- the final neural and hybrid evaluation pipeline,
- the raw experiment logs,
- scraped summary tables,
- and exported visualizations used to support the thesis results.