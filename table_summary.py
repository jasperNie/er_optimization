#!/usr/bin/env python3
"""
Quick summary of the comprehensive analysis table
"""

import pandas as pd

def print_table_summary():
    # Load the comprehensive table
    df = pd.read_csv('/Users/jaspernie/Desktop/er_optimization/logs/scraped_analysis/comprehensive_analysis_table.csv')
    
    print("="*80)
    print("COMPREHENSIVE ANALYSIS TABLE SUMMARY")
    print("="*80)
    
    print(f"Total rows (log files): {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    print(f"\nOptimizer types:")
    print(df['optimizer_type'].value_counts())
    
    print(f"\nPatterns:")
    print(df['pattern'].value_counts())
    
    print(f"\nNurse configurations:")
    print(sorted(df['nurses'].unique()))
    
    print(f"\nColumn categories:")
    
    # Basic info columns
    basic_cols = ['filename', 'pattern', 'nurses', 'optimizer_type', 'seed_count']
    print(f"  Basic info ({len(basic_cols)}): {basic_cols}")
    
    # Seed total columns
    total_cols = [col for col in df.columns if col.startswith('total_')]
    print(f"  Seed totals ({len(total_cols)}): {total_cols}")
    
    # Mean/aggregate columns
    mean_cols = [col for col in df.columns if 'mean_' in col or 'aggregate_' in col or 'neural_decision_rate' in col]
    print(f"  Aggregate stats ({len(mean_cols)}): {mean_cols}")
    
    # Baseline columns
    baseline_cols = [col for col in df.columns if 'baseline' in col or ('neural_' in col and 'wait' in col)]
    print(f"  Baseline comparisons ({len(baseline_cols)}): {baseline_cols}")
    
    # Performance detail columns
    perf_cols = [col for col in df.columns if 'perf_' in col]
    print(f"  Performance details ({len(perf_cols)}): {perf_cols}")
    
    # Improvement columns
    improvement_cols = [col for col in df.columns if 'improvement' in col]
    print(f"  Improvements ({len(improvement_cols)}): {improvement_cols}")
    
    print(f"\nSample data for first row:")
    print("-" * 60)
    first_row = df.iloc[0]
    for col in df.columns[:15]:  # Show first 15 columns
        print(f"  {col}: {first_row[col]}")
    print("  ... (and more columns)")
    
    print(f"\nHybrid-specific columns (for hybrid optimizer files):")
    hybrid_specific = ['total_neural_decisions', 'total_esi_fallbacks', 'neural_decision_rate']
    print(f"  {hybrid_specific}")
    
    print(f"\nFiles saved to:")
    print(f"  Main table: /Users/jaspernie/Desktop/er_optimization/logs/scraped_analysis/comprehensive_analysis_table.csv")
    print(f"  Detailed seed data: /Users/jaspernie/Desktop/er_optimization/logs/scraped_analysis/seed_level_data.csv")
    print(f"  Aggregate data: /Users/jaspernie/Desktop/er_optimization/logs/scraped_analysis/aggregate_data.csv")

if __name__ == '__main__':
    print_table_summary()