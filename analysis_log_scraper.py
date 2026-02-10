#!/usr/bin/env python3
"""
Analysis Log Scraper for ER Optimization Project

This script scrapes through all analysis logs in the complete_evaluation folder
and extracts key metrics from both individual seed results and aggregate statistics.

For neural logs: extracts standard metrics
For hybrid logs: extracts standard metrics + neural/ESI decision counts

Output: CSV files with all extracted data for easy analysis
"""

import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


class LogScraper:
    def __init__(self, log_directory: str):
        self.log_directory = Path(log_directory)
        self.seed_data = []
        self.aggregate_data = []
    
    def extract_seed_metrics(self, content: str, is_hybrid: bool = False) -> List[Dict]:
        """Extract metrics from individual seed results"""
        seed_results = []
        
        # Pattern to match seed results blocks
        seed_pattern = r'SEED (\d+) RESULTS:(.*?)(?=SEED \d+ RESULTS:|AGGREGATE STATISTICS:|$)'
        seed_matches = re.findall(seed_pattern, content, re.DOTALL)
        
        for seed_num, seed_content in seed_matches:
            metrics = {'seed': int(seed_num)}
            
            # Extract basic metrics
            patterns = {
                'patients_treated': r'Patients treated: (\d+)',
                'patients_waiting': r'Patients waiting: (\d+)',
                'average_wait': r'Average wait: ([\d.]+) timesteps',
                'weighted_wait': r'Weighted wait: ([\d.]+) timesteps',
                'decisions_explained': r'Decisions explained: (\d+)'
            }
            
            for metric, pattern in patterns.items():
                match = re.search(pattern, seed_content)
                metrics[metric] = int(float(match.group(1))) if match else 0
            
            # Extract hybrid-specific metrics
            if is_hybrid:
                neural_match = re.search(r'Neural decisions: (\d+) \(([\d.]+)%\)', seed_content)
                esi_match = re.search(r'ESI fallbacks: (\d+) \(([\d.]+)%\)', seed_content)
                
                if neural_match:
                    metrics['neural_decisions'] = int(neural_match.group(1))
                    metrics['neural_percentage'] = float(neural_match.group(2))
                if esi_match:
                    metrics['esi_fallbacks'] = int(esi_match.group(1))
                    metrics['esi_percentage'] = float(esi_match.group(2))
            
            # Extract ESI metrics
            esi_match = re.search(r'ESI treated: (\d+), waiting: (\d+) \| avg: ([\d.]+) timesteps.*?weighted: ([\d.]+)', seed_content)
            if esi_match:
                metrics['esi_treated'] = int(esi_match.group(1))
                metrics['esi_waiting'] = int(esi_match.group(2))
                metrics['esi_avg_wait'] = float(esi_match.group(3))
                metrics['esi_weighted_wait'] = float(esi_match.group(4))
            
            # Extract MTS metrics
            mts_match = re.search(r'MTS treated: (\d+), waiting: (\d+) \| avg: ([\d.]+) timesteps.*?weighted: ([\d.]+)', seed_content)
            if mts_match:
                metrics['mts_treated'] = int(mts_match.group(1))
                metrics['mts_waiting'] = int(mts_match.group(2))
                metrics['mts_avg_wait'] = float(mts_match.group(3))
                metrics['mts_weighted_wait'] = float(mts_match.group(4))
            
            seed_results.append(metrics)
        
        return seed_results
    
    def extract_aggregate_metrics(self, content: str, is_hybrid: bool = False) -> Dict:
        """Extract aggregate statistics from the log"""
        metrics = {}
        
        # Extract aggregate statistics
        agg_pattern = r'AGGREGATE STATISTICS:(.*?)BASELINE COMPARISON:'
        agg_match = re.search(agg_pattern, content, re.DOTALL)
        
        if agg_match:
            agg_content = agg_match.group(1)
            
            # Mean patients treated
            mean_treated_match = re.search(r'Mean patients treated: ([\d.]+) \+/- ([\d.]+)', agg_content)
            if mean_treated_match:
                metrics['mean_patients_treated'] = float(mean_treated_match.group(1))
                metrics['mean_patients_treated_std'] = float(mean_treated_match.group(2))
            
            # Mean weighted wait
            mean_wait_match = re.search(r'Mean weighted wait: ([\d.]+) \+/- ([\d.]+) hours', agg_content)
            if mean_wait_match:
                metrics['mean_weighted_wait_hours'] = float(mean_wait_match.group(1))
                metrics['mean_weighted_wait_std'] = float(mean_wait_match.group(2))
            
            # Total decisions explained
            decisions_match = re.search(r'Total decisions explained: (\d+)', agg_content)
            if decisions_match:
                metrics['total_decisions_explained'] = int(decisions_match.group(1))
        
        # Extract baseline comparison
        baseline_pattern = r'BASELINE COMPARISON:(.*?)BASELINE PERFORMANCE DETAILS:'
        baseline_match = re.search(baseline_pattern, content, re.DOTALL)
        
        if baseline_match:
            baseline_content = baseline_match.group(1)
            
            if is_hybrid:
                # Hybrid format
                hybrid_match = re.search(r'Hybrid Network: ([\d.]+) hours weighted \(([\d.]+) hours avg\)', baseline_content)
                if hybrid_match:
                    metrics['hybrid_weighted_wait'] = float(hybrid_match.group(1))
                    metrics['hybrid_avg_wait'] = float(hybrid_match.group(2))
                
                # Neural decision rate
                neural_rate_match = re.search(r'Neural decision rate: ([\d.]+)%', baseline_content)
                if neural_rate_match:
                    metrics['neural_decision_rate'] = float(neural_rate_match.group(1))
            else:
                # Neural format
                neural_match = re.search(r'Neural Network: ([\d.]+) hours weighted \(([\d.]+) hours avg\)', baseline_content)
                if neural_match:
                    metrics['neural_weighted_wait'] = float(neural_match.group(1))
                    metrics['neural_avg_wait'] = float(neural_match.group(2))
            
            # ESI and MTS baselines
            esi_match = re.search(r'ESI Baseline: ([\d.]+) hours weighted \(([\d.]+) hours avg\)', baseline_content)
            if esi_match:
                metrics['esi_baseline_weighted'] = float(esi_match.group(1))
                metrics['esi_baseline_avg'] = float(esi_match.group(2))
            
            mts_match = re.search(r'MTS Baseline: ([\d.]+) hours weighted \(([\d.]+) hours avg\)', baseline_content)
            if mts_match:
                metrics['mts_baseline_weighted'] = float(mts_match.group(1))
                metrics['mts_baseline_avg'] = float(mts_match.group(2))
        
        # Extract baseline performance details
        perf_pattern = r'BASELINE PERFORMANCE DETAILS:(.*?)$'
        perf_match = re.search(perf_pattern, content, re.DOTALL)
        
        if perf_match:
            perf_content = perf_match.group(1)
            
            # ESI performance details
            esi_perf_match = re.search(r'ESI treated: ([\d.]+), waiting: ([\d.]+) \| avg: ([\d.]+) timesteps.*?weighted: ([\d.]+)', perf_content)
            if esi_perf_match:
                metrics['esi_perf_treated'] = float(esi_perf_match.group(1))
                metrics['esi_perf_waiting'] = float(esi_perf_match.group(2))
                metrics['esi_perf_avg'] = float(esi_perf_match.group(3))
                metrics['esi_perf_weighted'] = float(esi_perf_match.group(4))
            
            # MTS performance details
            mts_perf_match = re.search(r'MTS treated: ([\d.]+), waiting: ([\d.]+) \| avg: ([\d.]+) timesteps.*?weighted: ([\d.]+)', perf_content)
            if mts_perf_match:
                metrics['mts_perf_treated'] = float(mts_perf_match.group(1))
                metrics['mts_perf_waiting'] = float(mts_perf_match.group(2))
                metrics['mts_perf_avg'] = float(mts_perf_match.group(3))
                metrics['mts_perf_weighted'] = float(mts_perf_match.group(4))
            
            # Performance improvements
            esi_improvement_match = re.search(r'-> Neural beats ESI by ([\d.]+)%', perf_content)
            if esi_improvement_match:
                metrics['esi_improvement_percent'] = float(esi_improvement_match.group(1))
            
            mts_improvement_match = re.search(r'-> Neural beats MTS by ([\d.]+)%', perf_content)
            if mts_improvement_match:
                metrics['mts_improvement_percent'] = float(mts_improvement_match.group(1))
        
        return metrics
    
    def parse_filename(self, filename: str) -> Tuple[str, str, str]:
        """Parse filename to extract pattern, nurses, and optimizer type"""
        # Remove _analysis.txt suffix
        base_name = filename.replace('_analysis.txt', '')
        
        # Extract pattern and nurses
        parts = base_name.split('_')
        if len(parts) >= 2:
            # Last part should be like "2nurses", "3nurses", etc.
            nurses_part = parts[-1]
            pattern = '_'.join(parts[:-1])
            
            # Extract number of nurses
            nurses_match = re.search(r'(\d+)nurses?', nurses_part)
            nurses = nurses_match.group(1) if nurses_match else 'unknown'
            
            return pattern, nurses
        
        return 'unknown', 'unknown'
    
    def process_log_file(self, filepath: Path):
        """Process a single log file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine if this is a hybrid log
            is_hybrid = '/hybrid/' in str(filepath)
            optimizer_type = 'hybrid' if is_hybrid else 'neural'
            
            # Parse filename
            pattern, nurses = self.parse_filename(filepath.name)
            
            # Extract seed-level data
            seed_results = self.extract_seed_metrics(content, is_hybrid)
            for seed_data in seed_results:
                seed_data.update({
                    'filename': f"{optimizer_type}_{filepath.name}",  # Make filename unique
                    'original_filename': filepath.name,
                    'pattern': pattern,
                    'nurses': nurses,
                    'optimizer_type': optimizer_type
                })
                self.seed_data.append(seed_data)
            
            # Extract aggregate data
            agg_metrics = self.extract_aggregate_metrics(content, is_hybrid)
            agg_metrics.update({
                'filename': f"{optimizer_type}_{filepath.name}",  # Make filename unique
                'original_filename': filepath.name,
                'pattern': pattern,
                'nurses': nurses,
                'optimizer_type': optimizer_type
            })
            self.aggregate_data.append(agg_metrics)
            
            print(f"Processed: {filepath.name} ({len(seed_results)} seeds)")
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    def scrape_all_logs(self):
        """Scrape all analysis logs in the directory"""
        log_files = list(self.log_directory.rglob('*analysis.txt'))
        
        print(f"Found {len(log_files)} analysis log files")
        
        for log_file in sorted(log_files):
            self.process_log_file(log_file)
    
    def calculate_seed_totals(self) -> List[Dict]:
        """Calculate totals for each log file from seed data"""
        log_totals = {}
        
        # Group seeds by filename
        for seed in self.seed_data:
            filename = seed['filename']
            if filename not in log_totals:
                log_totals[filename] = {
                    'filename': filename,
                    'pattern': seed['pattern'],
                    'nurses': seed['nurses'],
                    'optimizer_type': seed['optimizer_type'],
                    'total_patients_treated': 0,
                    'total_patients_waiting': 0,
                    'total_decisions_explained': 0,
                    'total_esi_treated': 0,
                    'total_esi_waiting': 0,
                    'total_mts_treated': 0,
                    'total_mts_waiting': 0,
                    'seed_count': 0
                }
                
                # Add hybrid-specific totals
                if seed['optimizer_type'] == 'hybrid':
                    log_totals[filename]['total_neural_decisions'] = 0
                    log_totals[filename]['total_esi_fallbacks'] = 0
            
            # Sum up the totals
            log_totals[filename]['total_patients_treated'] += seed.get('patients_treated', 0)
            log_totals[filename]['total_patients_waiting'] += seed.get('patients_waiting', 0)
            log_totals[filename]['total_decisions_explained'] += seed.get('decisions_explained', 0)
            log_totals[filename]['total_esi_treated'] += seed.get('esi_treated', 0)
            log_totals[filename]['total_esi_waiting'] += seed.get('esi_waiting', 0)
            log_totals[filename]['total_mts_treated'] += seed.get('mts_treated', 0)
            log_totals[filename]['total_mts_waiting'] += seed.get('mts_waiting', 0)
            log_totals[filename]['seed_count'] += 1
            
            if seed['optimizer_type'] == 'hybrid':
                log_totals[filename]['total_neural_decisions'] += seed.get('neural_decisions', 0)
                log_totals[filename]['total_esi_fallbacks'] += seed.get('esi_fallbacks', 0)
        
        return list(log_totals.values())
    
    def create_comprehensive_table(self) -> List[Dict]:
        """Create a comprehensive table combining totals and aggregate stats"""
        seed_totals = self.calculate_seed_totals()
        
        # Create comprehensive table by combining totals with aggregate data
        comprehensive_table = []
        
        for totals in seed_totals:
            filename = totals['filename']
            
            # Find corresponding aggregate data
            agg_data = next((agg for agg in self.aggregate_data if agg['filename'] == filename), {})
            
            # Combine all data
            row = {
                # Basic info
                'filename': filename,
                'pattern': totals['pattern'],
                'nurses': int(totals['nurses']) if totals['nurses'].isdigit() else totals['nurses'],
                'optimizer_type': totals['optimizer_type'],
                
                # Seed totals
                'total_patients_treated': totals['total_patients_treated'],
                'total_patients_waiting': totals['total_patients_waiting'],
                'total_decisions_explained': totals['total_decisions_explained'],
                'total_esi_treated': totals['total_esi_treated'],
                'total_esi_waiting': totals['total_esi_waiting'],
                'total_mts_treated': totals['total_mts_treated'],
                'total_mts_waiting': totals['total_mts_waiting'],
                
                # Aggregate statistics
                'mean_patients_treated': agg_data.get('mean_patients_treated', 0),
                'mean_patients_treated_std': agg_data.get('mean_patients_treated_std', 0),
                'mean_weighted_wait_hours': agg_data.get('mean_weighted_wait_hours', 0),
                'mean_weighted_wait_std': agg_data.get('mean_weighted_wait_std', 0),
                'aggregate_total_decisions': agg_data.get('total_decisions_explained', 0),
                
                # Baseline comparisons
                'neural_weighted_wait': agg_data.get('neural_weighted_wait') or agg_data.get('hybrid_weighted_wait', 0),
                'neural_avg_wait': agg_data.get('neural_avg_wait') or agg_data.get('hybrid_avg_wait', 0),
                'esi_baseline_weighted': agg_data.get('esi_baseline_weighted', 0),
                'esi_baseline_avg': agg_data.get('esi_baseline_avg', 0),
                'mts_baseline_weighted': agg_data.get('mts_baseline_weighted', 0),
                'mts_baseline_avg': agg_data.get('mts_baseline_avg', 0),
                
                # Baseline performance details
                'esi_perf_treated': agg_data.get('esi_perf_treated', 0),
                'esi_perf_waiting': agg_data.get('esi_perf_waiting', 0),
                'esi_perf_avg': agg_data.get('esi_perf_avg', 0),
                'esi_perf_weighted': agg_data.get('esi_perf_weighted', 0),
                'mts_perf_treated': agg_data.get('mts_perf_treated', 0),
                'mts_perf_waiting': agg_data.get('mts_perf_waiting', 0),
                'mts_perf_avg': agg_data.get('mts_perf_avg', 0),
                'mts_perf_weighted': agg_data.get('mts_perf_weighted', 0),
                
                # Improvement percentages
                'esi_improvement_percent': agg_data.get('esi_improvement_percent', 0),
                'mts_improvement_percent': agg_data.get('mts_improvement_percent', 0),
            }
            
            # Add hybrid-specific columns
            if totals['optimizer_type'] == 'hybrid':
                row['total_neural_decisions'] = totals['total_neural_decisions']
                row['total_esi_fallbacks'] = totals['total_esi_fallbacks']
                row['neural_decision_rate'] = agg_data.get('neural_decision_rate', 0)
            else:
                row['total_neural_decisions'] = 'N/A'
                row['total_esi_fallbacks'] = 'N/A' 
                row['neural_decision_rate'] = 'N/A'
            
            comprehensive_table.append(row)
        
        return comprehensive_table
    
    def save_to_csv(self, output_dir: str = None):
        """Save extracted data to CSV files"""
        if output_dir is None:
            output_dir = str(self.log_directory.parent / 'scraped_analysis')
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create and save comprehensive table
        comprehensive_table = self.create_comprehensive_table()
        
        if comprehensive_table:
            # Define column order for better readability
            column_order = [
                'filename', 'pattern', 'nurses', 'optimizer_type',
                # Seed totals
                'total_patients_treated', 'total_patients_waiting', 'total_decisions_explained',
                'total_esi_treated', 'total_esi_waiting', 'total_mts_treated', 'total_mts_waiting',
                'total_neural_decisions', 'total_esi_fallbacks',
                # Aggregate stats
                'mean_patients_treated', 'mean_patients_treated_std',
                'mean_weighted_wait_hours', 'mean_weighted_wait_std',
                'aggregate_total_decisions', 'neural_decision_rate',
                # Baseline comparisons
                'neural_weighted_wait', 'neural_avg_wait',
                'esi_baseline_weighted', 'esi_baseline_avg',
                'mts_baseline_weighted', 'mts_baseline_avg',
                # Performance details
                'esi_perf_treated', 'esi_perf_waiting', 'esi_perf_avg', 'esi_perf_weighted',
                'mts_perf_treated', 'mts_perf_waiting', 'mts_perf_avg', 'mts_perf_weighted',
                # Improvements
                'esi_improvement_percent', 'mts_improvement_percent'
            ]
            
            # Sort by pattern, then nurses, then optimizer type
            comprehensive_table.sort(key=lambda x: (x['pattern'], x['nurses'], x['optimizer_type']))
            
            comp_csv_path = output_path / 'comprehensive_analysis_table.csv'
            
            with open(comp_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=column_order)
                writer.writeheader()
                writer.writerows(comprehensive_table)
            
            print(f"Saved comprehensive table to: {comp_csv_path}")
            print(f"Table contains {len(comprehensive_table)} log files with all metrics")
        
        # Also save the original separate files for detailed analysis
        if self.seed_data:
            seed_csv_path = output_path / 'seed_level_data.csv'
            fieldnames = set()
            for row in self.seed_data:
                fieldnames.update(row.keys())
            
            with open(seed_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                writer.writeheader()
                writer.writerows(self.seed_data)
            
            print(f"Saved detailed seed-level data to: {seed_csv_path}")
        
        if self.aggregate_data:
            agg_csv_path = output_path / 'aggregate_data.csv'
            fieldnames = set()
            for row in self.aggregate_data:
                fieldnames.update(row.keys())
            
            with open(agg_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                writer.writeheader()
                writer.writerows(self.aggregate_data)
            
            print(f"Saved aggregate statistics to: {agg_csv_path}")
    
    def print_summary(self):
        """Print a summary of the scraped data"""
        print("\n" + "="*60)
        print("SCRAPING SUMMARY")
        print("="*60)
        
        print(f"Total log files processed: {len(self.aggregate_data)}")
        print(f"Total seeds extracted: {len(self.seed_data)}")
        
        # Count by optimizer type
        neural_files = len([d for d in self.aggregate_data if d['optimizer_type'] == 'neural'])
        hybrid_files = len([d for d in self.aggregate_data if d['optimizer_type'] == 'hybrid'])
        
        print(f"Neural optimizer logs: {neural_files}")
        print(f"Hybrid optimizer logs: {hybrid_files}")
        
        # Count by pattern
        patterns = set(d['pattern'] for d in self.aggregate_data)
        print(f"Unique patterns: {len(patterns)}")
        for pattern in sorted(patterns):
            count = len([d for d in self.aggregate_data if d['pattern'] == pattern])
            print(f"  {pattern}: {count} files")
        
        # Count by nurses
        nurses = set(d['nurses'] for d in self.aggregate_data)
        print(f"Nurse configurations: {sorted(nurses)}")


def main():
    parser = argparse.ArgumentParser(description='Scrape analysis logs from ER optimization project')
    parser.add_argument('--log-dir', 
                       default='/Users/jaspernie/Desktop/er_optimization/logs/complete_evaluation',
                       help='Directory containing analysis logs')
    parser.add_argument('--output-dir', 
                       help='Output directory for CSV files (default: log_dir/../scraped_analysis)')
    
    args = parser.parse_args()
    
    scraper = LogScraper(args.log_dir)
    scraper.scrape_all_logs()
    scraper.save_to_csv(args.output_dir)
    scraper.print_summary()


if __name__ == '__main__':
    main()