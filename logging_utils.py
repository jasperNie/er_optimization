import json
from datetime import datetime
from classes import ERSimulation
from evolutionary_optimizer import EvolutionaryTriageOptimizer
from triage_policies import compare_policies


def log_simulation_results(results, filename, policy_name="Unknown", timestamp=None):
    """Log simulation results to a file"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Simulation Results - {policy_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*50}\n")
        
        for key, value in results.items():
            if value is not None:
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: None\n")
        
        f.write("\n")


def log_policy_comparison(results, filename="policy_comparison_log.txt"):
    """Log policy comparison results"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Policy Comparison Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*60}\n")
        
        for policy_name, result in results.items():
            f.write(f"\n{policy_name} Policy:\n")
            f.write("-" * 20 + "\n")
            for key, value in result.items():
                if value is not None:
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.2f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {key}: None\n")
        
        f.write("\n")


def log_evolutionary_progress(best_policy, filename="evolution_summary_log.txt"):
    """Log evolutionary algorithm results"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Evolutionary Optimization Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Best evolved policy:\n")
        for key, value in best_policy.items():
            f.write(f"  {key}: {value:.4f}\n")
        f.write("\n")


def export_results_to_json(results, filename, metadata=None):
    """Export results to JSON format for further analysis"""
    timestamp = datetime.now().isoformat()
    
    output_data = {
        "timestamp": timestamp,
        "results": results
    }
    
    if metadata:
        output_data["metadata"] = metadata
    
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)


def create_summary_report(evolution_results=None, comparison_results=None, filename="simulation_summary_report.txt"):
    """Create a comprehensive summary report of all simulations"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, "w") as f:
        f.write("ER TRIAGE OPTIMIZATION SUMMARY REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {timestamp}\n\n")
        
        if evolution_results:
            f.write("EVOLUTIONARY OPTIMIZATION RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Best evolved policy:\n")
            for key, value in evolution_results.items():
                f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")
        
        if comparison_results:
            f.write("POLICY COMPARISON RESULTS\n")
            f.write("-" * 40 + "\n")
            
            best_policy = None
            best_score = float('inf')
            
            for policy_name, result in comparison_results.items():
                f.write(f"\n{policy_name} Policy Results:\n")
                for key, value in result.items():
                    if value is not None:
                        if isinstance(value, float):
                            f.write(f"  {key}: {value:.2f}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {key}: None\n")
                
                # Track best performing policy
                if result.get('avg_weighted_wait') and result['avg_weighted_wait'] < best_score:
                    best_score = result['avg_weighted_wait']
                    best_policy = policy_name
            
            if best_policy:
                f.write(f"\nBEST PERFORMING POLICY: {best_policy}\n")
                f.write(f"Best avg_weighted_wait: {best_score:.2f}\n")
        
        f.write("\nRECOMMendations:\n")
        f.write("-" * 20 + "\n")
        f.write("- Use the best performing policy for production ER triage\n")
        f.write("- Consider running longer simulations for more stable results\n")
        f.write("- Monitor real-world performance and adjust policies as needed\n")


# Convenience functions for different logging scenarios
def log_single_run(sim_result, policy_name, filename="single_run_log.txt"):
    """Log a single simulation run"""
    log_simulation_results(sim_result, filename, policy_name)


def log_and_export_comparison(comparison_results, base_filename="comparison"):
    """Log comparison results to both text and JSON"""
    log_policy_comparison(comparison_results, f"{base_filename}_log.txt")
    export_results_to_json(comparison_results, f"{base_filename}_results.json", 
                          metadata={"type": "policy_comparison"})


def log_full_optimization_run(evolution_results, comparison_results, base_filename="full_optimization"):
    """Log complete optimization run with all results"""
    # Individual logs
    log_evolutionary_progress(evolution_results, f"{base_filename}_evolution_log.txt")
    log_policy_comparison(comparison_results, f"{base_filename}_comparison_log.txt")
    
    # Summary report
    create_summary_report(evolution_results, comparison_results, f"{base_filename}_summary.txt")
    
    # JSON exports
    export_results_to_json(evolution_results, f"{base_filename}_evolution.json", 
                          metadata={"type": "evolutionary_optimization"})
    export_results_to_json(comparison_results, f"{base_filename}_comparison.json", 
                          metadata={"type": "policy_comparison"})