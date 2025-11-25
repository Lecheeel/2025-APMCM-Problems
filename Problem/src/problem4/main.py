"""
Problem 4: Problem Scale Reduction under Quantum Hardware Constraints
=======================================================================

Main script for implementing reduction strategies and solving reduced QUBO model.

Kaiwu SDK Documentation:
- English: https://kaiwu-sdk-docs.qboson.com/en/latest/
- Chinese: https://kaiwu-sdk-docs.qboson.com/zh/v1.2.0/source/getting_started/index.html

Usage:
    python main.py [--max-binary-vars N] [--num-bits N] [--time-aggregation METHOD] 
                   [--optimizer {simulated_annealing,cim}] [--results-dir DIR]
"""

import argparse
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict
import sys

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import modules
from problem4.data_loader import ReducedDataLoader
from problem4.reduction_strategy import ReductionStrategy
from problem4.discretization import GenerationDiscretizer
from problem4.qubo_builder import ReducedQUBOBuilder
from problem4.solver import ReducedQUBOSolver
from problem4.verifier import ReducedSolutionVerifier

try:
    import kaiwu as kw
except ImportError:
    print("ERROR: Kaiwu SDK is not installed.")
    print("Please install Kaiwu SDK:")
    print("  pip install kaiwu")
    print("Or install from wheel file:")
    print("  pip install ../kaiwu/kaiwu-*.whl")
    sys.exit(1)


def expand_solution_to_full_scale(reduced_generation: np.ndarray, 
                                   reduction_info: Dict,
                                   original_data: ReducedDataLoader) -> np.ndarray:
    """
    Expand reduced solution to full-scale (24 periods, 6 units).
    
    Args:
        reduced_generation: Generation matrix for reduced problem
        reduction_info: Reduction information dictionary
        original_data: Original data loader
        
    Returns:
        Full-scale generation matrix (6 units, 24 periods)
    """
    selected_units = reduction_info['selected_units']
    selected_periods = reduction_info['selected_periods']
    period_mapping = reduction_info['period_mapping']
    
    # Create full-scale generation matrix
    full_generation = np.zeros((original_data.num_units, original_data.num_periods))
    
    # Map reduced solution to full scale
    for orig_unit_idx in range(original_data.num_units):
        if orig_unit_idx in selected_units:
            # This unit is in reduced problem
            reduced_unit_idx = selected_units.index(orig_unit_idx)
            for orig_period in range(original_data.num_periods):
                mapped_period = period_mapping[orig_period]
                if mapped_period in selected_periods:
                    reduced_period_idx = selected_periods.index(mapped_period)
                    full_generation[orig_unit_idx, orig_period] = reduced_generation[reduced_unit_idx, reduced_period_idx]
                else:
                    # Use nearest period value
                    distances = [abs(orig_period - sp) for sp in selected_periods]
                    nearest_idx = np.argmin(distances)
                    reduced_period_idx = nearest_idx
                    full_generation[orig_unit_idx, orig_period] = reduced_generation[reduced_unit_idx, reduced_period_idx]
        else:
            # Unit not in reduced problem: use average from Problem 2 or P_min
            # For simplicity, use P_min (can be improved with Problem 2 reference)
            full_generation[orig_unit_idx, :] = original_data.P_min[orig_unit_idx]
    
    return full_generation


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Problem 4: Problem Scale Reduction under Quantum Hardware Constraints',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--max-binary-vars',
        type=int,
        default=100,
        help='Maximum number of binary variables allowed (default: 100, conservative CIM limit)'
    )
    parser.add_argument(
        '--num-bits',
        type=int,
        default=1,
        choices=[1, 2],
        help='Number of bits per unit per period (default: 1)'
    )
    parser.add_argument(
        '--time-aggregation',
        type=str,
        choices=['uniform', 'peak_valley', None],
        default=None,
        help='Time aggregation method (default: None, auto-select)'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['simulated_annealing', 'cim'],
        default='simulated_annealing',
        help='Optimizer type (default: simulated_annealing)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory containing Problem 2 results (default: ../results/problem2)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: ../results/problem4)'
    )
    parser.add_argument(
        '--num-solutions',
        type=int,
        default=3,
        help='Number of solutions to find (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Set default directories
    if args.results_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        args.results_dir = os.path.join(project_root, 'results', 'problem2')
    
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        args.output_dir = os.path.join(project_root, 'results', 'problem4')
    
    print("=" * 70)
    print("Problem 4: Problem Scale Reduction under Quantum Hardware Constraints")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    data_loader = ReducedDataLoader(results_dir=args.results_dir)
    summary = data_loader.get_summary()
    print(f"  Units: {summary['num_units']}")
    print(f"  Periods: {summary['num_periods']}")
    print(f"  Total capacity: {summary['total_capacity']:.1f} MW")
    print(f"  Load range: {summary['load_range'][0]:.1f} - {summary['load_range'][1]:.1f} MW")
    print(f"  All units online: {summary['all_units_online']}")
    
    # Step 2: Apply reduction strategy
    print("\n[Step 2] Applying reduction strategy...")
    reduction_strategy = ReductionStrategy(data_loader)
    
    # Ensure we stay within CIM bit capacity limit
    # Target: ≤ max_binary_vars (default 100)
    args.num_bits = 1  # Use 1-bit to minimize variables
    
    # Calculate maximum periods we can fit with all units
    max_periods_with_all_units = args.max_binary_vars // (summary['num_units'] * args.num_bits)
    print(f"  Target: ≤ {args.max_binary_vars} binary variables")
    print(f"  Configuration: {summary['num_units']} units × {summary['num_periods']} periods × {args.num_bits} bit = {summary['num_units'] * summary['num_periods'] * args.num_bits} binary variables")
    
    if summary['num_units'] * summary['num_periods'] * args.num_bits > args.max_binary_vars:
        print(f"  ⚠ Exceeds limit: Need to reduce to ≤ {args.max_binary_vars} variables")
        print(f"  Strategy: Keep all {summary['num_units']} units, reduce periods to {max_periods_with_all_units}")
        # Use uniform aggregation to reduce periods
        args.time_aggregation = 'uniform'
    else:
        print(f"  ✓ Within limit: No reduction needed")
        args.time_aggregation = None
    
    reduction_info = reduction_strategy.apply_reduction(
        max_binary_vars=args.max_binary_vars,
        num_bits=args.num_bits,
        time_aggregation=args.time_aggregation,
        unit_selection=None,  # Keep all units if possible
        force_no_reduction=False  # Allow reduction to meet CIM limits
    )
    
    # Verify we're within limits
    if reduction_info['total_binary_vars'] > args.max_binary_vars:
        print(f"  ⚠ Still exceeds limit: {reduction_info['total_binary_vars']} > {args.max_binary_vars}")
        print(f"  Applying additional reduction...")
        # Force further reduction: reduce periods
        max_periods = max(1, args.max_binary_vars // (len(reduction_info['selected_units']) * args.num_bits))
        reduction_strategy.selected_units = reduction_info['selected_units']
        reduction_strategy.selected_periods = reduction_strategy._select_key_periods(max_periods)
        reduction_strategy.period_mapping = reduction_strategy._create_period_mapping()
        reduction_info['selected_periods'] = reduction_strategy.selected_periods
        reduction_info['period_mapping'] = reduction_strategy.period_mapping
        reduction_info['total_binary_vars'] = len(reduction_info['selected_units']) * len(reduction_info['selected_periods']) * args.num_bits
        reduction_info['reduction_ratio']['periods'] = len(reduction_info['selected_periods']) / summary['num_periods']
        reduction_info['reduction_ratio']['total'] = reduction_info['total_binary_vars'] / (summary['num_units'] * summary['num_periods'] * 2)
    
    # Final verification
    if reduction_info['total_binary_vars'] > args.max_binary_vars:
        print(f"  ✗ ERROR: Still exceeds limit after reduction: {reduction_info['total_binary_vars']} > {args.max_binary_vars}")
        print(f"  This should not happen. Please check reduction logic.")
    else:
        print(f"  ✓ Verified: {reduction_info['total_binary_vars']} ≤ {args.max_binary_vars} (within CIM limit)")
    
    print(f"  Selected units: {reduction_info['selected_units']}")
    print(f"  Selected periods: {reduction_info['selected_periods']}")
    print(f"  Number of bits: {reduction_info['num_bits']}")
    print(f"  Total binary variables: {reduction_info['total_binary_vars']}")
    print(f"  Reduction ratios:")
    print(f"    Units: {reduction_info['reduction_ratio']['units']:.2%}")
    print(f"    Periods: {reduction_info['reduction_ratio']['periods']:.2%}")
    print(f"    Total: {reduction_info['reduction_ratio']['total']:.2%}")
    
    # Step 3: Get reduced data
    print("\n[Step 3] Preparing reduced data...")
    reduced_data = reduction_strategy.get_reduced_data()
    print(f"  Reduced units: {reduced_data['num_units']}")
    print(f"  Reduced periods: {reduced_data['num_periods']}")
    
    # Step 4: Initialize discretizer
    print("\n[Step 4] Initializing discretizer...")
    discretizer = GenerationDiscretizer(
        P_min=reduced_data['P_min'],
        P_max=reduced_data['P_max'],
        num_bits=reduction_info['num_bits']
    )
    
    # Step 5: Build QUBO model
    print("\n[Step 5] Building QUBO model...")
    try:
        qubo_builder = ReducedQUBOBuilder(reduced_data, discretizer)
        qubo_model = qubo_builder.build_qubo_model()
        print("  ✓ QUBO model built successfully")
    except Exception as e:
        print(f"  ✗ Error building QUBO model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Solve QUBO
    print("\n[Step 6] Solving QUBO model...")
    try:
        solver = ReducedQUBOSolver(qubo_model, discretizer, reduced_data, 
                                   optimizer_type=args.optimizer)
        solutions = solver.solve(num_solutions=args.num_solutions)
        print("  ✓ QUBO solved successfully")
    except Exception as e:
        print(f"  ✗ Error solving QUBO: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 7: Extract and verify solutions
    print("\n[Step 7] Extracting and verifying solutions...")
    verifier = ReducedSolutionVerifier(reduced_data)
    
    best_solution = None
    best_cost = float('inf')
    best_generation = None
    best_score = float('inf')
    
    for idx, solution in enumerate(solutions):
        print(f"\n  Solution {idx + 1}:")
        
        # Extract generation
        generation = solver.extract_generation(solution)
        
        # Calculate cost
        cost = verifier.calculate_cost(generation)
        print(f"    Total cost: ${cost:.2f}")
        
        # Verify constraints
        verification = verifier.verify_all(generation)
        
        # Calculate constraint violation score
        pb_violation = verification['power_balance']['max_violation'] if not verification['power_balance']['is_valid'] else 0.0
        ramp_violations = len(verification['ramp_constraints']['violations'])
        reserve_violations = len(verification['reserve_constraints']['violations'])
        
        constraint_penalty = (pb_violation * 2000.0 + 
                             ramp_violations * 5000.0 + 
                             reserve_violations * 3000.0)
        score = cost + constraint_penalty
        
        print(f"    Power balance: {'✓' if verification['power_balance']['is_valid'] else '✗'}")
        if not verification['power_balance']['is_valid']:
            print(f"      Max violation: {verification['power_balance']['max_violation']:.3f} MW")
        
        print(f"    Generation limits: {'✓' if verification['generation_limits']['is_valid'] else '✗'}")
        print(f"    Ramp constraints: {'✓' if verification['ramp_constraints']['is_valid'] else '✗'}")
        if not verification['ramp_constraints']['is_valid']:
            print(f"      Violations: {ramp_violations}")
        print(f"    Reserve constraints: {'✓' if verification['reserve_constraints']['is_valid'] else '✗'}")
        print(f"    Score (cost + violations): {score:.2f}")
        
        if score < best_score:
            best_score = score
            best_cost = cost
            best_solution = solution
            best_generation = generation
    
    # Step 8: Expand solution to full scale
    print("\n[Step 8] Expanding solution to full scale...")
    full_generation = expand_solution_to_full_scale(
        best_generation, reduction_info, data_loader
    )
    print("  ✓ Solution expanded to full scale (6 units, 24 periods)")
    
    # Calculate full-scale cost
    full_scale_cost = 0.0
    for i in range(data_loader.num_units):
        for t in range(data_loader.num_periods):
            p = full_generation[i, t]
            fuel_cost = (data_loader.a_coeff[i] * p**2 + 
                        data_loader.b_coeff[i] * p + 
                        data_loader.c_coeff[i])
            full_scale_cost += fuel_cost
    
    print(f"  Full-scale cost (24 periods): ${full_scale_cost:.2f}")
    print(f"  Reduced problem cost ({len(reduction_info['selected_periods'])} periods): ${best_cost:.2f}")
    
    # Step 9: Compare with Problem 2 (if available)
    print("\n[Step 9] Comparing with Problem 2 results...")
    if data_loader.problem2_results:
        problem2_cost = data_loader.problem2_results.get('optimization_info', {}).get('total_cost', None)
        if problem2_cost:
            print(f"  Full-scale QUBO cost: ${full_scale_cost:.2f}")
            print(f"  Problem 2 cost (continuous): ${problem2_cost:.2f}")
            cost_diff = full_scale_cost - problem2_cost
            cost_ratio = full_scale_cost / problem2_cost if problem2_cost > 0 else 1.0
            print(f"  Cost difference: ${cost_diff:.2f} ({cost_diff/problem2_cost*100:.2f}%)")
            print(f"  Cost ratio: {cost_ratio:.4f}")
            if full_scale_cost < problem2_cost * 0.9:
                print(f"  ⚠ Warning: Full-scale cost is significantly lower than Problem 2.")
                print(f"    This may indicate an issue with solution expansion or cost calculation.")
    else:
        print("  Problem 2 results not available for comparison")
    
    # Step 10: Save results
    print("\n[Step 10] Saving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save reduced generation schedule
    generation_data = []
    for t in range(reduced_data['num_periods']):
        row = {'Period': t + 1, 'Load_MW': float(reduced_data['load_demand'][t])}
        for i, unit_id in enumerate(reduced_data['units']):
            row[f'Unit_{unit_id}_Generation_MW'] = float(best_generation[i, t])
        row['Total_Generation_MW'] = float(np.sum(best_generation[:, t]))
        generation_data.append(row)
    
    import csv
    csv_path = os.path.join(args.output_dir, f"reduced_generation_schedule_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        if generation_data:
            writer = csv.DictWriter(f, fieldnames=generation_data[0].keys())
            writer.writeheader()
            writer.writerows(generation_data)
    print(f"  ✓ Reduced generation schedule saved to: {csv_path}")
    
    # Save full-scale generation schedule
    full_generation_data = []
    for t in range(data_loader.num_periods):
        row = {'Period': t + 1, 'Load_MW': float(data_loader.load_demand[t])}
        for i, unit_id in enumerate(data_loader.units):
            row[f'Unit_{unit_id}_Generation_MW'] = float(full_generation[i, t])
        row['Total_Generation_MW'] = float(np.sum(full_generation[:, t]))
        full_generation_data.append(row)
    
    full_csv_path = os.path.join(args.output_dir, f"full_generation_schedule_{timestamp}.csv")
    with open(full_csv_path, 'w', newline='') as f:
        if full_generation_data:
            writer = csv.DictWriter(f, fieldnames=full_generation_data[0].keys())
            writer.writeheader()
            writer.writerows(full_generation_data)
    print(f"  ✓ Full-scale generation schedule saved to: {full_csv_path}")
    
    # Save summary
    reduction_info_with_limit = reduction_info.copy()
    reduction_info_with_limit['max_binary_vars'] = args.max_binary_vars
    
    summary_data = {
        'optimization_info': {
            'timestamp': timestamp,
            'reduced_problem_cost': float(best_cost),
            'full_scale_cost': float(full_scale_cost),
            'num_units': reduced_data['num_units'],
            'num_periods': reduced_data['num_periods'],
            'num_binary_vars': reduction_info['total_binary_vars'],
            'num_bits_per_unit': reduction_info['num_bits'],
            'optimizer_type': args.optimizer,
            'reduction_info': reduction_info_with_limit,
            'verification': verification
        },
        'reduced_generation': best_generation.tolist(),
        'full_generation': full_generation.tolist()
    }
    
    json_path = os.path.join(args.output_dir, f"summary_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Summary saved to: {json_path}")
    
    # Step 11: Generate advanced visualizations
    print("\n[Step 11] Generating advanced visualizations...")
    try:
        from problem4.advanced_visualizations import create_advanced_visualizations
        create_advanced_visualizations(
            results_dir=args.output_dir,
            timestamp=timestamp,
            problem2_results_dir=args.results_dir
        )
        print("  ✓ Advanced visualizations generated successfully")
    except ImportError as e:
        print(f"  ⚠ Visualization libraries not available: {e}")
        print("  Skipping visualization generation")
    except Exception as e:
        print(f"  ⚠ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Problem 4 completed successfully!")
    print("=" * 70)
    print(f"\nKey Results:")
    print(f"  Binary variables: {reduction_info['total_binary_vars']} (reduced from {data_loader.num_units * data_loader.num_periods * 2})")
    print(f"  CIM limit: {args.max_binary_vars} variables")
    print(f"  Status: {'✓ Within limit' if reduction_info['total_binary_vars'] <= args.max_binary_vars else '✗ Exceeds limit'}")
    print(f"  Reduction ratio: {reduction_info['reduction_ratio']['total']:.2%}")
    print(f"  Units: {len(reduction_info['selected_units'])}/{data_loader.num_units} ({reduction_info['reduction_ratio']['units']:.2%})")
    print(f"  Periods: {len(reduction_info['selected_periods'])}/{data_loader.num_periods} ({reduction_info['reduction_ratio']['periods']:.2%})")
    print(f"  Reduced problem cost ({len(reduction_info['selected_periods'])} periods): ${best_cost:.2f}")
    print(f"  Full-scale cost (24 periods): ${full_scale_cost:.2f}")
    print(f"  All constraints satisfied: {verification['all_valid']}")


if __name__ == '__main__':
    main()

