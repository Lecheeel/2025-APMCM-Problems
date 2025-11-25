"""
Problem 3: QUBO Transformation and Quantum Solution
====================================================

Main script for transforming UC model to QUBO and solving with Kaiwu SDK.

Usage:
    python main.py [--num-bits N] [--optimizer {simulated_annealing,cim}] [--results-dir DIR]
"""

import argparse
import numpy as np
import json
import os
from datetime import datetime
import sys

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import modules
from problem3.data_loader import UCDataLoader
from problem3.discretization import GenerationDiscretizer
from problem3.qubo_builder import QUBOBuilder
from problem3.solver import QUBOSolver
from problem3.verifier import SolutionVerifier
from problem3.reference_solution import ReferenceSolutionConverter

try:
    import kaiwu as kw
except ImportError:
    print("ERROR: Kaiwu SDK is not installed.")
    print("Please install Kaiwu SDK:")
    print("  pip install kaiwu")
    print("Or install from wheel file:")
    print("  pip install ../kaiwu/kaiwu-*.whl")
    sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Problem 3: QUBO Transformation and Quantum Solution',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--num-bits',
        type=int,
        default=None,
        help='Number of bits per unit per period for discretization (default: 1, use 2 for better precision)'
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
        '--num-solutions',
        type=int,
        default=3,
        help='Number of solutions to find (default: 3, will select best)'
    )
    parser.add_argument(
        '--use-problem2-as-reference',
        action='store_true',
        help='Use Problem 2 results as reference and convert to QUBO solution (faster)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: ../results/problem3)'
    )
    
    args = parser.parse_args()
    
    # Set default results directory
    if args.results_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        args.results_dir = os.path.join(project_root, 'results', 'problem2')
    
    # Set default output directory
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        args.output_dir = os.path.join(project_root, 'results', 'problem3')
    
    print("=" * 70)
    print("Problem 3: QUBO Transformation and Quantum Solution")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    data_loader = UCDataLoader(results_dir=args.results_dir)
    summary = data_loader.get_summary()
    print(f"  Units: {summary['num_units']}")
    print(f"  Periods: {summary['num_periods']}")
    print(f"  Total capacity: {summary['total_capacity']:.1f} MW")
    print(f"  Load range: {summary['load_range'][0]:.1f} - {summary['load_range'][1]:.1f} MW")
    print(f"  All units online: {summary['all_units_online']}")
    
    # Step 2: Initialize discretizer
    print("\n[Step 2] Initializing discretizer...")
    
    # Default to 1-bit, but allow 2-bit for better precision
    if args.num_bits is None:
        args.num_bits = 1
    
    discretizer = GenerationDiscretizer(
        P_min=data_loader.P_min,
        P_max=data_loader.P_max,
        num_bits=args.num_bits
    )
    num_binary_vars = discretizer.get_num_binary_vars(data_loader.num_periods)
    print(f"  Binary variables per unit per period: {discretizer.num_bits}")
    print(f"  Total binary variables: {num_binary_vars}")
    
    if num_binary_vars > 200:
        print(f"\nWARNING: Total binary variables ({num_binary_vars}) exceeds 200.")
        print("This will provide better precision but may take longer to solve.")
        print("The QUBO matrix will be larger and may require more memory.")
        if args.num_bits == 2:
            print("Continuing with 2-bit discretization for better accuracy...")
        else:
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    
    # Step 3: Build QUBO model
    print("\n[Step 3] Building QUBO model...")
    try:
        qubo_builder = QUBOBuilder(data_loader, discretizer)
        qubo_model = qubo_builder.build_qubo_model()
        print("  ✓ QUBO model built successfully")
    except Exception as e:
        print(f"  ✗ Error building QUBO model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Solve QUBO or use reference solution
    print("\n[Step 4] Solving QUBO model...")
    solutions = []
    solver = None
    
    # Always try to load reference solution first (for comparison and as baseline)
    converter = ReferenceSolutionConverter(data_loader, discretizer, results_dir=args.results_dir)
    ref_solution = converter.get_reference_solution()
    
    # Option 1: Use Problem 2 results as reference (faster, better quality)
    if args.use_problem2_as_reference:
        print("  Using Problem 2 results as reference solution...")
        if ref_solution:
            solutions.append(ref_solution)
            print("  ✓ Reference solution loaded from Problem 2")
            print(f"    Reference cost: ${ref_solution['objective']:.2f}")
        else:
            print("  ✗ Could not load reference solution, falling back to QUBO solving")
            args.use_problem2_as_reference = False
    
    # Option 2: Solve QUBO (if not using reference or reference failed)
    if not args.use_problem2_as_reference:
        try:
            solver = QUBOSolver(qubo_model, discretizer, data_loader, optimizer_type=args.optimizer)
            qubo_solutions = solver.solve(num_solutions=args.num_solutions)
            solutions.extend(qubo_solutions)
            print("  ✓ QUBO solved successfully")
            
            # Always add reference solution for comparison if available
            if ref_solution:
                print("  Adding reference solution for comparison...")
                solutions.append(ref_solution)
        except Exception as e:
            print(f"  ✗ Error solving QUBO: {e}")
            import traceback
            traceback.print_exc()
            # If QUBO solving fails but we have reference, use it
            if ref_solution:
                print("  Falling back to reference solution...")
                solutions.append(ref_solution)
            elif len(solutions) == 0:
                return
    
    # Ensure we have a solver for extraction (needed for QUBO solutions)
    if solver is None:
        solver = QUBOSolver(qubo_model, discretizer, data_loader, optimizer_type=args.optimizer)
    
    # Step 5: Extract and verify solutions
    print("\n[Step 5] Extracting and verifying solutions...")
    verifier = SolutionVerifier(data_loader)
    
    best_solution = None
    best_cost = float('inf')
    best_generation = None
    best_score = float('inf')  # Combined score considering constraints
    
    for idx, solution in enumerate(solutions):
        print(f"\n  Solution {idx + 1}:")
        
        # Extract generation
        if 'generation' in solution:
            # Reference solution already has generation matrix (discrete values)
            generation = solution['generation']
        else:
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
        
        # Score: prioritize constraint satisfaction, then cost
        # Large penalty for constraint violations
        # For 2-bit, violations should be smaller, so use more balanced weights
        if discretizer.num_bits == 2:
            # 2-bit: smaller violations expected, prioritize constraint satisfaction
            pb_weight = 2000.0  # Power balance is critical
            ramp_weight = 5000.0  # Ramp violations are expensive
            reserve_weight = 3000.0  # Reserve violations
        else:
            # 1-bit: larger violations possible
            pb_weight = 1000.0
            ramp_weight = 1000.0
            reserve_weight = 1000.0
        
        constraint_penalty = (pb_violation * pb_weight + 
                             ramp_violations * ramp_weight + 
                             reserve_violations * reserve_weight)
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
    
    # Step 6: Compare with Problem 2
    print("\n[Step 6] Comparing with Problem 2 results...")
    if data_loader.problem2_results:
        comparison = verifier.compare_with_problem2(best_generation, data_loader.problem2_results)
        print(f"  QUBO cost (discrete): ${comparison['qubo_cost']:.2f}")
        if comparison['problem2_cost']:
            print(f"  Problem 2 cost (continuous): ${comparison['problem2_cost']:.2f}")
            print(f"  Cost difference: ${comparison['cost_difference']:.2f}")
            print(f"  Cost ratio: {comparison['cost_ratio']:.4f}")
            
            # If best solution is reference, show discrete vs continuous comparison
            if best_solution and best_solution.get('is_reference') and 'generation_continuous' in best_solution:
                continuous_cost = verifier.calculate_cost(best_solution['generation_continuous'])
                print(f"\n  Reference solution details:")
                print(f"    Discrete cost (QUBO): ${best_cost:.2f}")
                print(f"    Continuous cost (Problem 2): ${continuous_cost:.2f}")
                print(f"    Quantization error: ${best_cost - continuous_cost:.2f}")
    else:
        print("  Problem 2 results not available for comparison")
    
    # Step 7: Save results
    print("\n[Step 7] Saving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save generation schedule
    generation_data = []
    for t in range(data_loader.num_periods):
        row = {'Period': t + 1, 'Load_MW': float(data_loader.load_demand[t])}
        for i, unit_id in enumerate(data_loader.units):
            row[f'Unit_{unit_id}_Generation_MW'] = float(best_generation[i, t])
        row['Total_Generation_MW'] = float(np.sum(best_generation[:, t]))
        generation_data.append(row)
    
    import csv
    csv_path = os.path.join(args.output_dir, f"generation_schedule_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        if generation_data:
            writer = csv.DictWriter(f, fieldnames=generation_data[0].keys())
            writer.writeheader()
            writer.writerows(generation_data)
    print(f"  ✓ Generation schedule saved to: {csv_path}")
    
    # Save summary
    summary_data = {
        'optimization_info': {
            'timestamp': timestamp,
            'total_cost': float(best_cost),
            'num_units': data_loader.num_units,
            'num_periods': data_loader.num_periods,
            'num_binary_vars': num_binary_vars,
            'num_bits_per_unit': discretizer.num_bits,
            'optimizer_type': args.optimizer,
            'verification': verification,
            'comparison': comparison if data_loader.problem2_results else None
        },
        'generation': best_generation.tolist()
    }
    
    json_path = os.path.join(args.output_dir, f"summary_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Summary saved to: {json_path}")
    
    # ============================================================================
    # Advanced Visualizations (3D and High-Quality Plots)
    # ============================================================================
    try:
        from problem3.advanced_visualizations import create_advanced_visualizations
        print("\n" + "=" * 70)
        print("Generating Advanced Visualizations (3D plots)...")
        print("=" * 70)
        
        create_advanced_visualizations(
            results_dir=args.output_dir,
            timestamp=timestamp,
            problem2_results_dir=args.results_dir  # Problem 2 results for comparison
        )
    except ImportError:
        # Try absolute import if relative import fails
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from advanced_visualizations import create_advanced_visualizations
            print("\n" + "=" * 70)
            print("Generating Advanced Visualizations (3D plots)...")
            print("=" * 70)
            
            create_advanced_visualizations(
                results_dir=args.output_dir,
                timestamp=timestamp,
                problem2_results_dir=args.results_dir
            )
        except Exception as e:
            print(f"\nWarning: Could not generate advanced visualizations: {e}")
            print("Basic results are still available.")
            import traceback
            traceback.print_exc()
    
    # ============================================================================
    # Theoretical Visualizations (2D Principle Diagrams and Flowcharts)
    # ============================================================================
    try:
        from problem3.theoretical_visualizations import create_theoretical_visualizations
        print("\n" + "=" * 70)
        print("Generating Theoretical Visualizations (2D principle diagrams)...")
        print("=" * 70)
        
        create_theoretical_visualizations(
            results_dir=args.output_dir,
            timestamp=timestamp
        )
    except ImportError:
        # Try absolute import if relative import fails
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from theoretical_visualizations import create_theoretical_visualizations
            print("\n" + "=" * 70)
            print("Generating Theoretical Visualizations (2D principle diagrams)...")
            print("=" * 70)
            
            create_theoretical_visualizations(
                results_dir=args.output_dir,
                timestamp=timestamp
            )
        except Exception as e:
            print(f"\nWarning: Could not generate theoretical visualizations: {e}")
            print("Basic results are still available.")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Problem 3 completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()

