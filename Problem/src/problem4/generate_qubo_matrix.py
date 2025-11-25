"""
Generate Standard QUBO Matrix Data for Problem 4
==================================================

This script generates simplified QUBO matrix data for Problem 4,
including the QUBO matrix, offset, variable names, and metadata.
"""

import argparse
import numpy as np
import json
import os
import sys
from datetime import datetime
from typing import Dict

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Try to import kaiwu first (before importing modules that depend on it)
try:
    import kaiwu as kw
except ImportError:
    print("ERROR: Kaiwu SDK is not installed.")
    print("Please install Kaiwu SDK:")
    print("  pip install kaiwu")
    print("Or install from wheel file:")
    print("  pip install ../kaiwu/kaiwu-*.whl")
    sys.exit(1)

# Import modules (after kaiwu is available)
from problem4.data_loader import ReducedDataLoader
from problem4.reduction_strategy import ReductionStrategy
from problem4.discretization import GenerationDiscretizer
from problem4.qubo_builder import ReducedQUBOBuilder


def generate_qubo_matrix_data(max_binary_vars: int = 100,
                              num_bits: int = 1,
                              time_aggregation: str = None,
                              results_dir: str = None,
                              output_file: str = None):
    """
    Generate standard QUBO matrix data for Problem 4.
    
    Args:
        max_binary_vars: Maximum number of binary variables allowed
        num_bits: Number of bits per unit per period
        time_aggregation: Time aggregation method ('uniform', 'peak_valley', None)
        results_dir: Directory containing Problem 2 results
        output_file: Output file path (default: auto-generated)
    
    Returns:
        Dictionary containing QUBO matrix data
    """
    print("=" * 70)
    print("Problem 4: Generate Standard QUBO Matrix Data")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    data_loader = ReducedDataLoader(results_dir=results_dir)
    summary = data_loader.get_summary()
    print(f"  Units: {summary['num_units']}")
    print(f"  Periods: {summary['num_periods']}")
    print(f"  Total capacity: {summary['total_capacity']:.1f} MW")
    
    # Step 2: Apply reduction strategy
    print("\n[Step 2] Applying reduction strategy...")
    reduction_strategy = ReductionStrategy(data_loader)
    
    # Ensure we stay within binary variable limit
    args_num_bits = num_bits
    max_periods_with_all_units = max_binary_vars // (summary['num_units'] * args_num_bits)
    
    if summary['num_units'] * summary['num_periods'] * args_num_bits > max_binary_vars:
        print(f"  ⚠ Exceeds limit: Need to reduce to ≤ {max_binary_vars} variables")
        print(f"  Strategy: Keep all {summary['num_units']} units, reduce periods to {max_periods_with_all_units}")
        if time_aggregation is None:
            time_aggregation = 'uniform'
    else:
        print(f"  ✓ Within limit: No reduction needed")
        if time_aggregation is None:
            time_aggregation = None
    
    reduction_info = reduction_strategy.apply_reduction(
        max_binary_vars=max_binary_vars,
        num_bits=args_num_bits,
        time_aggregation=time_aggregation,
        unit_selection=None,
        force_no_reduction=False
    )
    
    # Verify we're within limits
    if reduction_info['total_binary_vars'] > max_binary_vars:
        print(f"  ⚠ Still exceeds limit: {reduction_info['total_binary_vars']} > {max_binary_vars}")
        print(f"  Applying additional reduction...")
        max_periods = max(1, max_binary_vars // (len(reduction_info['selected_units']) * args_num_bits))
        reduction_strategy.selected_units = reduction_info['selected_units']
        reduction_strategy.selected_periods = reduction_strategy._select_key_periods(max_periods)
        reduction_strategy.period_mapping = reduction_strategy._create_period_mapping()
        reduction_info['selected_periods'] = reduction_strategy.selected_periods
        reduction_info['period_mapping'] = reduction_strategy.period_mapping
        reduction_info['total_binary_vars'] = len(reduction_info['selected_units']) * len(reduction_info['selected_periods']) * args_num_bits
    
    print(f"  Selected units: {reduction_info['selected_units']}")
    print(f"  Selected periods: {reduction_info['selected_periods']}")
    print(f"  Number of bits: {reduction_info['num_bits']}")
    print(f"  Total binary variables: {reduction_info['total_binary_vars']}")
    
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
        return None
    
    # Step 6: Extract QUBO matrix data
    print("\n[Step 6] Extracting QUBO matrix data...")
    
    # Get QUBO matrix
    qubo_matrix = qubo_model.get_matrix()
    print(f"  QUBO matrix shape: {qubo_matrix.shape}")
    print(f"  QUBO matrix size: {qubo_matrix.size}")
    print(f"  QUBO matrix dtype: {qubo_matrix.dtype}")
    
    # Get offset (constant term)
    qubo_offset = qubo_model.get_offset()
    print(f"  QUBO offset: {qubo_offset}")
    
    # Get variables
    qubo_variables = qubo_model.get_variables()
    variable_names = [str(var) for var in qubo_variables]
    print(f"  Number of variables: {len(variable_names)}")
    
    # Get variable mapping (index -> name)
    variable_mapping = {i: name for i, name in enumerate(variable_names)}
    
    # Calculate matrix statistics
    matrix_stats = {
        'min': float(np.min(qubo_matrix)),
        'max': float(np.max(qubo_matrix)),
        'mean': float(np.mean(qubo_matrix)),
        'std': float(np.std(qubo_matrix)),
        'non_zero_count': int(np.count_nonzero(qubo_matrix)),
        'sparsity': float(1.0 - np.count_nonzero(qubo_matrix) / qubo_matrix.size)
    }
    
    print(f"  Matrix statistics:")
    print(f"    Min: {matrix_stats['min']:.6e}")
    print(f"    Max: {matrix_stats['max']:.6e}")
    print(f"    Mean: {matrix_stats['mean']:.6e}")
    print(f"    Std: {matrix_stats['std']:.6e}")
    print(f"    Non-zero elements: {matrix_stats['non_zero_count']} ({100*(1-matrix_stats['sparsity']):.2f}%)")
    print(f"    Sparsity: {matrix_stats['sparsity']:.2%}")
    
    # Check bit width compatibility
    bit_width_check = {}
    try:
        kw.qubo.check_qubo_matrix_bit_width(qubo_matrix, bit_width=8)
        bit_width_check['passes_8bit'] = True
        bit_width_check['message'] = "Matrix passes 8-bit CIM compatibility check"
        print(f"  ✓ {bit_width_check['message']}")
    except ValueError as e:
        bit_width_check['passes_8bit'] = False
        bit_width_check['message'] = str(e)
        print(f"  ⚠ {bit_width_check['message']}")
    
    # Step 7: Prepare output data
    print("\n[Step 7] Preparing output data...")
    
    qubo_data = {
        'metadata': {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'problem': 'Problem 4: Problem Scale Reduction under Quantum Hardware Constraints',
            'reduction_info': {
                'max_binary_vars': max_binary_vars,
                'num_bits': reduction_info['num_bits'],
                'selected_units': reduction_info['selected_units'],
                'selected_periods': reduction_info['selected_periods'],
                'total_binary_vars': reduction_info['total_binary_vars'],
                'reduction_ratio': reduction_info['reduction_ratio'],
                'time_aggregation': time_aggregation
            },
            'problem_size': {
                'num_units': reduced_data['num_units'],
                'num_periods': reduced_data['num_periods'],
                'num_binary_vars': reduction_info['total_binary_vars']
            },
            'matrix_info': {
                'shape': list(qubo_matrix.shape),
                'dtype': str(qubo_matrix.dtype),
                'offset': float(qubo_offset),
                'statistics': matrix_stats,
                'bit_width_check': bit_width_check
            }
        },
        'qubo_matrix': qubo_matrix.tolist(),
        'qubo_offset': float(qubo_offset),
        'variable_names': variable_names,
        'variable_mapping': variable_mapping,
        'reduced_data': {
            'units': reduced_data['units'],
            'P_min': reduced_data['P_min'].tolist(),
            'P_max': reduced_data['P_max'].tolist(),
            'load_demand': reduced_data['load_demand'].tolist(),
            'spinning_reserve_req': reduced_data['spinning_reserve_req'].tolist(),
            'a_coeff': reduced_data['a_coeff'].tolist(),
            'b_coeff': reduced_data['b_coeff'].tolist(),
            'c_coeff': reduced_data['c_coeff'].tolist(),
            'Ramp_Up': reduced_data['Ramp_Up'].tolist(),
            'Ramp_Down': reduced_data['Ramp_Down'].tolist()
        }
    }
    
    # Step 8: Save to file
    print("\n[Step 8] Saving QUBO matrix data...")
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        output_dir = os.path.join(project_root, 'results', 'problem4')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"qubo_matrix_{timestamp}.json")
    
    # Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qubo_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ QUBO matrix data saved to: {output_file}")
    
    # Also save matrix in CSV format (sparse format: row, col, value)
    csv_file = output_file.replace('.json', '_matrix.csv')
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("row,col,value\n")
        rows, cols = qubo_matrix.shape
        for i in range(rows):
            for j in range(cols):
                if qubo_matrix[i, j] != 0:
                    f.write(f"{i},{j},{qubo_matrix[i, j]}\n")
    print(f"  ✓ QUBO matrix (sparse CSV) saved to: {csv_file}")
    
    # Save full matrix as CSV (dense format)
    dense_csv_file = output_file.replace('.json', '_matrix_dense.csv')
    np.savetxt(dense_csv_file, qubo_matrix, delimiter=',', fmt='%.6e')
    print(f"  ✓ QUBO matrix (dense CSV) saved to: {dense_csv_file}")
    
    print("\n" + "=" * 70)
    print("QUBO Matrix Data Generation Completed!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Binary variables: {reduction_info['total_binary_vars']}")
    print(f"  Matrix size: {qubo_matrix.shape[0]} × {qubo_matrix.shape[1]}")
    print(f"  Offset: {qubo_offset:.6e}")
    print(f"  Output files:")
    print(f"    - JSON: {output_file}")
    print(f"    - Sparse CSV: {csv_file}")
    print(f"    - Dense CSV: {dense_csv_file}")
    
    return qubo_data


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate standard QUBO matrix data for Problem 4',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--max-binary-vars',
        type=int,
        default=100,
        help='Maximum number of binary variables allowed (default: 100)'
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
        '--results-dir',
        type=str,
        default=None,
        help='Directory containing Problem 2 results (default: ../results/problem2)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file path (default: auto-generated in results/problem4)'
    )
    
    args = parser.parse_args()
    
    # Set default directories
    if args.results_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        args.results_dir = os.path.join(project_root, 'results', 'problem2')
    
    # Generate QUBO matrix data
    qubo_data = generate_qubo_matrix_data(
        max_binary_vars=args.max_binary_vars,
        num_bits=args.num_bits,
        time_aggregation=args.time_aggregation,
        results_dir=args.results_dir,
        output_file=args.output_file
    )
    
    if qubo_data is None:
        print("\n✗ Failed to generate QUBO matrix data")
        sys.exit(1)
    else:
        print("\n✓ QUBO matrix data generated successfully")


if __name__ == '__main__':
    main()

