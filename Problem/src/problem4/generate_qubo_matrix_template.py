"""
Generate QUBO Matrix Data for Problem 4 in Template Format
===========================================================

This script generates QUBO matrix data matching the template format (100x100 matrix).
"""

import argparse
import numpy as np
import json
import os
import sys
from datetime import datetime
from typing import Dict, Tuple

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import modules
from problem4.data_loader import ReducedDataLoader
from problem4.reduction_strategy import ReductionStrategy
from problem4.discretization import GenerationDiscretizer
from problem4.generate_qubo_matrix_standalone import compute_qubo_matrix_directly


def format_matrix_for_template(qubo_matrix: np.ndarray, target_size: int = 100) -> np.ndarray:
    """
    Format QUBO matrix to match template format.
    
    Args:
        qubo_matrix: Original QUBO matrix
        target_size: Target matrix size (default: 100)
    
    Returns:
        Formatted matrix matching template format
    """
    n = qubo_matrix.shape[0]
    
    if n == target_size:
        # Already correct size
        formatted = qubo_matrix.copy()
    elif n < target_size:
        # Pad with zeros
        formatted = np.zeros((target_size, target_size))
        formatted[:n, :n] = qubo_matrix
    else:
        # Truncate (shouldn't happen for our reduced problem)
        formatted = qubo_matrix[:target_size, :target_size]
    
    # Ensure symmetric (upper triangular)
    formatted = np.triu(formatted) + np.triu(formatted, k=1).T
    
    return formatted


def save_matrix_template_format(qubo_matrix: np.ndarray, output_file: str):
    """
    Save QUBO matrix in template format (CSV with -0.0 for zeros).
    
    Args:
        qubo_matrix: QUBO matrix to save
        output_file: Output file path
    """
    n = qubo_matrix.shape[0]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(n):
            row_values = []
            for j in range(n):
                val = qubo_matrix[i, j]
                if val == 0.0:
                    # Format zero as -0.0 to match template
                    row_values.append('-0.0')
                else:
                    # Format as float with one decimal place
                    row_values.append(f'{val:.1f}')
            
            # Write row with comma separation
            f.write(','.join(row_values))
            if i < n - 1:  # Don't add newline after last row
                f.write('\n')
    
    print(f"  ✓ QUBO matrix (template format) saved to: {output_file}")


def generate_qubo_matrix_template(max_binary_vars: int = 100,
                                   num_bits: int = 1,
                                   time_aggregation: str = None,
                                   results_dir: str = None,
                                   output_file: str = None):
    """
    Generate QUBO matrix data in template format.
    """
    print("=" * 70)
    print("Problem 4: Generate QUBO Matrix Data (Template Format)")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    data_loader = ReducedDataLoader(results_dir=results_dir)
    summary = data_loader.get_summary()
    print(f"  Units: {summary['num_units']}")
    print(f"  Periods: {summary['num_periods']}")
    
    # Step 2: Apply reduction strategy
    print("\n[Step 2] Applying reduction strategy...")
    reduction_strategy = ReductionStrategy(data_loader)
    
    args_num_bits = num_bits
    max_periods_with_all_units = max_binary_vars // (summary['num_units'] * args_num_bits)
    
    if summary['num_units'] * summary['num_periods'] * args_num_bits > max_binary_vars:
        if time_aggregation is None:
            time_aggregation = 'uniform'
    else:
        if time_aggregation is None:
            time_aggregation = None
    
    reduction_info = reduction_strategy.apply_reduction(
        max_binary_vars=max_binary_vars,
        num_bits=args_num_bits,
        time_aggregation=time_aggregation,
        unit_selection=None,
        force_no_reduction=False
    )
    
    print(f"  Selected units: {reduction_info['selected_units']}")
    print(f"  Selected periods: {reduction_info['selected_periods']}")
    print(f"  Total binary variables: {reduction_info['total_binary_vars']}")
    
    # Step 3: Get reduced data
    print("\n[Step 3] Preparing reduced data...")
    reduced_data = reduction_strategy.get_reduced_data()
    
    # Step 4: Initialize discretizer
    print("\n[Step 4] Initializing discretizer...")
    discretizer = GenerationDiscretizer(
        P_min=reduced_data['P_min'],
        P_max=reduced_data['P_max'],
        num_bits=reduction_info['num_bits']
    )
    
    # Step 5: Compute QUBO matrix directly
    print("\n[Step 5] Computing QUBO matrix directly...")
    try:
        qubo_matrix, qubo_offset = compute_qubo_matrix_directly(reduced_data, discretizer)
        print("  ✓ QUBO matrix computed successfully")
        print(f"  Original matrix shape: {qubo_matrix.shape}")
    except Exception as e:
        print(f"  ✗ Error computing QUBO matrix: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 6: Format matrix to match template (100x100)
    print("\n[Step 6] Formatting matrix to template format...")
    formatted_matrix = format_matrix_for_template(qubo_matrix, target_size=max_binary_vars)
    print(f"  Formatted matrix shape: {formatted_matrix.shape}")
    
    # Calculate matrix statistics
    matrix_stats = {
        'min': float(np.min(formatted_matrix)),
        'max': float(np.max(formatted_matrix)),
        'mean': float(np.mean(formatted_matrix)),
        'std': float(np.std(formatted_matrix)),
        'non_zero_count': int(np.count_nonzero(formatted_matrix)),
        'sparsity': float(1.0 - np.count_nonzero(formatted_matrix) / formatted_matrix.size)
    }
    
    print(f"  Matrix statistics:")
    print(f"    Min: {matrix_stats['min']:.6e}")
    print(f"    Max: {matrix_stats['max']:.6e}")
    print(f"    Mean: {matrix_stats['mean']:.6e}")
    print(f"    Non-zero elements: {matrix_stats['non_zero_count']} ({100*(1-matrix_stats['sparsity']):.2f}%)")
    
    # Step 7: Save files
    print("\n[Step 7] Saving QUBO matrix data...")
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        output_dir = os.path.join(project_root, 'results', 'problem4')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"qubo_matrix_template_{timestamp}.csv")
    
    # Save in template format (CSV with -0.0 for zeros)
    save_matrix_template_format(formatted_matrix, output_file)
    
    # Also save as standard CSV (for comparison)
    standard_csv = output_file.replace('_template_', '_standard_')
    np.savetxt(standard_csv, formatted_matrix, delimiter=',', fmt='%.6e')
    print(f"  ✓ QUBO matrix (standard CSV) saved to: {standard_csv}")
    
    # Save metadata as JSON
    json_file = output_file.replace('.csv', '_metadata.json')
    metadata = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'problem': 'Problem 4: Problem Scale Reduction under Quantum Hardware Constraints',
        'format': 'template_format',
        'reduction_info': {
            'max_binary_vars': max_binary_vars,
            'num_bits': reduction_info['num_bits'],
            'selected_units': reduction_info['selected_units'],
            'selected_periods': reduction_info['selected_periods'],
            'total_binary_vars': reduction_info['total_binary_vars'],
            'reduction_ratio': reduction_info['reduction_ratio'],
            'time_aggregation': time_aggregation
        },
        'matrix_info': {
            'shape': list(formatted_matrix.shape),
            'dtype': str(formatted_matrix.dtype),
            'offset': float(qubo_offset),
            'statistics': matrix_stats
        },
        'reduced_data': {
            'units': reduced_data['units'],
            'P_min': reduced_data['P_min'].tolist(),
            'P_max': reduced_data['P_max'].tolist(),
            'load_demand': reduced_data['load_demand'].tolist(),
            'spinning_reserve_req': reduced_data['spinning_reserve_req'].tolist(),
        }
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Metadata saved to: {json_file}")
    
    print("\n" + "=" * 70)
    print("QUBO Matrix Data Generation Completed!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Binary variables: {reduction_info['total_binary_vars']}")
    print(f"  Matrix size: {formatted_matrix.shape[0]} × {formatted_matrix.shape[1]}")
    print(f"  Offset: {qubo_offset:.6e}")
    print(f"  Output files:")
    print(f"    - Template CSV: {output_file}")
    print(f"    - Standard CSV: {standard_csv}")
    print(f"    - Metadata JSON: {json_file}")
    
    return {
        'matrix': formatted_matrix,
        'offset': qubo_offset,
        'metadata': metadata
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate QUBO matrix data for Problem 4 in template format',
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
        choices=['uniform', 'peak_valley'],
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
    result = generate_qubo_matrix_template(
        max_binary_vars=args.max_binary_vars,
        num_bits=args.num_bits,
        time_aggregation=args.time_aggregation,
        results_dir=args.results_dir,
        output_file=args.output_file
    )
    
    if result is None:
        print("\n✗ Failed to generate QUBO matrix data")
        sys.exit(1)
    else:
        print("\n✓ QUBO matrix data generated successfully")


if __name__ == '__main__':
    main()

