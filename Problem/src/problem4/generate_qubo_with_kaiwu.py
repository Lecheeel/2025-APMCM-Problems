"""
Generate QUBO Matrix Using Kaiwu SDK (Official Method)
=======================================================

This script uses Kaiwu SDK's official methods to generate and adjust QUBO matrix
to meet CIM hardware 8-bit integer requirements.

Reference: https://kaiwu-sdk-docs.qboson.com/en/latest/
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

try:
    import kaiwu as kw
    KAIWU_AVAILABLE = True
except ImportError:
    KAIWU_AVAILABLE = False
    print("ERROR: Kaiwu SDK is not installed.")
    print("Please install Kaiwu SDK:")
    print("  pip install kaiwu")
    print("Or install from wheel file:")
    print("  pip install ../kaiwu/kaiwu-*.whl")
    print("\nDocumentation: https://kaiwu-sdk-docs.qboson.com/en/latest/")
    sys.exit(1)

# Import modules
from problem4.data_loader import ReducedDataLoader
from problem4.reduction_strategy import ReductionStrategy
from problem4.discretization import GenerationDiscretizer
from problem4.qubo_builder import ReducedQUBOBuilder


def format_matrix_template(qubo_matrix: np.ndarray) -> str:
    """Format QUBO matrix in template CSV format."""
    n = qubo_matrix.shape[0]
    lines = []
    
    for i in range(n):
        row_values = []
        for j in range(n):
            val = int(qubo_matrix[i, j])
            if val == 0:
                row_values.append('-0.0')
            else:
                row_values.append(f'{val}.0')
        
        lines.append(','.join(row_values))
    
    return '\n'.join(lines)


def generate_qubo_with_kaiwu(max_binary_vars: int = 100,
                               num_bits: int = 1,
                               time_aggregation: str = None,
                               results_dir: str = None,
                               output_file: str = None,
                               target_max: int = 80):
    """
    Generate QUBO matrix using Kaiwu SDK and ensure 8-bit compatibility.
    
    Reference: https://kaiwu-sdk-docs.qboson.com/en/latest/
    """
    print("=" * 70)
    print("Problem 4: Generate QUBO Matrix (Kaiwu SDK Method)")
    print("=" * 70)
    print(f"Kaiwu SDK Documentation: https://kaiwu-sdk-docs.qboson.com/en/latest/")
    
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
    
    # Step 5: Build QUBO model using Kaiwu SDK
    print("\n[Step 5] Building QUBO model with Kaiwu SDK...")
    try:
        qubo_builder = ReducedQUBOBuilder(reduced_data, discretizer)
        qubo_model = qubo_builder.build_qubo_model()
        print("  ✓ QUBO model built successfully")
    except Exception as e:
        print(f"  ✗ Error building QUBO model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 6: Extract QUBO matrix
    print("\n[Step 6] Extracting QUBO matrix...")
    qubo_matrix = qubo_model.get_matrix()
    qubo_offset = qubo_model.get_offset()
    
    print(f"  Original matrix shape: {qubo_matrix.shape}")
    print(f"  Original range: [{np.min(qubo_matrix):.2e}, {np.max(qubo_matrix):.2e}]")
    print(f"  Original max abs: {np.max(np.abs(qubo_matrix)):.2e}")
    
    # Step 7: Check and adjust precision using Kaiwu SDK
    print("\n[Step 7] Checking and adjusting precision with Kaiwu SDK...")
    
    try:
        # Check if matrix passes 8-bit check
        kw.qubo.check_qubo_matrix_bit_width(qubo_matrix, bit_width=8)
        print("  ✓ Matrix passes 8-bit check (no adjustment needed)")
        adjusted_matrix = qubo_matrix.copy()
    except ValueError as e:
        print(f"  ⚠ Matrix fails 8-bit check: {e}")
        print("  Applying Kaiwu SDK precision adjustment...")
        
        try:
            # Use Kaiwu SDK's official method to adjust precision
            adjusted_matrix = kw.qubo.adjust_qubo_matrix_precision(qubo_matrix, bit_width=8)
            print("  ✓ Precision adjusted using Kaiwu SDK")
            
            # Verify adjustment
            try:
                kw.qubo.check_qubo_matrix_bit_width(adjusted_matrix, bit_width=8)
                print("  ✓ Adjusted matrix passes 8-bit check")
            except ValueError as e2:
                print(f"  ⚠ Adjusted matrix still fails check: {e2}")
                print("  Applying additional manual scaling...")
                # Additional manual scaling if needed
                max_abs = np.max(np.abs(adjusted_matrix))
                if max_abs > target_max:
                    scale_factor = target_max / max_abs
                    adjusted_matrix = adjusted_matrix * scale_factor
                    adjusted_matrix = np.round(adjusted_matrix).astype(np.int32)
                    adjusted_matrix = np.clip(adjusted_matrix, -128, 127)
                    print(f"  Applied manual scaling: {scale_factor:.6f}")
        except Exception as adjust_error:
            print(f"  ✗ Error adjusting precision: {adjust_error}")
            print("  Falling back to manual adjustment...")
            # Fallback to manual adjustment
            max_abs = np.max(np.abs(qubo_matrix))
            scale_factor = target_max / max_abs if max_abs > 0 else 1.0
            adjusted_matrix = qubo_matrix * scale_factor
            adjusted_matrix = np.round(adjusted_matrix).astype(np.int32)
            adjusted_matrix = np.clip(adjusted_matrix, -128, 127)
            print(f"  Manual scale factor: {scale_factor:.6e}")
    
    # Ensure matrix is integer and in range
    adjusted_matrix = np.round(adjusted_matrix).astype(np.int32)
    adjusted_matrix = np.clip(adjusted_matrix, -128, 127)
    
    # Make symmetric
    adjusted_matrix = np.triu(adjusted_matrix) + np.triu(adjusted_matrix, k=1).T
    
    # Final verification
    print("\n[Step 8] Final verification...")
    final_min = np.min(adjusted_matrix)
    final_max = np.max(adjusted_matrix)
    final_max_abs = np.max(np.abs(adjusted_matrix))
    out_of_range = np.sum((adjusted_matrix < -128) | (adjusted_matrix > 127))
    
    print(f"  Final range: [{final_min}, {final_max}]")
    print(f"  Max absolute: {final_max_abs}")
    print(f"  Values out of range: {out_of_range}")
    
    if final_min >= -128 and final_max <= 127 and out_of_range == 0:
        print("  ✓ Matrix is strictly within [-128, 127] range!")
    else:
        print("  ✗ Warning: Some values may be outside range")
    
    # Statistics
    non_zero = np.count_nonzero(adjusted_matrix)
    sparsity = 1.0 - non_zero / adjusted_matrix.size
    
    print(f"  Non-zero elements: {non_zero} ({100*(1-sparsity):.2f}%)")
    print(f"  Sparsity: {sparsity:.2%}")
    
    # Step 9: Save results
    print("\n[Step 9] Saving results...")
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        output_dir = os.path.join(project_root, 'results', 'problem4')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"qubo_matrix_kaiwu_{timestamp}.csv")
    
    # Save template format
    csv_content = format_matrix_template(adjusted_matrix)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(csv_content)
    print(f"  ✓ Template CSV saved to: {output_file}")
    
    # Save standard format
    standard_csv = output_file.replace('.csv', '_standard.csv')
    np.savetxt(standard_csv, adjusted_matrix, delimiter=',', fmt='%d')
    print(f"  ✓ Standard CSV saved to: {standard_csv}")
    
    # Save metadata
    json_file = output_file.replace('.csv', '_metadata.json')
    metadata = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'method': 'kaiwu_sdk',
        'kaiwu_docs': 'https://kaiwu-sdk-docs.qboson.com/en/latest/',
        'reduction_info': reduction_info,
        'matrix_info': {
            'shape': list(adjusted_matrix.shape),
            'dtype': str(adjusted_matrix.dtype),
            'offset': float(qubo_offset),
            'min': int(final_min),
            'max': int(final_max),
            'max_abs': int(final_max_abs),
            'non_zero_count': int(non_zero),
            'sparsity': float(sparsity),
            'original_range': [float(np.min(qubo_matrix)), float(np.max(qubo_matrix))],
            'original_max_abs': float(np.max(np.abs(qubo_matrix)))
        },
        'verification': {
            'in_range': final_min >= -128 and final_max <= 127,
            'out_of_range_count': int(out_of_range),
            'all_integers': True
        }
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Metadata saved to: {json_file}")
    
    print("\n" + "=" * 70)
    print("QUBO Matrix Generation Completed!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Method: Kaiwu SDK (Official)")
    print(f"  Documentation: https://kaiwu-sdk-docs.qboson.com/en/latest/")
    print(f"  Matrix size: {adjusted_matrix.shape[0]} × {adjusted_matrix.shape[1]}")
    print(f"  Value range: [{final_min}, {final_max}]")
    print(f"  Max absolute: {final_max_abs}")
    print(f"  ✓ Strictly within [-128, 127]")
    print(f"\nOutput files:")
    print(f"  - Template CSV: {output_file}")
    print(f"  - Standard CSV: {standard_csv}")
    print(f"  - Metadata JSON: {json_file}")
    
    return adjusted_matrix, qubo_offset, metadata


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate QUBO matrix using Kaiwu SDK official methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Kaiwu SDK Documentation: https://kaiwu-sdk-docs.qboson.com/en/latest/'
    )
    parser.add_argument(
        '--max-binary-vars',
        type=int,
        default=100,
        help='Maximum number of binary variables (default: 100)'
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
        help='Directory containing Problem 2 results'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file path (default: auto-generated)'
    )
    parser.add_argument(
        '--target-max',
        type=int,
        default=80,
        help='Target maximum absolute value for final matrix (default: 80)'
    )
    
    args = parser.parse_args()
    
    if args.results_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        args.results_dir = os.path.join(project_root, 'results', 'problem2')
    
    if not KAIWU_AVAILABLE:
        print("ERROR: Kaiwu SDK is required for this script.")
        print("Please install Kaiwu SDK:")
        print("  pip install kaiwu")
        print("\nDocumentation: https://kaiwu-sdk-docs.qboson.com/en/latest/")
        sys.exit(1)
    
    generate_qubo_with_kaiwu(
        max_binary_vars=args.max_binary_vars,
        num_bits=args.num_bits,
        time_aggregation=args.time_aggregation,
        results_dir=args.results_dir,
        output_file=args.output_file,
        target_max=args.target_max
    )


if __name__ == '__main__':
    main()

