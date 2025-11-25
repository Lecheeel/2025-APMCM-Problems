"""
Adjust QUBO Matrix Precision to 8-bit Integer Range
=====================================================

This script adjusts QUBO matrix precision to meet CIM hardware requirements:
8-bit signed integer range [-128, 127].
"""

import argparse
import numpy as np
import json
import os
import sys
from datetime import datetime
from typing import Tuple, Optional

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

try:
    import kaiwu as kw
    KAIWU_AVAILABLE = True
except ImportError:
    KAIWU_AVAILABLE = False
    print("Warning: Kaiwu SDK not available. Using manual precision adjustment.")


def adjust_qubo_precision_manual(qubo_matrix: np.ndarray, 
                                 bit_width: int = 8,
                                 preserve_structure: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Manually adjust QUBO matrix precision to 8-bit integer range.
    
    Args:
        qubo_matrix: Original QUBO matrix
        bit_width: Bit width (default: 8)
        preserve_structure: Whether to preserve matrix structure (default: True)
    
    Returns:
        Tuple of (adjusted_matrix, adjustment_info)
    """
    # Calculate range
    min_val = np.min(qubo_matrix)
    max_val = np.max(qubo_matrix)
    abs_max = max(abs(min_val), abs(max_val))
    
    # 8-bit signed integer range: [-128, 127]
    int_min = -(2**(bit_width-1))
    int_max = 2**(bit_width-1) - 1
    
    # Calculate scaling factor
    if abs_max == 0:
        scale_factor = 1.0
    else:
        # Scale to fit in integer range
        scale_factor = int_max / abs_max
    
    # Scale matrix
    scaled_matrix = qubo_matrix * scale_factor
    
    # Round to integers
    adjusted_matrix = np.round(scaled_matrix).astype(np.int32)
    
    # Clip to valid range
    adjusted_matrix = np.clip(adjusted_matrix, int_min, int_max)
    
    # Make symmetric (upper triangular)
    adjusted_matrix = np.triu(adjusted_matrix) + np.triu(adjusted_matrix, k=1).T
    
    adjustment_info = {
        'original_min': float(min_val),
        'original_max': float(max_val),
        'original_abs_max': float(abs_max),
        'scale_factor': float(scale_factor),
        'adjusted_min': int(np.min(adjusted_matrix)),
        'adjusted_max': int(np.max(adjusted_matrix)),
        'precision_loss': float(abs_max / scale_factor) if scale_factor > 0 else 0.0
    }
    
    return adjusted_matrix, adjustment_info


def adjust_qubo_precision_kaiwu(qubo_matrix: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Adjust QUBO matrix precision using Kaiwu SDK.
    
    Args:
        qubo_matrix: Original QUBO matrix
    
    Returns:
        Tuple of (adjusted_matrix, adjustment_info)
    """
    if not KAIWU_AVAILABLE:
        raise ImportError("Kaiwu SDK is not available")
    
    # Check original precision
    original_min = np.min(qubo_matrix)
    original_max = np.max(qubo_matrix)
    original_abs_max = max(abs(original_min), abs(original_max))
    
    try:
        # Use Kaiwu SDK to adjust precision
        adjusted_matrix = kw.qubo.adjust_qubo_matrix_precision(qubo_matrix, bit_width=8)
        
        # Verify adjustment
        kw.qubo.check_qubo_matrix_bit_width(adjusted_matrix, bit_width=8)
        
        adjusted_min = np.min(adjusted_matrix)
        adjusted_max = np.max(adjusted_matrix)
        
        adjustment_info = {
            'method': 'kaiwu_sdk',
            'original_min': float(original_min),
            'original_max': float(original_max),
            'original_abs_max': float(original_abs_max),
            'adjusted_min': float(adjusted_min),
            'adjusted_max': float(adjusted_max),
            'passes_check': True
        }
        
        return adjusted_matrix, adjustment_info
        
    except ValueError as e:
        # If adjustment fails, fall back to manual method
        print(f"Warning: Kaiwu SDK adjustment failed: {e}")
        print("Falling back to manual adjustment...")
        return adjust_qubo_precision_manual(qubo_matrix)


def format_matrix_for_template(qubo_matrix: np.ndarray) -> str:
    """
    Format QUBO matrix in template CSV format.
    
    Args:
        qubo_matrix: QUBO matrix (should be integer)
    
    Returns:
        Formatted CSV string
    """
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


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Adjust QUBO matrix precision to 8-bit integer range',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Input QUBO matrix file (CSV format)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file path (default: auto-generated)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['kaiwu', 'manual', 'auto'],
        default='auto',
        help='Adjustment method (default: auto)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("QUBO Matrix Precision Adjustment")
    print("=" * 70)
    
    # Load matrix
    print(f"\n[Step 1] Loading QUBO matrix from: {args.input_file}")
    try:
        qubo_matrix = np.loadtxt(args.input_file, delimiter=',')
        print(f"  Matrix shape: {qubo_matrix.shape}")
        print(f"  Matrix dtype: {qubo_matrix.dtype}")
    except Exception as e:
        print(f"  ✗ Error loading matrix: {e}")
        sys.exit(1)
    
    # Check original precision
    print("\n[Step 2] Checking original precision...")
    min_val = np.min(qubo_matrix)
    max_val = np.max(qubo_matrix)
    abs_max = max(abs(min_val), abs(max_val))
    
    print(f"  Original range: [{min_val:.2e}, {max_val:.2e}]")
    print(f"  Absolute max: {abs_max:.2e}")
    
    # Check if adjustment is needed
    int_min = -128
    int_max = 127
    
    if abs_max <= int_max:
        print(f"  ✓ Matrix already within 8-bit range!")
        adjusted_matrix = np.round(qubo_matrix).astype(np.int32)
        adjusted_matrix = np.clip(adjusted_matrix, int_min, int_max)
        adjustment_info = {
            'method': 'none_needed',
            'original_min': float(min_val),
            'original_max': float(max_val),
            'adjusted_min': int(np.min(adjusted_matrix)),
            'adjusted_max': int(np.max(adjusted_matrix))
        }
    else:
        print(f"  ⚠ Matrix exceeds 8-bit range, adjustment needed")
        
        # Adjust precision
        print("\n[Step 3] Adjusting precision...")
        if args.method == 'kaiwu' and KAIWU_AVAILABLE:
            print("  Using Kaiwu SDK method...")
            adjusted_matrix, adjustment_info = adjust_qubo_precision_kaiwu(qubo_matrix)
        elif args.method == 'manual' or (args.method == 'auto' and not KAIWU_AVAILABLE):
            print("  Using manual method...")
            adjusted_matrix, adjustment_info = adjust_qubo_precision_manual(qubo_matrix)
        else:
            print("  Using Kaiwu SDK method...")
            adjusted_matrix, adjustment_info = adjust_qubo_precision_kaiwu(qubo_matrix)
        
        print(f"  Scale factor: {adjustment_info.get('scale_factor', 'N/A')}")
        print(f"  Adjusted range: [{adjustment_info.get('adjusted_min', 'N/A')}, {adjustment_info.get('adjusted_max', 'N/A')}]")
    
    # Verify adjusted matrix
    print("\n[Step 4] Verifying adjusted matrix...")
    adj_min = np.min(adjusted_matrix)
    adj_max = np.max(adjusted_matrix)
    
    if adj_min >= int_min and adj_max <= int_max:
        print(f"  ✓ Matrix is within 8-bit range: [{adj_min}, {adj_max}]")
    else:
        print(f"  ✗ Warning: Matrix still outside range: [{adj_min}, {adj_max}]")
    
    # Calculate statistics
    non_zero_count = np.count_nonzero(adjusted_matrix)
    sparsity = 1.0 - non_zero_count / adjusted_matrix.size
    
    print(f"  Non-zero elements: {non_zero_count} ({100*(1-sparsity):.2f}%)")
    print(f"  Sparsity: {sparsity:.2%}")
    
    # Save results
    print("\n[Step 5] Saving adjusted matrix...")
    
    if args.output_file is None:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_dir = os.path.dirname(args.input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = os.path.join(output_dir, f"{base_name}_8bit_{timestamp}.csv")
    
    # Save in template format
    csv_content = format_matrix_for_template(adjusted_matrix)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(csv_content)
    print(f"  ✓ Adjusted matrix saved to: {args.output_file}")
    
    # Save as standard CSV (integer format)
    standard_csv = args.output_file.replace('.csv', '_standard.csv')
    np.savetxt(standard_csv, adjusted_matrix, delimiter=',', fmt='%d')
    print(f"  ✓ Standard format saved to: {standard_csv}")
    
    # Save metadata
    json_file = args.output_file.replace('.csv', '_metadata.json')
    metadata = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'input_file': args.input_file,
        'output_file': args.output_file,
        'adjustment_info': adjustment_info,
        'matrix_info': {
            'shape': list(adjusted_matrix.shape),
            'dtype': str(adjusted_matrix.dtype),
            'min': int(adj_min),
            'max': int(adj_max),
            'non_zero_count': int(non_zero_count),
            'sparsity': float(sparsity)
        }
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Metadata saved to: {json_file}")
    
    print("\n" + "=" * 70)
    print("Precision Adjustment Completed!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Original range: [{min_val:.2e}, {max_val:.2e}]")
    print(f"  Adjusted range: [{adj_min}, {adj_max}]")
    print(f"  Method: {adjustment_info.get('method', 'manual')}")
    if 'scale_factor' in adjustment_info:
        print(f"  Scale factor: {adjustment_info['scale_factor']:.6e}")
    print(f"  Output files:")
    print(f"    - Template CSV: {args.output_file}")
    print(f"    - Standard CSV: {standard_csv}")
    print(f"    - Metadata JSON: {json_file}")


if __name__ == '__main__':
    main()

