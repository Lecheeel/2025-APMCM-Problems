"""
Force QUBO Matrix to Strict 8-bit Integer Range
================================================

This script aggressively scales QUBO matrix to ensure ALL values are strictly
within [-128, 127] range for CIM hardware compatibility.
"""

import argparse
import numpy as np
import json
import os
import sys
from datetime import datetime

def force_8bit_range(qubo_matrix: np.ndarray, target_max: int = 120) -> np.ndarray:
    """
    Force matrix to strict 8-bit range with aggressive scaling.
    
    Args:
        qubo_matrix: Input QUBO matrix
        target_max: Target maximum absolute value (default: 120, leaving margin)
    
    Returns:
        Scaled matrix with all values in [-128, 127]
    """
    # Find current maximum absolute value
    max_abs = np.max(np.abs(qubo_matrix))
    
    if max_abs == 0:
        return qubo_matrix.astype(np.int32)
    
    # Calculate aggressive scale factor
    scale_factor = target_max / max_abs
    
    # Apply scaling
    scaled = qubo_matrix * scale_factor
    
    # Round to integers
    rounded = np.round(scaled).astype(np.int32)
    
    # Clip to strict range [-128, 127]
    clipped = np.clip(rounded, -128, 127)
    
    # Make symmetric (upper triangular)
    clipped = np.triu(clipped) + np.triu(clipped, k=1).T
    
    return clipped


def format_matrix_template(qubo_matrix: np.ndarray) -> str:
    """Format matrix in template CSV format."""
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
    parser = argparse.ArgumentParser(
        description='Force QUBO matrix to strict 8-bit integer range',
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
        '--target-max',
        type=int,
        default=120,
        help='Target maximum absolute value (default: 120)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Force QUBO Matrix to Strict 8-bit Range")
    print("=" * 70)
    
    # Load matrix
    print(f"\n[Step 1] Loading matrix from: {args.input_file}")
    try:
        qubo_matrix = np.loadtxt(args.input_file, delimiter=',')
        print(f"  Matrix shape: {qubo_matrix.shape}")
        print(f"  Original range: [{np.min(qubo_matrix):.2f}, {np.max(qubo_matrix):.2f}]")
        print(f"  Max absolute value: {np.max(np.abs(qubo_matrix)):.2f}")
    except Exception as e:
        print(f"  ✗ Error loading matrix: {e}")
        sys.exit(1)
    
    # Check if adjustment needed
    min_val = np.min(qubo_matrix)
    max_val = np.max(qubo_matrix)
    max_abs = np.max(np.abs(qubo_matrix))
    
    if max_abs <= args.target_max and min_val >= -128 and max_val <= 127:
        print(f"  ✓ Matrix already within range!")
        adjusted_matrix = np.round(qubo_matrix).astype(np.int32)
        adjusted_matrix = np.clip(adjusted_matrix, -128, 127)
    else:
        print(f"\n[Step 2] Applying aggressive scaling...")
        print(f"  Current max abs: {max_abs:.2f}")
        print(f"  Target max: {args.target_max}")
        
        adjusted_matrix = force_8bit_range(qubo_matrix, target_max=args.target_max)
        
        scale_factor = args.target_max / max_abs if max_abs > 0 else 1.0
        print(f"  Scale factor: {scale_factor:.6e}")
    
    # Verify result
    print(f"\n[Step 3] Verifying result...")
    final_min = np.min(adjusted_matrix)
    final_max = np.max(adjusted_matrix)
    final_max_abs = np.max(np.abs(adjusted_matrix))
    out_of_range = np.sum((adjusted_matrix < -128) | (adjusted_matrix > 127))
    
    print(f"  Final range: [{final_min}, {final_max}]")
    print(f"  Max absolute value: {final_max_abs}")
    print(f"  Values out of range: {out_of_range}")
    
    if final_min >= -128 and final_max <= 127:
        print(f"  ✓ Matrix is strictly within [-128, 127] range!")
    else:
        print(f"  ✗ Warning: Some values still outside range!")
        # Force clip again
        adjusted_matrix = np.clip(adjusted_matrix, -128, 127)
        print(f"  Applied additional clipping")
    
    # Statistics
    non_zero = np.count_nonzero(adjusted_matrix)
    sparsity = 1.0 - non_zero / adjusted_matrix.size
    
    print(f"  Non-zero elements: {non_zero} ({100*(1-sparsity):.2f}%)")
    print(f"  Sparsity: {sparsity:.2%}")
    
    # Save results
    print(f"\n[Step 4] Saving results...")
    
    if args.output_file is None:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_dir = os.path.dirname(args.input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = os.path.join(output_dir, f"{base_name}_strict8bit_{timestamp}.csv")
    
    # Save template format
    csv_content = format_matrix_template(adjusted_matrix)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(csv_content)
    print(f"  ✓ Template CSV saved to: {args.output_file}")
    
    # Save standard format
    standard_csv = args.output_file.replace('.csv', '_standard.csv')
    np.savetxt(standard_csv, adjusted_matrix, delimiter=',', fmt='%d')
    print(f"  ✓ Standard CSV saved to: {standard_csv}")
    
    # Save metadata
    json_file = args.output_file.replace('.csv', '_metadata.json')
    metadata = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'input_file': args.input_file,
        'output_file': args.output_file,
        'original_range': [float(min_val), float(max_val)],
        'original_max_abs': float(max_abs),
        'final_range': [int(final_min), int(final_max)],
        'final_max_abs': int(final_max_abs),
        'target_max': args.target_max,
        'scale_factor': float(args.target_max / max_abs) if max_abs > 0 else 1.0,
        'matrix_info': {
            'shape': list(adjusted_matrix.shape),
            'dtype': str(adjusted_matrix.dtype),
            'min': int(np.min(adjusted_matrix)),
            'max': int(np.max(adjusted_matrix)),
            'non_zero_count': int(non_zero),
            'sparsity': float(sparsity)
        }
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Metadata saved to: {json_file}")
    
    print("\n" + "=" * 70)
    print("8-bit Range Enforcement Completed!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Original max abs: {max_abs:.2f}")
    print(f"  Final range: [{final_min}, {final_max}]")
    print(f"  Final max abs: {final_max_abs}")
    print(f"  ✓ All values in [-128, 127]")
    print(f"\nOutput files:")
    print(f"  - Template CSV: {args.output_file}")
    print(f"  - Standard CSV: {standard_csv}")
    print(f"  - Metadata JSON: {json_file}")


if __name__ == '__main__':
    main()

