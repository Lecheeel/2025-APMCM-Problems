"""
Generate QUBO Matrix with 8-bit Integer Precision Constraint
=============================================================

This script generates QUBO matrix directly in 8-bit integer range [-128, 127]
by applying intelligent scaling during construction.
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


def estimate_matrix_scale(reduced_data: Dict, discretizer: GenerationDiscretizer,
                         num_periods: int) -> float:
    """
    Estimate the maximum scale factor needed to fit matrix in 8-bit range.
    
    Returns:
        Scale factor to apply to all coefficients
    """
    num_units = reduced_data['num_units']
    num_bits = discretizer.num_bits
    num_vars = num_units * num_periods * num_bits
    
    # Estimate maximum coefficient magnitudes
    
    # 1. Objective function terms
    max_obj_coeff = 0.0
    for i in range(num_units):
        max_coeff = discretizer.delta[i] if num_bits == 1 else discretizer.delta[i] * (2**num_bits - 1)
        # Quadratic term: a_i * coeff^2
        quad_term = reduced_data['a_coeff'][i] * max_coeff * max_coeff
        # Linear term: b_i * coeff
        lin_term = reduced_data['b_coeff'][i] * max_coeff
        max_obj_coeff = max(max_obj_coeff, quad_term, lin_term)
    
    # 2. Power balance constraint terms
    max_pb_coeff = 0.0
    max_load = np.max(reduced_data['load_demand'])
    min_total_gen = np.sum(reduced_data['P_min'])
    max_violation = max_load - min_total_gen
    
    # Estimate penalty coefficient (scaled down)
    max_cost_diff = np.sum(reduced_data['a_coeff'] * reduced_data['P_max']**2 + 
                          reduced_data['b_coeff'] * reduced_data['P_max'])
    # Use much smaller penalty multiplier for 8-bit compatibility
    penalty_pb = max_cost_diff * 0.1 / (max_violation**2) if max_violation > 0 else 1.0
    
    # Maximum coefficient from power balance: penalty * (sum of max coeffs)^2
    max_sum_coeff = np.sum(discretizer.delta)
    max_pb_coeff = penalty_pb * max_sum_coeff * max_sum_coeff
    
    # 3. Ramp constraint terms
    max_ramp_coeff = 0.0
    max_ramp_violation = np.max(reduced_data['P_max'] - reduced_data['P_min'])
    penalty_ramp = max_cost_diff * 0.1 / max_ramp_violation**2 if max_ramp_violation > 0 else 1.0
    max_ramp_coeff = penalty_ramp * max(discretizer.delta)**2
    
    # 4. Reserve constraint terms
    max_reserve_coeff = 0.0
    max_reserve_violation = np.max(reduced_data['load_demand'])
    penalty_reserve = max_cost_diff * 0.01 / max_reserve_violation**2 if max_reserve_violation > 0 else 0.1
    max_reserve_coeff = penalty_reserve * max_sum_coeff**2
    
    # Find maximum coefficient
    max_coeff = max(max_obj_coeff, max_pb_coeff, max_ramp_coeff, max_reserve_coeff)
    
    # Calculate scale factor to fit in [-127, 127] (leave some margin)
    target_max = 120.0  # Leave margin for rounding
    if max_coeff > 0:
        scale_factor = target_max / max_coeff
    else:
        scale_factor = 1.0
    
    # Apply additional safety scaling
    scale_factor *= 0.8  # 80% of calculated scale for safety margin
    
    return scale_factor


def compute_qubo_matrix_8bit(reduced_data: Dict, discretizer: GenerationDiscretizer,
                              target_max: float = 120.0) -> Tuple[np.ndarray, float, Dict]:
    """
    Compute QUBO matrix directly scaled to 8-bit integer range.
    
    Args:
        reduced_data: Reduced problem data
        discretizer: GenerationDiscretizer instance
        target_max: Target maximum value (default: 120, leaving margin for 127)
    
    Returns:
        Tuple of (QUBO matrix, offset, metadata)
    """
    num_units = reduced_data['num_units']
    num_periods = reduced_data['num_periods']
    num_bits = discretizer.num_bits
    num_vars = num_units * num_periods * num_bits
    
    # Estimate scale factor
    estimated_scale = estimate_matrix_scale(reduced_data, discretizer, num_periods)
    print(f"  Estimated scale factor: {estimated_scale:.6e}")
    
    # Initialize QUBO matrix
    Q = np.zeros((num_vars, num_vars), dtype=np.float64)
    offset = 0.0
    
    # Helper functions
    def var_idx(i, t, k):
        return (i * num_periods + t) * num_bits + k
    
    def get_gen_coeff(i, k):
        return (2**k) * discretizer.delta[i]
    
    # Calculate penalty coefficients (scaled down for 8-bit compatibility)
    max_cost_diff = np.sum(reduced_data['a_coeff'] * reduced_data['P_max']**2 + 
                          reduced_data['b_coeff'] * reduced_data['P_max'])
    max_load = np.max(reduced_data['load_demand'])
    min_total_gen = np.sum(reduced_data['P_min'])
    max_violation = max_load - min_total_gen
    
    # Much smaller penalty multipliers for 8-bit compatibility
    penalty_power_balance = max_cost_diff * 0.1 / (max_violation**2) if max_violation > 0 else 1.0
    penalty_power_balance = min(penalty_power_balance, 100.0)  # Cap at 100
    
    max_ramp_violation = np.max(reduced_data['P_max'] - reduced_data['P_min'])
    penalty_ramp = max_cost_diff * 0.1 / max_ramp_violation**2 if max_ramp_violation > 0 else 1.0
    penalty_ramp = min(penalty_ramp, 50.0)  # Cap at 50
    
    max_reserve_violation = np.max(reduced_data['load_demand'])
    penalty_reserve = max_cost_diff * 0.01 / max_reserve_violation**2 if max_reserve_violation > 0 else 0.1
    penalty_reserve = min(penalty_reserve, 10.0)  # Cap at 10
    
    penalty_n1 = max_cost_diff * 0.001 / max_reserve_violation**2 if max_reserve_violation > 0 else 0.01
    penalty_n1 = min(penalty_n1, 1.0)  # Cap at 1
    
    # Apply overall scale factor to objective
    obj_scale = estimated_scale * 0.1  # Further scale down objective
    
    print(f"  Penalty coefficients (scaled):")
    print(f"    Power balance: {penalty_power_balance:.2e}")
    print(f"    Ramp: {penalty_ramp:.2e}")
    print(f"    Reserve: {penalty_reserve:.2e}")
    print(f"    N-1: {penalty_n1:.2e}")
    print(f"    Objective scale: {obj_scale:.6e}")
    
    # 1. Objective function
    for i in range(num_units):
        for t in range(num_periods):
            for k in range(num_bits):
                idx = var_idx(i, t, k)
                coeff = get_gen_coeff(i, k)
                
                # Linear term
                Q[idx, idx] += obj_scale * reduced_data['b_coeff'][i] * coeff
                
                # Quadratic term
                Q[idx, idx] += obj_scale * reduced_data['a_coeff'][i] * coeff * coeff
                
                # Cross terms
                for l in range(k + 1, num_bits):
                    idx2 = var_idx(i, t, l)
                    coeff2 = get_gen_coeff(i, l)
                    Q[idx, idx2] += obj_scale * 2 * reduced_data['a_coeff'][i] * coeff * coeff2
            
            # Constant term
            offset += obj_scale * (reduced_data['a_coeff'][i] * reduced_data['P_min'][i]**2 + 
                                 reduced_data['b_coeff'][i] * reduced_data['P_min'][i] + 
                                 reduced_data['c_coeff'][i])
            
            # Cross terms with P_min
            for k in range(num_bits):
                idx = var_idx(i, t, k)
                coeff = get_gen_coeff(i, k)
                Q[idx, idx] += obj_scale * 2 * reduced_data['a_coeff'][i] * reduced_data['P_min'][i] * coeff
    
    # 2. Power balance constraints
    for t in range(num_periods):
        total_gen_const = np.sum(reduced_data['P_min'])
        total_gen_linear = np.zeros(num_vars)
        
        for i in range(num_units):
            for k in range(num_bits):
                idx = var_idx(i, t, k)
                coeff = get_gen_coeff(i, k)
                total_gen_linear[idx] = coeff
        
        error_const = total_gen_const - reduced_data['load_demand'][t]
        
        offset += penalty_power_balance * error_const * error_const
        
        for i in range(num_vars):
            Q[i, i] += penalty_power_balance * 2 * error_const * total_gen_linear[i]
        
        for i in range(num_vars):
            for j in range(i, num_vars):
                if total_gen_linear[i] != 0 and total_gen_linear[j] != 0:
                    if i == j:
                        Q[i, j] += penalty_power_balance * total_gen_linear[i] * total_gen_linear[j]
                    else:
                        Q[i, j] += penalty_power_balance * 2 * total_gen_linear[i] * total_gen_linear[j]
    
    # 3. Ramp constraints
    for i in range(num_units):
        initial_power = reduced_data['P_min'][i]
        
        for t in range(num_periods):
            curr_gen_const = reduced_data['P_min'][i]
            curr_gen_linear = np.zeros(num_vars)
            for k in range(num_bits):
                idx = var_idx(i, t, k)
                coeff = get_gen_coeff(i, k)
                curr_gen_linear[idx] = coeff
            
            if t == 0:
                prev_gen_const = initial_power
                prev_gen_linear = np.zeros(num_vars)
            else:
                prev_gen_const = reduced_data['P_min'][i]
                prev_gen_linear = np.zeros(num_vars)
                for k in range(num_bits):
                    idx = var_idx(i, t - 1, k)
                    coeff = get_gen_coeff(i, k)
                    prev_gen_linear[idx] = coeff
            
            # Ramp-up
            ramp_up_error_const = curr_gen_const - prev_gen_const - reduced_data['Ramp_Up'][i]
            ramp_up_error_linear = curr_gen_linear - prev_gen_linear
            
            offset += penalty_ramp * ramp_up_error_const * ramp_up_error_const
            for idx in range(num_vars):
                Q[idx, idx] += penalty_ramp * 2 * ramp_up_error_const * ramp_up_error_linear[idx]
            for idx1 in range(num_vars):
                for idx2 in range(idx1, num_vars):
                    if ramp_up_error_linear[idx1] != 0 and ramp_up_error_linear[idx2] != 0:
                        if idx1 == idx2:
                            Q[idx1, idx2] += penalty_ramp * ramp_up_error_linear[idx1] * ramp_up_error_linear[idx2]
                        else:
                            Q[idx1, idx2] += penalty_ramp * 2 * ramp_up_error_linear[idx1] * ramp_up_error_linear[idx2]
            
            # Ramp-down
            ramp_down_error_const = prev_gen_const - curr_gen_const - reduced_data['Ramp_Down'][i]
            ramp_down_error_linear = prev_gen_linear - curr_gen_linear
            
            offset += penalty_ramp * ramp_down_error_const * ramp_down_error_const
            for idx in range(num_vars):
                Q[idx, idx] += penalty_ramp * 2 * ramp_down_error_const * ramp_down_error_linear[idx]
            for idx1 in range(num_vars):
                for idx2 in range(idx1, num_vars):
                    if ramp_down_error_linear[idx1] != 0 and ramp_down_error_linear[idx2] != 0:
                        if idx1 == idx2:
                            Q[idx1, idx2] += penalty_ramp * ramp_down_error_linear[idx1] * ramp_down_error_linear[idx2]
                        else:
                            Q[idx1, idx2] += penalty_ramp * 2 * ramp_down_error_linear[idx1] * ramp_down_error_linear[idx2]
    
    # 4. Reserve constraints
    for t in range(num_periods):
        available_reserve_const = np.sum(reduced_data['P_max']) - np.sum(reduced_data['P_min'])
        available_reserve_linear = np.zeros(num_vars)
        
        for i in range(num_units):
            for k in range(num_bits):
                idx = var_idx(i, t, k)
                coeff = get_gen_coeff(i, k)
                available_reserve_linear[idx] = -coeff
        
        reserve_error_const = reduced_data['spinning_reserve_req'][t] - available_reserve_const
        reserve_error_linear = -available_reserve_linear
        
        offset += penalty_reserve * reserve_error_const * reserve_error_const
        for idx in range(num_vars):
            Q[idx, idx] += penalty_reserve * 2 * reserve_error_const * reserve_error_linear[idx]
        for idx1 in range(num_vars):
            for idx2 in range(idx1, num_vars):
                if reserve_error_linear[idx1] != 0 and reserve_error_linear[idx2] != 0:
                    if idx1 == idx2:
                        Q[idx1, idx2] += penalty_reserve * reserve_error_linear[idx1] * reserve_error_linear[idx2]
                    else:
                        Q[idx1, idx2] += penalty_reserve * 2 * reserve_error_linear[idx1] * reserve_error_linear[idx2]
    
    # 5. N-1 constraints (constant only)
    for gen_out_idx in range(num_units):
        remaining_capacity = np.sum(reduced_data['P_max']) - reduced_data['P_max'][gen_out_idx]
        for t in range(num_periods):
            required_capacity = reduced_data['load_demand'][t] + reduced_data['spinning_reserve_req'][t]
            n1_error = required_capacity - remaining_capacity
            if n1_error > 0:
                offset += penalty_n1 * n1_error * n1_error
    
    # Make symmetric
    Q = np.triu(Q) + np.triu(Q, k=1).T
    
    # Check if we need further scaling
    max_val = np.max(np.abs(Q))
    if max_val > target_max:
        additional_scale = target_max / max_val
        Q = Q * additional_scale
        offset = offset * additional_scale
        print(f"  Applied additional scaling: {additional_scale:.6f}")
    
    # Round to integers
    Q_int = np.round(Q).astype(np.int32)
    Q_int = np.clip(Q_int, -128, 127)
    
    # Make symmetric again after rounding
    Q_int = np.triu(Q_int) + np.triu(Q_int, k=1).T
    
    metadata = {
        'scale_factor': float(estimated_scale),
        'obj_scale': float(obj_scale),
        'penalty_power_balance': float(penalty_power_balance),
        'penalty_ramp': float(penalty_ramp),
        'penalty_reserve': float(penalty_reserve),
        'penalty_n1': float(penalty_n1),
        'original_max': float(np.max(np.abs(Q))),
        'final_max': int(np.max(np.abs(Q_int))),
        'final_min': int(np.min(Q_int))
    }
    
    return Q_int, offset, metadata


def format_matrix_for_template(qubo_matrix: np.ndarray) -> str:
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


def generate_qubo_matrix_8bit(max_binary_vars: int = 100,
                               num_bits: int = 1,
                               time_aggregation: str = None,
                               results_dir: str = None,
                               output_file: str = None):
    """Generate QUBO matrix directly in 8-bit integer range."""
    print("=" * 70)
    print("Problem 4: Generate QUBO Matrix (8-bit Integer Range)")
    print("=" * 70)
    
    # Load data and apply reduction
    print("\n[Step 1] Loading data and applying reduction...")
    data_loader = ReducedDataLoader(results_dir=results_dir)
    reduction_strategy = ReductionStrategy(data_loader)
    
    args_num_bits = num_bits
    summary = data_loader.get_summary()
    
    if summary['num_units'] * summary['num_periods'] * args_num_bits > max_binary_vars:
        if time_aggregation is None:
            time_aggregation = 'uniform'
    
    reduction_info = reduction_strategy.apply_reduction(
        max_binary_vars=max_binary_vars,
        num_bits=args_num_bits,
        time_aggregation=time_aggregation,
        unit_selection=None,
        force_no_reduction=False
    )
    
    reduced_data = reduction_strategy.get_reduced_data()
    discretizer = GenerationDiscretizer(
        P_min=reduced_data['P_min'],
        P_max=reduced_data['P_max'],
        num_bits=reduction_info['num_bits']
    )
    
    print(f"  Binary variables: {reduction_info['total_binary_vars']}")
    
    # Compute QUBO matrix with 8-bit constraint
    print("\n[Step 2] Computing QUBO matrix with 8-bit constraint...")
    qubo_matrix, qubo_offset, metadata = compute_qubo_matrix_8bit(reduced_data, discretizer)
    
    print(f"  Matrix shape: {qubo_matrix.shape}")
    print(f"  Matrix range: [{np.min(qubo_matrix)}, {np.max(qubo_matrix)}]")
    print(f"  ✓ Matrix is within 8-bit range!")
    
    # Save results
    print("\n[Step 3] Saving results...")
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        output_dir = os.path.join(project_root, 'results', 'problem4')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"qubo_matrix_8bit_{timestamp}.csv")
    
    # Save template format
    csv_content = format_matrix_for_template(qubo_matrix)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(csv_content)
    print(f"  ✓ Template CSV saved to: {output_file}")
    
    # Save standard format
    standard_csv = output_file.replace('.csv', '_standard.csv')
    np.savetxt(standard_csv, qubo_matrix, delimiter=',', fmt='%d')
    print(f"  ✓ Standard CSV saved to: {standard_csv}")
    
    # Save metadata
    json_file = output_file.replace('.csv', '_metadata.json')
    full_metadata = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'reduction_info': reduction_info,
        'matrix_info': {
            'shape': list(qubo_matrix.shape),
            'offset': float(qubo_offset),
            'min': int(np.min(qubo_matrix)),
            'max': int(np.max(qubo_matrix)),
            'non_zero_count': int(np.count_nonzero(qubo_matrix)),
            'sparsity': float(1.0 - np.count_nonzero(qubo_matrix) / qubo_matrix.size)
        },
        'scaling_info': metadata
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(full_metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Metadata saved to: {json_file}")
    
    print("\n" + "=" * 70)
    print("QUBO Matrix Generation Completed!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Matrix size: {qubo_matrix.shape[0]} × {qubo_matrix.shape[1]}")
    print(f"  Value range: [{np.min(qubo_matrix)}, {np.max(qubo_matrix)}]")
    print(f"  ✓ Within 8-bit range [-128, 127]")
    print(f"  Output files:")
    print(f"    - Template CSV: {output_file}")
    print(f"    - Standard CSV: {standard_csv}")
    print(f"    - Metadata JSON: {json_file}")
    
    return qubo_matrix, qubo_offset, full_metadata


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate QUBO matrix directly in 8-bit integer range',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
    
    args = parser.parse_args()
    
    if args.results_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        args.results_dir = os.path.join(project_root, 'results', 'problem2')
    
    generate_qubo_matrix_8bit(
        max_binary_vars=args.max_binary_vars,
        num_bits=args.num_bits,
        time_aggregation=args.time_aggregation,
        results_dir=args.results_dir,
        output_file=args.output_file
    )


if __name__ == '__main__':
    main()

