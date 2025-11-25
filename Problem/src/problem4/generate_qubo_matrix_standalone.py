"""
Generate Standard QUBO Matrix Data for Problem 4 (Standalone Version)
=====================================================================

This script generates simplified QUBO matrix data for Problem 4 without requiring kaiwu.
It directly computes the QUBO matrix based on the mathematical formulation.
"""

import argparse
import numpy as np
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import modules (these don't require kaiwu)
from problem4.data_loader import ReducedDataLoader
from problem4.reduction_strategy import ReductionStrategy
from problem4.discretization import GenerationDiscretizer


def compute_qubo_matrix_directly(reduced_data: Dict, discretizer: GenerationDiscretizer,
                                 penalty_power_balance: float = None,
                                 penalty_ramp: float = None,
                                 penalty_reserve: float = None,
                                 penalty_n1: float = None) -> Tuple[np.ndarray, float]:
    """
    Compute QUBO matrix directly from mathematical formulation.
    
    QUBO form: min x^T Q x + c
    where x is binary vector, Q is symmetric matrix, c is constant offset.
    
    Returns:
        Tuple of (QUBO matrix, offset)
    """
    num_units = reduced_data['num_units']
    num_periods = reduced_data['num_periods']
    num_bits = discretizer.num_bits
    num_vars = num_units * num_periods * num_bits
    
    # Initialize QUBO matrix
    Q = np.zeros((num_vars, num_vars))
    offset = 0.0
    
    # Estimate penalty coefficients if not provided
    if penalty_power_balance is None:
        max_obj = np.sum(reduced_data['a_coeff'] * reduced_data['P_max']**2 + 
                        reduced_data['b_coeff'] * reduced_data['P_max'])
        min_obj = np.sum(reduced_data['a_coeff'] * reduced_data['P_min']**2 + 
                        reduced_data['b_coeff'] * reduced_data['P_min'])
        max_cost_diff = max_obj - min_obj
        max_load = np.max(reduced_data['load_demand'])
        min_total_gen = np.sum(reduced_data['P_min'])
        max_violation = max_load - min_total_gen
        penalty_power_balance = max(max_cost_diff * 100.0 / (max_violation**2) if max_violation > 0 else 1e4, 1e3)
    
    if penalty_ramp is None:
        max_obj = np.sum(reduced_data['a_coeff'] * reduced_data['P_max']**2)
        max_ramp_violation = np.max(reduced_data['P_max'] - reduced_data['P_min'])
        penalty_ramp = max(max_obj * 100.0 / max_ramp_violation**2, 1e2)
    
    if penalty_reserve is None:
        max_reserve_violation = np.max(reduced_data['load_demand'])
        max_obj = np.sum(reduced_data['a_coeff'] * reduced_data['P_max']**2)
        penalty_reserve = max(max_obj * 5.0 / max_reserve_violation**2, 1e-1)
    
    if penalty_n1 is None:
        max_n1_violation = np.max(reduced_data['load_demand'])
        max_obj = np.sum(reduced_data['a_coeff'] * reduced_data['P_max']**2)
        penalty_n1 = max(max_obj * 0.5 / max_n1_violation**2, 1e-2)
    
    # Scale factor for objective (to keep coefficients smaller)
    scale_factor = 0.01
    
    # Helper function to get variable index
    def var_idx(i, t, k):
        """Get variable index for unit i, period t, bit k."""
        return (i * num_periods + t) * num_bits + k
    
    # Helper function to get generation coefficient
    def get_gen_coeff(i, k):
        """Get coefficient for bit k of unit i."""
        return (2**k) * discretizer.delta[i]
    
    # 1. Objective function: fuel cost
    # Cost = sum_{i,t} [a_i * p_{i,t}^2 + b_i * p_{i,t} + c_i]
    # p_{i,t} = P_min[i] + sum_k (coeff_k * x_{i,t,k})
    for i in range(num_units):
        for t in range(num_periods):
            # Linear terms
            for k in range(num_bits):
                idx = var_idx(i, t, k)
                coeff = get_gen_coeff(i, k)
                
                # Linear term: b_i * coeff_k * x_{i,t,k}
                Q[idx, idx] += scale_factor * reduced_data['b_coeff'][i] * coeff
                
                # Quadratic term: a_i * coeff_k^2 * x_{i,t,k}^2
                Q[idx, idx] += scale_factor * reduced_data['a_coeff'][i] * coeff * coeff
                
                # Cross terms: 2 * a_i * coeff_k * coeff_l * x_{i,t,k} * x_{i,t,l}
                for l in range(k + 1, num_bits):
                    idx2 = var_idx(i, t, l)
                    coeff2 = get_gen_coeff(i, l)
                    Q[idx, idx2] += scale_factor * 2 * reduced_data['a_coeff'][i] * coeff * coeff2
            
            # Constant term: a_i * P_min[i]^2 + b_i * P_min[i] + c_i
            offset += scale_factor * (reduced_data['a_coeff'][i] * reduced_data['P_min'][i]**2 + 
                                     reduced_data['b_coeff'][i] * reduced_data['P_min'][i] + 
                                     reduced_data['c_coeff'][i])
            
            # Cross terms with P_min
            for k in range(num_bits):
                idx = var_idx(i, t, k)
                coeff = get_gen_coeff(i, k)
                Q[idx, idx] += scale_factor * 2 * reduced_data['a_coeff'][i] * reduced_data['P_min'][i] * coeff
    
    # 2. Power balance constraints: sum_i p_{i,t} = D_t
    # Penalty: lambda * (sum_i p_{i,t} - D_t)^2
    for t in range(num_periods):
        # Build total generation expression
        total_gen_const = np.sum(reduced_data['P_min'])
        total_gen_linear = np.zeros(num_vars)
        
        for i in range(num_units):
            for k in range(num_bits):
                idx = var_idx(i, t, k)
                coeff = get_gen_coeff(i, k)
                total_gen_linear[idx] = coeff
        
        # Penalty: lambda * (total_gen - D_t)^2
        # = lambda * [total_gen_const + sum(total_gen_linear * x) - D_t]^2
        error_const = total_gen_const - reduced_data['load_demand'][t]
        
        # Constant term
        offset += penalty_power_balance * error_const * error_const
        
        # Linear terms: 2 * lambda * error_const * total_gen_linear[i] * x[i]
        for i in range(num_vars):
            Q[i, i] += penalty_power_balance * 2 * error_const * total_gen_linear[i]
        
        # Quadratic terms: lambda * total_gen_linear[i] * total_gen_linear[j] * x[i] * x[j]
        for i in range(num_vars):
            for j in range(i, num_vars):
                if total_gen_linear[i] != 0 and total_gen_linear[j] != 0:
                    if i == j:
                        Q[i, j] += penalty_power_balance * total_gen_linear[i] * total_gen_linear[j]
                    else:
                        Q[i, j] += penalty_power_balance * 2 * total_gen_linear[i] * total_gen_linear[j]
    
    # 3. Ramp constraints
    # Ramp-up: p_{i,t} - p_{i,t-1} <= Ramp_Up[i]
    # Ramp-down: p_{i,t-1} - p_{i,t} <= Ramp_Down[i]
    for i in range(num_units):
        initial_power = reduced_data['P_min'][i]
        
        for t in range(num_periods):
            # Current power expression
            curr_gen_const = reduced_data['P_min'][i]
            curr_gen_linear = np.zeros(num_vars)
            for k in range(num_bits):
                idx = var_idx(i, t, k)
                coeff = get_gen_coeff(i, k)
                curr_gen_linear[idx] = coeff
            
            # Previous power expression
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
            
            # Ramp-up: (curr - prev - Ramp_Up)^2
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
            
            # Ramp-down: (prev - curr - Ramp_Down)^2
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
    
    # 4. Reserve constraints: sum_i (P_max[i] - p_{i,t}) >= R_t
    # Penalty: lambda * (R_t - available_reserve)^2
    for t in range(num_periods):
        available_reserve_const = np.sum(reduced_data['P_max']) - np.sum(reduced_data['P_min'])
        available_reserve_linear = np.zeros(num_vars)
        
        for i in range(num_units):
            for k in range(num_bits):
                idx = var_idx(i, t, k)
                coeff = get_gen_coeff(i, k)
                available_reserve_linear[idx] = -coeff  # Negative because reserve decreases with generation
        
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
    
    # 5. N-1 constraints (constant term only, as it doesn't depend on variables)
    for gen_out_idx in range(num_units):
        remaining_capacity = np.sum(reduced_data['P_max']) - reduced_data['P_max'][gen_out_idx]
        for t in range(num_periods):
            required_capacity = reduced_data['load_demand'][t] + reduced_data['spinning_reserve_req'][t]
            n1_error = required_capacity - remaining_capacity
            if n1_error > 0:
                offset += penalty_n1 * n1_error * n1_error
    
    # Make matrix symmetric (upper triangular)
    Q = np.triu(Q) + np.triu(Q, k=1).T
    
    return Q, offset


def generate_qubo_matrix_data(max_binary_vars: int = 100,
                              num_bits: int = 1,
                              time_aggregation: str = None,
                              results_dir: str = None,
                              output_file: str = None):
    """
    Generate standard QUBO matrix data for Problem 4.
    """
    print("=" * 70)
    print("Problem 4: Generate Standard QUBO Matrix Data (Standalone)")
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
    except Exception as e:
        print(f"  ✗ Error computing QUBO matrix: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 6: Prepare variable names
    print("\n[Step 6] Preparing variable information...")
    variable_names = []
    for i in range(reduced_data['num_units']):
        for t in range(reduced_data['num_periods']):
            for k in range(reduction_info['num_bits']):
                var_name = discretizer.get_variable_name(i, t, k)
                variable_names.append(var_name)
    
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
    
    print(f"  Matrix shape: {qubo_matrix.shape}")
    print(f"  Matrix statistics:")
    print(f"    Min: {matrix_stats['min']:.6e}")
    print(f"    Max: {matrix_stats['max']:.6e}")
    print(f"    Non-zero elements: {matrix_stats['non_zero_count']} ({100*(1-matrix_stats['sparsity']):.2f}%)")
    
    # Step 7: Prepare output data
    print("\n[Step 7] Preparing output data...")
    
    qubo_data = {
        'metadata': {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'problem': 'Problem 4: Problem Scale Reduction under Quantum Hardware Constraints',
            'method': 'direct_computation',
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
                'statistics': matrix_stats
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
        description='Generate standard QUBO matrix data for Problem 4 (standalone version)',
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

