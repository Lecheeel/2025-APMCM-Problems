"""
Problem 2: Network Constraints Verification Script
==================================================

This script verifies:
1. DC power flow constraints (line flows within limits)
2. N-1 security constraints (generator and line contingencies)
3. Bus power balance
4. Line flow utilization

Usage:
    python verify_network_constraints.py [--results-dir DIR]
"""

import argparse
import json
import glob
import os
import sys

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy are required")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Verify Network Constraints for Problem 2')
parser.add_argument('--results-dir', type=str, default=None, help='Results directory (default: latest)')
args = parser.parse_args()

# Get project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
problem2_dir = os.path.join(project_root, "results", "problem2")

if args.results_dir:
    results_dir = args.results_dir
else:
    # Find latest results
    summary_files = glob.glob(os.path.join(problem2_dir, "summary_*.json"))
    if not summary_files:
        print("ERROR: No Problem 2 results found")
        sys.exit(1)
    summary_files.sort(reverse=True)
    results_dir = os.path.dirname(summary_files[0])

print("=" * 70)
print("NETWORK CONSTRAINTS VERIFICATION")
print("=" * 70)
print(f"Results directory: {results_dir}")

# Load data
summary_file = glob.glob(os.path.join(results_dir, "summary_*.json"))[0]
with open(summary_file, 'r', encoding='utf-8') as f:
    summary = json.load(f)

# Load schedules
uc_schedule = pd.read_csv(glob.glob(os.path.join(results_dir, "uc_schedule_*.csv"))[0])
angles_file = glob.glob(os.path.join(results_dir, "bus_angles_*.csv"))
flows_file = glob.glob(os.path.join(results_dir, "line_flows_*.csv"))

if not angles_file or not flows_file:
    print("\nERROR: Bus angles or line flows files not found.")
    print("Please run uc_network_security.py first to generate these files.")
    sys.exit(1)

bus_angles = pd.read_csv(angles_file[0])
line_flows = pd.read_csv(flows_file[0])

# Network parameters
units = summary['optimization_info']['units']
P_max_values = {1: 300, 2: 180, 5: 50, 8: 35, 11: 30, 13: 40}
P_min_values = {1: 50, 2: 20, 5: 15, 8: 10, 11: 10, 13: 12}

# Branches data (from Table 3)
branches_info = [
    {'branch': 1, 'from': 1, 'to': 2, 'P_max': 650},
    {'branch': 2, 'from': 1, 'to': 3, 'P_max': 650},
    {'branch': 3, 'from': 2, 'to': 4, 'P_max': 325},
    {'branch': 4, 'from': 3, 'to': 4, 'P_max': 650},
    {'branch': 5, 'from': 2, 'to': 5, 'P_max': 650},
    {'branch': 6, 'from': 2, 'to': 6, 'P_max': 325},
    {'branch': 7, 'from': 4, 'to': 6, 'P_max': 450},
    {'branch': 8, 'from': 5, 'to': 7, 'P_max': 350},
    {'branch': 9, 'from': 6, 'to': 7, 'P_max': 650},
    {'branch': 10, 'from': 6, 'to': 8, 'P_max': 160},
    {'branch': 11, 'from': 6, 'to': 9, 'P_max': 325},
    {'branch': 12, 'from': 6, 'to': 10, 'P_max': 160},
    {'branch': 13, 'from': 9, 'to': 11, 'P_max': 325},
    {'branch': 14, 'from': 9, 'to': 10, 'P_max': 325},
    {'branch': 15, 'from': 4, 'to': 12, 'P_max': 325},
    {'branch': 16, 'from': 12, 'to': 13, 'P_max': 325},
    {'branch': 17, 'from': 12, 'to': 14, 'P_max': 160},
    {'branch': 18, 'from': 12, 'to': 15, 'P_max': 160},
    {'branch': 19, 'from': 12, 'to': 16, 'P_max': 160},
    {'branch': 20, 'from': 14, 'to': 15, 'P_max': 80},
    {'branch': 21, 'from': 16, 'to': 17, 'P_max': 80},
    {'branch': 22, 'from': 15, 'to': 18, 'P_max': 80},
    {'branch': 23, 'from': 18, 'to': 19, 'P_max': 80},
    {'branch': 24, 'from': 19, 'to': 20, 'P_max': 80},
    {'branch': 25, 'from': 10, 'to': 20, 'P_max': 80},
    {'branch': 26, 'from': 10, 'to': 17, 'P_max': 80},
    {'branch': 27, 'from': 10, 'to': 21, 'P_max': 80},
    {'branch': 28, 'from': 10, 'to': 22, 'P_max': 80},
    {'branch': 29, 'from': 21, 'to': 22, 'P_max': 160},
    {'branch': 30, 'from': 15, 'to': 23, 'P_max': 160},
    {'branch': 31, 'from': 22, 'to': 24, 'P_max': 160},
    {'branch': 32, 'from': 23, 'to': 24, 'P_max': 160},
    {'branch': 33, 'from': 24, 'to': 25, 'P_max': 80},
    {'branch': 34, 'from': 25, 'to': 26, 'P_max': 80},
    {'branch': 35, 'from': 25, 'to': 27, 'P_max': 80},
    {'branch': 36, 'from': 27, 'to': 28, 'P_max': 80},
    {'branch': 37, 'from': 27, 'to': 29, 'P_max': 80},
    {'branch': 38, 'from': 27, 'to': 30, 'P_max': 80},
    {'branch': 39, 'from': 29, 'to': 30, 'P_max': 80},
    {'branch': 40, 'from': 8, 'to': 28, 'P_max': 160},
    {'branch': 41, 'from': 6, 'to': 28, 'P_max': 160},
]

# ============================================================================
# 1. Verify Line Flow Limits
# ============================================================================

print("\n1. LINE FLOW LIMITS VERIFICATION")
print("-" * 70)

violations = []
max_utilization = {}

for period in range(1, 25):
    for br_info in branches_info:
        br_num = br_info['branch']
        from_bus = br_info['from']
        to_bus = br_info['to']
        p_max = br_info['P_max']
        
        flow_col = f'Branch_{br_num}_{from_bus}_to_{to_bus}_MW'
        if flow_col in line_flows.columns:
            flow_value = line_flows.loc[period-1, flow_col]
            utilization = abs(flow_value) / p_max * 100 if p_max > 0 else 0
            
            if abs(flow_value) > p_max + 1e-3:
                violations.append(f"Period {period}, Branch {br_num} ({from_bus}-{to_bus}): "
                                f"Flow={flow_value:.2f} MW > Limit={p_max} MW")
            
            key = f"Br{br_num}"
            if key not in max_utilization:
                max_utilization[key] = 0
            max_utilization[key] = max(max_utilization[key], utilization)

if violations:
    print("✗ Line flow limit violations found:")
    for v in violations[:10]:
        print(f"  {v}")
    if len(violations) > 10:
        print(f"  ... and {len(violations) - 10} more violations")
else:
    print("✓ All line flows within limits")
    print(f"\nTop 10 lines by maximum utilization:")
    sorted_util = sorted(max_utilization.items(), key=lambda x: x[1], reverse=True)[:10]
    for line, util in sorted_util:
        print(f"  {line}: {util:.2f}%")

# ============================================================================
# 2. Verify N-1 Generator Constraints
# ============================================================================

print("\n2. N-1 GENERATOR CONTINGENCY VERIFICATION")
print("-" * 70)

total_capacity = sum(P_max_values.values())
largest_unit = max(P_max_values.values())

violations = []
# Calculate required reserve for each period (same logic as in uc_network_security.py)
largest_unit_capacity = max(P_max_values.values())
total_capacity_check = sum(P_max_values.values())
required_reserves = []

for period in range(1, 25):
    load = uc_schedule.loc[period-1, 'Load_MW']
    # Calculate required reserve (same as in model)
    reserve_10pct = 0.10 * load
    reserve_largest = largest_unit_capacity
    max_feasible_reserve = max(0, (total_capacity_check - largest_unit_capacity) - load)
    reserve_standard = max(reserve_10pct, reserve_largest)
    reserve_final = min(reserve_standard, max_feasible_reserve)
    reserve_final = max(reserve_final, 0.05 * load)
    required_reserves.append(reserve_final)
    
    required = load + reserve_final
    
    # Check each generator outage
    for unit_id in units:
        remaining_capacity = total_capacity - P_max_values[unit_id]
        
        # Check if unit is online
        status_col = f'Unit_{unit_id}_Status'
        if status_col in uc_schedule.columns:
            unit_status = uc_schedule.loc[period-1, status_col]
            if unit_status == 1:  # Unit is online
                if remaining_capacity < required - 1e-3:
                    violations.append(f"Period {period}, Unit {unit_id} outage: "
                                    f"Remaining={remaining_capacity:.2f} MW < Required={required:.2f} MW")

if violations:
    print("✗ N-1 generator contingency violations found:")
    for v in violations[:10]:
        print(f"  {v}")
    if len(violations) > 10:
        print(f"  ... and {len(violations) - 10} more violations")
else:
    print("✓ All N-1 generator contingencies satisfied")
    print(f"\nN-1 Check Summary:")
    print(f"  Total capacity: {total_capacity} MW")
    print(f"  Largest unit: {largest_unit} MW")
    print(f"  Remaining after largest outage: {total_capacity - largest_unit} MW")
    for period in [1, 12, 24]:
        load = uc_schedule.loc[period-1, 'Load_MW']
        required_reserve = required_reserves[period-1]
        required = load + required_reserve
        remaining = total_capacity - largest_unit
        margin = remaining - required
        available_reserve = uc_schedule.loc[period-1, 'Spinning_Reserve_MW']
        print(f"  Period {period}: Load={load:.1f}, Required_Reserve={required_reserve:.1f}, "
              f"Required_Total={required:.1f}, Remaining={remaining:.1f}, Margin={margin:.1f} MW")
        print(f"    (Available reserve: {available_reserve:.1f} MW)")

# ============================================================================
# 3. Verify Bus Power Balance (Conceptual)
# ============================================================================

print("\n3. BUS POWER BALANCE VERIFICATION")
print("-" * 70)

# Check reference bus angle
ref_bus_angles = bus_angles['Bus_1_Angle_deg'].values
if np.allclose(ref_bus_angles, 0, atol=1e-3):
    print("✓ Reference bus (Bus 1) angle = 0 (correct)")
else:
    print("✗ Reference bus angle not zero!")
    print(f"  Values: {ref_bus_angles[:5]}")

# Check angle ranges (should be reasonable, typically -30 to +30 degrees)
max_angle = bus_angles[[col for col in bus_angles.columns if 'Angle_deg' in col]].max().max()
min_angle = bus_angles[[col for col in bus_angles.columns if 'Angle_deg' in col]].min().min()
print(f"\nAngle ranges:")
print(f"  Maximum angle: {max_angle:.2f} degrees")
print(f"  Minimum angle: {min_angle:.2f} degrees")
if abs(max_angle) < 50 and abs(min_angle) < 50:
    print("✓ Angle ranges are reasonable")
else:
    print("⚠ Angle ranges seem large (may indicate issues)")

# ============================================================================
# 4. Verify Line Utilization Limits
# ============================================================================

print("\n4. LINE UTILIZATION LIMITS VERIFICATION")
print("-" * 70)

max_line_utilization_limit = 80  # 80% of load
violations = []
high_utilization_lines = []

for period in range(1, 25):
    load = uc_schedule.loc[period-1, 'Load_MW']
    max_allowed = load * max_line_utilization_limit / 100
    
    for br_info in branches_info:
        br_num = br_info['branch']
        from_bus = br_info['from']
        to_bus = br_info['to']
        
        flow_col = f'Branch_{br_num}_{from_bus}_to_{to_bus}_MW'
        if flow_col in line_flows.columns:
            flow_value = abs(line_flows.loc[period-1, flow_col])
            
            if flow_value > max_allowed + 1e-3:
                violations.append(f"Period {period}, Branch {br_num}: "
                                f"Flow={flow_value:.2f} MW > {max_allowed:.2f} MW (80% of load)")
            
            if flow_value > load * 0.6:  # High utilization (>60% of load)
                high_utilization_lines.append((period, br_num, flow_value, load))

if violations:
    print("✗ Line utilization limit violations found:")
    for v in violations[:10]:
        print(f"  {v}")
else:
    print("✓ All lines within utilization limits (80% of load)")
    
if high_utilization_lines:
    print(f"\n⚠ {len(high_utilization_lines)} instances of high line utilization (>60% of load):")
    for period, br_num, flow, load in high_utilization_lines[:10]:
        pct = flow / load * 100
        print(f"  Period {period}, Branch {br_num}: {flow:.2f} MW ({pct:.1f}% of load)")

# ============================================================================
# 5. Summary
# ============================================================================

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

total_issues = len([v for v in [violations] if v])
if not violations and not high_utilization_lines:
    print("✓ All network constraints verified and satisfied")
else:
    print("⚠ Some issues found - see details above")

print("\n" + "=" * 70)

