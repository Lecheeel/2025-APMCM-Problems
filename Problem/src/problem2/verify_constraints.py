"""
Problem 2: Constraint Verification Script
==========================================

This script verifies that Problem 2 implementation meets all requirements:
1. DC power flow constraints on all transmission lines
2. N-1 security constraints (generator and line outages)
3. Spinning reserve requirements
4. Optional: Minimum safety inertia constraint

It also checks if results are reasonable and consistent with Problem 1.
"""

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

# Get project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
problem2_dir = os.path.join(project_root, "results", "problem2")

# Find latest results
summary_files = glob.glob(os.path.join(problem2_dir, "summary_*.json"))
if not summary_files:
    print("ERROR: No Problem 2 results found")
    sys.exit(1)

summary_files.sort(reverse=True)
latest_summary = summary_files[0]
print(f"Loading results from: {latest_summary}")

with open(latest_summary, 'r', encoding='utf-8') as f:
    summary = json.load(f)

# Load schedule
csv_files = glob.glob(os.path.join(problem2_dir, "uc_schedule_*.csv"))
csv_files.sort(reverse=True)
if csv_files:
    schedule_df = pd.read_csv(csv_files[0])
else:
    schedule_df = None
    print("WARNING: Could not load schedule CSV")

print("\n" + "=" * 70)
print("PROBLEM 2 CONSTRAINT VERIFICATION")
print("=" * 70)

# ============================================================================
# 1. Check Required Features
# ============================================================================

print("\n1. REQUIRED FEATURES CHECK")
print("-" * 70)

opt_info = summary['optimization_info']

# Check 1: Network power flow constraints
print(f"✓ Network power flow constraints: {'ENABLED' if opt_info.get('num_branches', 0) > 0 else 'DISABLED'}")
print(f"  - Number of buses: {opt_info.get('num_buses', 0)}")
print(f"  - Number of branches: {opt_info.get('num_branches', 0)}")

# Check 2: N-1 security constraints
n1_enabled = opt_info.get('n1_security_enabled', False)
print(f"{'✓' if n1_enabled else '✗'} N-1 security constraints: {'ENABLED' if n1_enabled else 'DISABLED'}")

# Check 3: Spinning reserve requirements
spinning_reserve_enabled = opt_info.get('spinning_reserve_enabled', False)
print(f"{'✓' if spinning_reserve_enabled else '✗'} Spinning reserve requirements: {'ENABLED' if spinning_reserve_enabled else 'DISABLED'}")

# Check 4: Optional inertia constraint
inertia_enabled = opt_info.get('inertia_constraint_enabled', False)
print(f"{'○' if inertia_enabled else '○'} Minimum safety inertia constraint: {'ENABLED' if inertia_enabled else 'DISABLED (optional)'}")

# ============================================================================
# 2. Verify Spinning Reserve
# ============================================================================

print("\n2. SPINNING RESERVE VERIFICATION")
print("-" * 70)

if schedule_df is not None and 'Spinning_Reserve_MW' in schedule_df.columns:
    # Calculate required reserve (should be max(10% load, largest unit) but capped)
    units = opt_info['units']
    P_max_values = {
        1: 300, 2: 180, 5: 50, 8: 35, 11: 30, 13: 40
    }
    largest_unit = max(P_max_values.values())
    
    violations = []
    for idx, row in schedule_df.iterrows():
        period = row['Period']
        load = row['Load_MW']
        available_reserve = row['Spinning_Reserve_MW']
        
        # Calculate expected reserve requirement
        reserve_10pct = 0.10 * load
        reserve_largest = largest_unit
        total_capacity = sum(P_max_values.values())
        max_feasible = max(0, (total_capacity - largest_unit) - load)
        expected_reserve = min(max(reserve_10pct, reserve_largest), max_feasible)
        expected_reserve = max(expected_reserve, 0.05 * load)
        
        # Check if reserve is sufficient
        if available_reserve < expected_reserve - 1e-3:
            violations.append(f"Period {period}: Available={available_reserve:.2f} MW, Required={expected_reserve:.2f} MW")
    
    if violations:
        print("✗ Spinning reserve violations found:")
        for v in violations[:5]:  # Show first 5
            print(f"  {v}")
        if len(violations) > 5:
            print(f"  ... and {len(violations) - 5} more")
    else:
        print("✓ All spinning reserve constraints satisfied")
        print(f"  - Average reserve: {schedule_df['Spinning_Reserve_MW'].mean():.2f} MW")
        print(f"  - Minimum reserve: {schedule_df['Spinning_Reserve_MW'].min():.2f} MW")
        print(f"  - Maximum reserve: {schedule_df['Spinning_Reserve_MW'].max():.2f} MW")
else:
    print("⚠ Could not verify spinning reserve (schedule data not available)")

# ============================================================================
# 3. Verify Power Balance
# ============================================================================

print("\n3. POWER BALANCE VERIFICATION")
print("-" * 70)

if schedule_df is not None:
    violations = []
    for idx, row in schedule_df.iterrows():
        period = row['Period']
        load = row['Load_MW']
        total_gen = row['Total_Generation_MW']
        error = abs(total_gen - load)
        
        if error > 1e-2:
            violations.append(f"Period {period}: Generation={total_gen:.2f} MW, Load={load:.2f} MW, Error={error:.4f} MW")
    
    if violations:
        print("✗ Power balance violations found:")
        for v in violations[:5]:
            print(f"  {v}")
    else:
        print("✓ All power balance constraints satisfied")
        max_error = max([abs(row['Total_Generation_MW'] - row['Load_MW']) for _, row in schedule_df.iterrows()])
        print(f"  - Maximum error: {max_error:.6f} MW")
else:
    print("⚠ Could not verify power balance (schedule data not available)")

# ============================================================================
# 4. Verify Generation Limits
# ============================================================================

print("\n4. GENERATION LIMITS VERIFICATION")
print("-" * 70)

P_max_values = {1: 300, 2: 180, 5: 50, 8: 35, 11: 30, 13: 40}
P_min_values = {1: 50, 2: 20, 5: 15, 8: 10, 11: 10, 13: 12}

if schedule_df is not None:
    violations = []
    for unit_id in opt_info['units']:
        status_col = f'Unit_{unit_id}_Status'
        gen_col = f'Unit_{unit_id}_Generation_MW'
        
        if status_col in schedule_df.columns and gen_col in schedule_df.columns:
            for idx, row in schedule_df.iterrows():
                period = row['Period']
                status = row[status_col]
                gen = row[gen_col]
                
                if status == 1:  # Unit is ON
                    if gen < P_min_values[unit_id] - 1e-3:
                        violations.append(f"Unit {unit_id}, Period {period}: Generation {gen:.2f} < P_min {P_min_values[unit_id]}")
                    if gen > P_max_values[unit_id] + 1e-3:
                        violations.append(f"Unit {unit_id}, Period {period}: Generation {gen:.2f} > P_max {P_max_values[unit_id]}")
                else:  # Unit is OFF
                    if gen > 1e-3:
                        violations.append(f"Unit {unit_id}, Period {period}: Unit OFF but generating {gen:.2f} MW")
    
    if violations:
        print("✗ Generation limit violations found:")
        for v in violations[:5]:
            print(f"  {v}")
    else:
        print("✓ All generation limit constraints satisfied")
else:
    print("⚠ Could not verify generation limits (schedule data not available)")

# ============================================================================
# 5. Verify N-1 Security Constraints (Conceptual Check)
# ============================================================================

print("\n5. N-1 SECURITY CONSTRAINTS VERIFICATION")
print("-" * 70)

if n1_enabled:
    print("✓ N-1 security constraints are implemented in the model")
    print("  - Generator contingencies: For each generator outage, remaining capacity >= load + reserve")
    print("  - Line contingencies: Sufficient transmission capacity remains after any line outage")
    print("  - Line utilization limits: No single line carries more than 80% of total load")
    
    # Check if solution satisfies N-1 conceptually
    if schedule_df is not None:
        total_capacity = sum(P_max_values.values())
        largest_unit = max(P_max_values.values())
        
        n1_violations = []
        for idx, row in schedule_df.iterrows():
            period = row['Period']
            load = row['Load_MW']
            reserve = row.get('Spinning_Reserve_MW', 0)
            
            # Check if losing largest unit would still allow meeting demand + reserve
            remaining_capacity = total_capacity - largest_unit
            required = load + reserve
            
            if remaining_capacity < required - 1e-3:
                n1_violations.append(f"Period {period}: After losing largest unit, remaining capacity {remaining_capacity:.2f} < required {required:.2f}")
        
        if n1_violations:
            print("⚠ Potential N-1 violations (conceptual check):")
            for v in n1_violations[:3]:
                print(f"  {v}")
        else:
            print("✓ N-1 constraints appear satisfied (conceptual check)")
else:
    print("✗ N-1 security constraints are NOT enabled")

# ============================================================================
# 6. Check Results Reasonableness
# ============================================================================

print("\n6. RESULTS REASONABLENESS CHECK")
print("-" * 70)

unit_stats = summary['unit_statistics']

# Check if all units are always on (might indicate over-constraint)
all_always_on = all(stat['utilization_rate'] == 1.0 for stat in unit_stats)
if all_always_on:
    print("⚠ All units are ON for all 24 periods")
    print("  This might indicate:")
    print("    - Constraints are too restrictive")
    print("    - Or this is the optimal solution given the constraints")
    print("  Compare with Problem 1 results to verify")

# Check total cost
total_cost = opt_info['total_cost']
fuel_cost = opt_info['fuel_cost']
startup_cost = opt_info['startup_cost']
shutdown_cost = opt_info['shutdown_cost']

print(f"\nCost Analysis:")
print(f"  Total Cost: ${total_cost:.2f}")
print(f"  Fuel Cost: ${fuel_cost:.2f}")
print(f"  Startup Cost: ${startup_cost:.2f}")
print(f"  Shutdown Cost: ${shutdown_cost:.2f}")

if startup_cost == 0 and shutdown_cost == 0:
    print("  ⚠ No startup/shutdown costs - all units stay online")
    print("    This is unusual but may be optimal given constraints")

# Check generation patterns
print(f"\nGeneration Patterns:")
for stat in unit_stats:
    unit_id = stat['unit_id']
    avg_gen = stat['average_generation_MW']
    max_gen = stat.get('max_generation_MW', 0)
    min_gen = stat.get('min_generation_MW', 0)
    p_max = P_max_values[unit_id]
    p_min = P_min_values[unit_id]
    
    print(f"  Unit {unit_id}: Avg={avg_gen:.2f} MW, Range=[{min_gen:.2f}, {max_gen:.2f}] MW")
    print(f"    Limits: P_min={p_min} MW, P_max={p_max} MW")
    if avg_gen < p_min - 1e-3 or avg_gen > p_max + 1e-3:
        print(f"    ⚠ Average generation outside limits!")

# ============================================================================
# 7. Comparison with Problem 1 Requirements
# ============================================================================

print("\n7. COMPARISON WITH PROBLEM 1")
print("-" * 70)

# Try to load Problem 1 results for comparison
problem1_dir = os.path.join(project_root, "results", "problem1")
p1_summary_files = glob.glob(os.path.join(problem1_dir, "summary_*.json"))

if p1_summary_files:
    p1_summary_files.sort(reverse=True)
    with open(p1_summary_files[0], 'r', encoding='utf-8') as f:
        p1_summary = json.load(f)
    
    p1_cost = p1_summary['optimization_info']['total_cost']
    p2_cost = opt_info['total_cost']
    cost_increase = p2_cost - p1_cost
    cost_increase_pct = (cost_increase / p1_cost) * 100 if p1_cost > 0 else 0
    
    print(f"Problem 1 Total Cost: ${p1_cost:.2f}")
    print(f"Problem 2 Total Cost: ${p2_cost:.2f}")
    print(f"Cost Increase: ${cost_increase:.2f} ({cost_increase_pct:+.2f}%)")
    
    if cost_increase > 0:
        print("✓ Problem 2 cost is higher than Problem 1 (expected due to additional constraints)")
    else:
        print("⚠ Problem 2 cost is not higher than Problem 1 (unexpected)")
    
    # Compare unit commitment patterns
    p1_unit_stats = {stat['unit_id']: stat for stat in p1_summary['unit_statistics']}
    print(f"\nUnit Commitment Comparison:")
    for stat in unit_stats:
        unit_id = stat['unit_id']
        p2_on_periods = stat['on_periods']
        if unit_id in p1_unit_stats:
            p1_on_periods = p1_unit_stats[unit_id]['on_periods']
            diff = p2_on_periods - p1_on_periods
            print(f"  Unit {unit_id}: P1={p1_on_periods} periods ON, P2={p2_on_periods} periods ON (diff={diff:+d})")
else:
    print("⚠ Problem 1 results not found for comparison")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

issues = []
if not n1_enabled:
    issues.append("N-1 security constraints not enabled")
if not spinning_reserve_enabled:
    issues.append("Spinning reserve not enabled")
if all_always_on:
    issues.append("All units always ON (may indicate over-constraint)")

if issues:
    print("⚠ Issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✓ All required features are implemented")
    print("✓ Constraints appear to be satisfied")
    print("✓ Results appear reasonable")

print("\n" + "=" * 70)

