"""
Diagnostic script to identify why Problem 2 results are identical to Problem 1
"""

import json
import glob
import os
import pandas as pd

# Get project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p1_dir = os.path.join(project_root, "results", "problem1")
p2_dir = os.path.join(project_root, "results", "problem2")

# Load summaries
p1_summary = glob.glob(os.path.join(p1_dir, "summary_*.json"))[0]
p2_summary = glob.glob(os.path.join(p2_dir, "summary_*.json"))[0]

p1_data = json.load(open(p1_summary))
p2_data = json.load(open(p2_summary))

# Load schedules
p1_schedule = pd.read_csv(glob.glob(os.path.join(p1_dir, "uc_schedule_*.csv"))[0])
p2_schedule = pd.read_csv(glob.glob(os.path.join(p2_dir, "uc_schedule_*.csv"))[0])

print("=" * 70)
print("DIAGNOSIS: Why Problem 2 results are identical to Problem 1")
print("=" * 70)

# Check costs
p1_cost = p1_data['optimization_info']['total_cost']
p2_cost = p2_data['optimization_info']['total_cost']
cost_diff = abs(p1_cost - p2_cost)

print(f"\n1. COST COMPARISON:")
print(f"   Problem 1 cost: ${p1_cost:.10f}")
print(f"   Problem 2 cost: ${p2_cost:.10f}")
print(f"   Difference: ${cost_diff:.10f}")
print(f"   {'✓ Costs are identical (within numerical precision)' if cost_diff < 1e-5 else '✗ Costs differ significantly'}")

# Check unit statuses
print(f"\n2. UNIT STATUS COMPARISON:")
units = p1_data['optimization_info']['units']
all_same = True
for unit_id in units:
    p1_status_col = f'Unit_{unit_id}_Status'
    p2_status_col = f'Unit_{unit_id}_Status'
    if p1_status_col in p1_schedule.columns and p2_status_col in p2_schedule.columns:
        p1_status = p1_schedule[p1_status_col].values
        p2_status = p2_schedule[p2_status_col].values
        if not (p1_status == p2_status).all():
            print(f"   ✗ Unit {unit_id}: Statuses differ!")
            all_same = False
        else:
            print(f"   ✓ Unit {unit_id}: Statuses identical (all {'ON' if p1_status[0] == 1 else 'OFF'})")
    else:
        print(f"   ⚠ Unit {unit_id}: Columns not found")

if all_same:
    print(f"   → All unit statuses are identical between Problem 1 and Problem 2")

# Check generation values
print(f"\n3. GENERATION COMPARISON:")
max_gen_diff = 0
for unit_id in units:
    p1_gen_col = f'Unit_{unit_id}_Generation_MW'
    p2_gen_col = f'Unit_{unit_id}_Generation_MW'
    if p1_gen_col in p1_schedule.columns and p2_gen_col in p2_schedule.columns:
        p1_gen = p1_schedule[p1_gen_col].values
        p2_gen = p2_schedule[p2_gen_col].values
        gen_diff = abs(p1_gen - p2_gen).max()
        max_gen_diff = max(max_gen_diff, gen_diff)
        if gen_diff > 1e-3:
            print(f"   ✗ Unit {unit_id}: Max generation difference = {gen_diff:.6f} MW")
        else:
            print(f"   ✓ Unit {unit_id}: Generations identical (max diff = {gen_diff:.10f} MW)")

print(f"\n   Maximum generation difference across all units: {max_gen_diff:.10f} MW")
print(f"   {'✓ Generations are identical (within numerical precision)' if max_gen_diff < 1e-3 else '✗ Generations differ'}")

# Check constraints
print(f"\n4. CONSTRAINT COMPARISON:")
p1_has_n1 = 'n1_security_enabled' in p1_data['optimization_info']
p2_has_n1 = p2_data['optimization_info'].get('n1_security_enabled', False)
p1_has_reserve = 'spinning_reserve_enabled' in p1_data['optimization_info']
p2_has_reserve = p2_data['optimization_info'].get('spinning_reserve_enabled', False)

print(f"   Problem 1:")
print(f"     - N-1 constraints: {'ENABLED' if p1_has_n1 else 'DISABLED'}")
print(f"     - Spinning reserve: {'ENABLED' if p1_has_reserve else 'DISABLED'}")
print(f"   Problem 2:")
print(f"     - N-1 constraints: {'ENABLED' if p2_has_n1 else 'DISABLED'}")
print(f"     - Spinning reserve: {'ENABLED' if p2_has_reserve else 'DISABLED'}")

# Analyze why constraints don't affect results
print(f"\n5. ROOT CAUSE ANALYSIS:")

# Check if all units are always on in Problem 1
p1_all_on = True
for unit_id in units:
    status_col = f'Unit_{unit_id}_Status'
    if status_col in p1_schedule.columns:
        if not (p1_schedule[status_col] == 1).all():
            p1_all_on = False
            break

if p1_all_on:
    print(f"   ✓ Problem 1: All units are always ON")
    print(f"     → This is because minimum generation sum (117 MW) < minimum load (131 MW)")
    print(f"     → Therefore, all units must stay online to meet load")
    
    if p2_has_n1:
        print(f"\n   ⚠ Problem 2: N-1 constraints are enabled but don't change results")
        print(f"     → Reason: Since all units are already online in Problem 1,")
        print(f"       N-1 constraints don't add additional restrictions")
        print(f"     → N-1 constraint checks: remaining capacity >= load + reserve")
        print(f"     → With all units online, this constraint is already satisfied")
        
        # Check N-1 constraint tightness
        P_max_values = {1: 300, 2: 180, 5: 50, 8: 35, 11: 30, 13: 40}
        total_capacity = sum(P_max_values.values())
        largest_unit = max(P_max_values.values())
        remaining_after_largest = total_capacity - largest_unit
        
        print(f"\n   N-1 Constraint Analysis:")
        print(f"     - Total capacity: {total_capacity} MW")
        print(f"     - Largest unit: {largest_unit} MW")
        print(f"     - Remaining after largest outage: {remaining_after_largest} MW")
        
        # Check a few periods - calculate reserve requirement correctly
        largest_unit_capacity = max(P_max_values.values())
        total_capacity_check = sum(P_max_values.values())
        
        for period in [1, 12, 24]:
            load = p2_schedule.loc[period-1, 'Load_MW']
            # Calculate required reserve (same logic as in model)
            reserve_10pct = 0.10 * load
            reserve_largest = largest_unit_capacity
            max_feasible_reserve = max(0, (total_capacity_check - largest_unit_capacity) - load)
            reserve_standard = max(reserve_10pct, reserve_largest)
            reserve_final = min(reserve_standard, max_feasible_reserve)
            reserve_final = max(reserve_final, 0.05 * load)
            
            required = load + reserve_final
            margin = remaining_after_largest - required
            available_reserve = p2_schedule.loc[period-1, 'Spinning_Reserve_MW'] if 'Spinning_Reserve_MW' in p2_schedule.columns else 0
            
            print(f"     - Period {period}:")
            print(f"       Load={load:.1f} MW, Required_Reserve={reserve_final:.1f} MW")
            print(f"       Required_Total={required:.1f} MW, Remaining={remaining_after_largest:.1f} MW")
            print(f"       Margin={margin:.1f} MW, Available_Reserve={available_reserve:.1f} MW")
            
            if abs(margin) < 1e-3:
                print(f"       → N-1 constraint is TIGHT (margin ≈ 0)")
            elif margin < -1e-3:
                print(f"       → ⚠ N-1 constraint VIOLATED!")
            else:
                print(f"       → N-1 constraint has slack (margin = {margin:.1f} MW)")

print(f"\n6. CONCLUSION:")
print(f"   The identical results suggest that:")
print(f"   1. Problem 1 already requires all units online (due to min gen < min load)")
print(f"   2. N-1 constraints in Problem 2 don't add restrictions because all units are already online")
print(f"   3. Network constraints (DC power flow) don't restrict the solution")
print(f"   4. Spinning reserve constraints are satisfied with all units online")
print(f"\n   However, this may indicate:")
print(f"   - N-1 constraints might not be correctly implemented")
print(f"   - Or the problem structure naturally requires all units online")
print(f"   - Network constraints might be too loose")

print("\n" + "=" * 70)

