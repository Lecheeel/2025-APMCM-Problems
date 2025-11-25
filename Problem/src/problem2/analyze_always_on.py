"""
Problem 2: Analysis of "Always On" Units
=========================================

This script analyzes why all units remain online and checks if this is optimal.
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
results_dir = os.path.dirname(summary_files[0])

with open(summary_files[0], 'r', encoding='utf-8') as f:
    summary = json.load(f)

uc_schedule = pd.read_csv(glob.glob(os.path.join(results_dir, "uc_schedule_*.csv"))[0])

print("=" * 70)
print("ANALYSIS: Why All Units Stay Online")
print("=" * 70)

# Parameters
units = summary['optimization_info']['units']
P_max_values = {1: 300, 2: 180, 5: 50, 8: 35, 11: 30, 13: 40}
P_min_values = {1: 50, 2: 20, 5: 15, 8: 10, 11: 10, 13: 12}
Startup_Cost = {1: 180, 2: 180, 5: 40, 8: 60, 11: 60, 13: 40}

total_capacity = sum(P_max_values.values())
largest_unit = max(P_max_values.values())
remaining_after_largest = total_capacity - largest_unit

print(f"\nSystem Capacity Analysis:")
print(f"  Total capacity: {total_capacity} MW")
print(f"  Largest unit: Unit {max(P_max_values, key=P_max_values.get)} = {largest_unit} MW")
print(f"  Remaining after largest outage: {remaining_after_largest} MW")

print(f"\nN-1 Constraint Analysis:")
print(f"  N-1 requires: After losing largest unit, remaining capacity >= Load + Reserve")
print(f"  Maximum remaining capacity: {remaining_after_largest} MW")

# Check each period
print(f"\nPeriod-by-Period Analysis:")
print(f"  {'Period':<8} {'Load':<8} {'Req_Reserve':<12} {'Required':<10} {'Remaining':<10} {'Margin':<10} {'Can Shut Down?'}")
print(f"  {'-'*75}")

can_shutdown_periods = []
cannot_shutdown_periods = []

# Calculate required reserve for each period (same logic as in model)
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

for period in range(1, 25):
    load = uc_schedule.loc[period-1, 'Load_MW']
    required_reserve = required_reserves[period-1]
    available_reserve = uc_schedule.loc[period-1, 'Spinning_Reserve_MW']
    required = load + required_reserve
    margin = remaining_after_largest - required
    
    # Check if we can shut down any unit
    # If we shut down the smallest unit (30 MW), remaining = 635 - 30 = 605 MW
    smallest_unit = min(P_max_values.values())
    remaining_after_smallest = total_capacity - smallest_unit
    
    can_shutdown = remaining_after_smallest >= required
    
    if can_shutdown:
        can_shutdown_periods.append(period)
    else:
        cannot_shutdown_periods.append(period)
    
    status = "YES" if can_shutdown else "NO"
    print(f"  {period:<8} {load:<8.1f} {required_reserve:<10.1f} {required:<10.1f} "
          f"{remaining_after_largest:<10.1f} {margin:<10.1f} {status}")

print(f"\nSummary:")
print(f"  Periods where smallest unit CAN be shut down: {len(can_shutdown_periods)}")
print(f"  Periods where NO unit can be shut down: {len(cannot_shutdown_periods)}")

if cannot_shutdown_periods:
    print(f"\n⚠ N-1 constraint prevents shutting down ANY unit in periods: {cannot_shutdown_periods}")
    print(f"  Reason: Even shutting down the smallest unit would violate N-1 constraint")

# Check if it's economically optimal
print(f"\nEconomic Analysis:")
print(f"  Startup costs:")
for unit_id in units:
    print(f"    Unit {unit_id}: ${Startup_Cost[unit_id]}")

# Calculate potential savings if we could shut down units
print(f"\nPotential Cost Analysis:")
print(f"  If we shut down Unit 11 (smallest, 30 MW) during low-load periods:")
print(f"    Startup cost: ${Startup_Cost[11]} per startup")
print(f"    Shutdown cost: $60 per shutdown")

# Check minimum load periods
min_load_period = uc_schedule['Load_MW'].idxmin() + 1
min_load = uc_schedule['Load_MW'].min()
print(f"\n  Minimum load period: {min_load_period} (Load = {min_load:.1f} MW)")
required_reserve_min = required_reserves[min_load_period-1]
print(f"  Required reserve: {required_reserve_min:.1f} MW")
print(f"  Required capacity (Load + Required Reserve): {uc_schedule.loc[min_load_period-1, 'Load_MW'] + required_reserve_min:.1f} MW")
print(f"  Remaining after shutting smallest unit: {remaining_after_smallest} MW")

if remaining_after_smallest >= uc_schedule.loc[min_load_period-1, 'Load_MW'] + required_reserve_min:
    print(f"  ✓ Could potentially shut down smallest unit in period {min_load_period}")
    print(f"  ⚠ But solver chose to keep all units online - may be due to:")
    print(f"    - Minimum up-time constraints")
    print(f"    - Ramp rate constraints")
    print(f"    - Or startup cost > savings from shutting down")
else:
    print(f"  ✗ Cannot shut down even smallest unit (N-1 constraint)")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("The 'always on' behavior is likely due to:")
print("1. N-1 security constraint requiring sufficient capacity after largest unit outage")
print("2. Spinning reserve requirements")
print("3. The combination of these constraints may require all units online")
print("\nTo verify if this is truly optimal, consider:")
print("- Running without N-1 constraints (using --relax-n1 flag)")
print("- Comparing costs with Problem 1")
print("- Checking if startup costs justify keeping units online")
print("=" * 70)

