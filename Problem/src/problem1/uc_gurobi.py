"""
Problem 1: Classical Unit Commitment Model using Gurobi
========================================================

This script implements a classical Unit Commitment optimization model with:
- Objective: Fuel cost (quadratic) + Startup/Shutdown costs
- Constraints: Power balance, generation limits, min up/down time, 
               ramp rates, startup/shutdown logic

Usage:
    python problem1_uc_gurobi.py [--min-up-time-table {1,2}]
    
    Options:
        --min-up-time-table: Select which table to use for Minimum Up Time
                             1 = Table 1 (Part A), 2 = Table 2 (Part B)
                             Default: 2 (Table 2 is authoritative per problem.md)

Note: According to problem.md, Table 2 is authoritative for Minimum Up Time.
      Table 1 values are for reference only. Default uses Table 2.

   # 使用 Table 2（默认，符合 problem.md 规则）
   python problem1_uc_gurobi.py
   
   # 使用 Table 1（仅用于参考）
   python problem1_uc_gurobi.py --min-up-time-table 1
   Parameters of each unit						
Unit (bus)	Maximum power generation (MW)	Minimum power generation (MW)	Minimum Up Time (h)	Startup Cost ($)	Shutdown Cost ($)	Ramp-Up Limit (MW/h)
1	300	50	8	180	180	80
2	180	20	8	180	180	80
5	50	15	5	40	40	50
8	35	10	5	60	60	50
11	30	10	6	60	60	60
13	40	12	3	40	40	60

"""

import argparse

# ============================================================================
# Configuration: Command Line Arguments (parse before importing Gurobi)
# ============================================================================

parser = argparse.ArgumentParser(
    description='Unit Commitment Problem Solver',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python problem1_uc_gurobi.py                    # Use Table 2 (default, authoritative)
  python problem1_uc_gurobi.py --min-up-time-table 2  # Use Table 2 (explicit)
  python problem1_uc_gurobi.py --min-up-time-table 1  # Use Table 1 (reference only)
    """
)
parser.add_argument(
    '--min-up-time-table',
    type=int,
    choices=[1, 2],
    default=2,
    help='Select which table to use for Minimum Up Time: 1 (Table 1) or 2 (Table 2). Default: 2 (Table 2 is authoritative per problem.md)'
)
args = parser.parse_args()

# Store the selected table number
MIN_UP_TIME_TABLE = args.min_up_time_table

# Now import required libraries (after parsing arguments)
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: Gurobi is not installed.")
    print("Please install Gurobi using one of the following methods:")
    print("1. pip install gurobipy")
    print("2. Download from https://www.gurobi.com/downloads/")
    print("   Note: Gurobi requires a license (free academic license available)")
    raise

import numpy as np
import json
import os
from datetime import datetime

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
    # Set style for better-looking plots
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    sns.set_palette("husl")
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    print(f"Warning: Visualization libraries not available ({e})")
    print("Data saving will still work, but visualizations will be skipped.")
    print("To enable visualizations, install: pip install pandas matplotlib seaborn")

# ============================================================================
# Data Extraction from Problem.md Tables
# ============================================================================

# Units: 1, 2, 5, 8, 11, 13
units = [1, 2, 5, 8, 11, 13]
num_units = len(units)
num_periods = 24

# Create unit index mapping
unit_to_idx = {unit: idx for idx, unit in enumerate(units)}

# Table 1: Parameters Part A
# Unit | P_max | P_min | Min_Up_Time_Table1 | Startup_Cost | Shutdown_Cost | Ramp_Up
table1_data = {
    1: {'P_max': 300, 'P_min': 50, 'Min_Up_Time_Table1': 8, 'Startup_Cost': 180, 'Shutdown_Cost': 180, 'Ramp_Up': 80},
    2: {'P_max': 180, 'P_min': 20, 'Min_Up_Time_Table1': 8, 'Startup_Cost': 180, 'Shutdown_Cost': 180, 'Ramp_Up': 80},
    5: {'P_max': 50, 'P_min': 15, 'Min_Up_Time_Table1': 5, 'Startup_Cost': 40, 'Shutdown_Cost': 40, 'Ramp_Up': 50},
    8: {'P_max': 35, 'P_min': 10, 'Min_Up_Time_Table1': 5, 'Startup_Cost': 60, 'Shutdown_Cost': 60, 'Ramp_Up': 50},
    11: {'P_max': 30, 'P_min': 10, 'Min_Up_Time_Table1': 6, 'Startup_Cost': 60, 'Shutdown_Cost': 60, 'Ramp_Up': 60},
    13: {'P_max': 40, 'P_min': 12, 'Min_Up_Time_Table1': 3, 'Startup_Cost': 40, 'Shutdown_Cost': 40, 'Ramp_Up': 60},
}

# Table 2: Parameters Part B
# Unit | Ramp_Down | Min_Up_Time | Min_Down_Time | Initial_Up_Time | Initial_Down_Time | a | b | c | H
table2_data = {
    1: {'Ramp_Down': 80, 'Min_Up_Time': 5, 'Min_Down_Time': 3, 'Initial_Up_Time': 5, 'Initial_Down_Time': 0, 
        'a': 0.02, 'b': 2.00, 'c': 0, 'H': 7.0},
    2: {'Ramp_Down': 80, 'Min_Up_Time': 4, 'Min_Down_Time': 2, 'Initial_Up_Time': 4, 'Initial_Down_Time': 0,
        'a': 0.0175, 'b': 1.75, 'c': 0, 'H': 4.5},
    5: {'Ramp_Down': 50, 'Min_Up_Time': 3, 'Min_Down_Time': 2, 'Initial_Up_Time': 3, 'Initial_Down_Time': 0,
        'a': 0.0625, 'b': 1.00, 'c': 0, 'H': 4.5},
    8: {'Ramp_Down': 50, 'Min_Up_Time': 3, 'Min_Down_Time': 2, 'Initial_Up_Time': 3, 'Initial_Down_Time': 0,
        'a': 0.00834, 'b': 3.25, 'c': 0, 'H': 3.2},
    11: {'Ramp_Down': 60, 'Min_Up_Time': 1, 'Min_Down_Time': 1, 'Initial_Up_Time': 3, 'Initial_Down_Time': 0,
        'a': 0.025, 'b': 3.00, 'c': 0, 'H': 3.0},
    13: {'Ramp_Down': 60, 'Min_Up_Time': 4, 'Min_Down_Time': 2, 'Initial_Up_Time': 4, 'Initial_Down_Time': 0,
        'a': 0.025, 'b': 3.00, 'c': 0, 'H': 3.0},
}

# Table 4: Load demands (24 periods)
load_demand = [166, 196, 229, 257, 263, 252, 246, 213, 192, 161, 147, 160,
               170, 185, 208, 232, 246, 241, 236, 225, 204, 182, 161, 131]

# Extract parameters into arrays indexed by unit index
P_max = np.array([table1_data[u]['P_max'] for u in units])
P_min = np.array([table1_data[u]['P_min'] for u in units])

# Select Minimum Up Time from Table 1 or Table 2 based on user choice
if MIN_UP_TIME_TABLE == 1:
    Min_Up_Time = np.array([table1_data[u]['Min_Up_Time_Table1'] for u in units])
    min_up_time_source = "Table 1 (Part A)"
else:
    Min_Up_Time = np.array([table2_data[u]['Min_Up_Time'] for u in units])
    min_up_time_source = "Table 2 (Part B)"

Min_Down_Time = np.array([table2_data[u]['Min_Down_Time'] for u in units])
Ramp_Up = np.array([table1_data[u]['Ramp_Up'] for u in units])
Ramp_Down = np.array([table2_data[u]['Ramp_Down'] for u in units])
Startup_Cost = np.array([table1_data[u]['Startup_Cost'] for u in units])
Shutdown_Cost = np.array([table1_data[u]['Shutdown_Cost'] for u in units])
a_coeff = np.array([table2_data[u]['a'] for u in units])
b_coeff = np.array([table2_data[u]['b'] for u in units])
c_coeff = np.array([table2_data[u]['c'] for u in units])
Initial_Up_Time = np.array([table2_data[u]['Initial_Up_Time'] for u in units])
Initial_Down_Time = np.array([table2_data[u]['Initial_Down_Time'] for u in units])

# Determine initial status: if Initial_Up_Time > 0, unit is ON; if Initial_Down_Time > 0, unit is OFF
initial_status = np.array([1 if Initial_Up_Time[i] > 0 else 0 for i in range(num_units)])

print("=" * 70)
print("Unit Commitment Problem - Data Summary")
print("=" * 70)
print(f"Number of units: {num_units}")
print(f"Units: {units}")
print(f"Number of time periods: {num_periods}")
print(f"\nMinimum Up Time Source: {min_up_time_source}")
print("\nUnit Parameters:")
for i, u in enumerate(units):
    # Show both Table 1 and Table 2 values for comparison
    table1_value = table1_data[u]['Min_Up_Time_Table1']
    table2_value = table2_data[u]['Min_Up_Time']
    min_up_display = f"{Min_Up_Time[i]}"
    if table1_value != table2_value:
        min_up_display += f" (Table1={table1_value}, Table2={table2_value})"
    
    print(f"  Unit {u}: P_max={P_max[i]}, P_min={P_min[i]}, "
          f"Min_Up={min_up_display}, Min_Down={Min_Down_Time[i]}, "
          f"Ramp_Up={Ramp_Up[i]}, Ramp_Down={Ramp_Down[i]}, "
          f"Initial_Status={'ON' if initial_status[i] else 'OFF'}")
print(f"\nTotal Load Range: {min(load_demand)} - {max(load_demand)} MW")
print("=" * 70)

# ============================================================================
# Gurobi Model Formulation
# ============================================================================

# Create model
model = gp.Model("UnitCommitment")
model.setParam('OutputFlag', 1)
model.setParam('MIPGap', 1e-4)  # 0.01% optimality gap

# Decision Variables
# u[i,t]: Binary variable for unit i commitment status at time t
u = model.addVars(num_units, num_periods, vtype=GRB.BINARY, name="u")

# p[i,t]: Continuous variable for power generation of unit i at time t
p = model.addVars(num_units, num_periods, lb=0, name="p")

# v[i,t]: Binary variable for startup indicator
v = model.addVars(num_units, num_periods, vtype=GRB.BINARY, name="v")

# w[i,t]: Binary variable for shutdown indicator
w = model.addVars(num_units, num_periods, vtype=GRB.BINARY, name="w")

# ============================================================================
# Objective Function
# ============================================================================
# Minimize: Σᵢ Σₜ [aᵢ·p[i,t]² + bᵢ·p[i,t] + cᵢ + StartupCostᵢ·v[i,t] + ShutdownCostᵢ·w[i,t]]

# Fuel cost (quadratic)
fuel_cost = gp.quicksum(a_coeff[i] * p[i, t] * p[i, t] + 
                         b_coeff[i] * p[i, t] + 
                         c_coeff[i]
                         for i in range(num_units) for t in range(num_periods))

# Startup and shutdown costs
startup_shutdown_cost = gp.quicksum(Startup_Cost[i] * v[i, t] + 
                                     Shutdown_Cost[i] * w[i, t]
                                     for i in range(num_units) for t in range(num_periods))

model.setObjective(fuel_cost + startup_shutdown_cost, GRB.MINIMIZE)

# ============================================================================
# Constraints
# ============================================================================

# 1. Power balance constraint: Σᵢ p[i,t] = Load[t] for all t
print("\nAdding power balance constraints...")
for t in range(num_periods):
    model.addConstr(gp.quicksum(p[i, t] for i in range(num_units)) == load_demand[t],
                    name=f"PowerBalance_t{t+1}")

# 2. Generation limits: P_min[i]·u[i,t] ≤ p[i,t] ≤ P_max[i]·u[i,t]
print("Adding generation limit constraints...")
for i in range(num_units):
    for t in range(num_periods):
        model.addConstr(p[i, t] >= P_min[i] * u[i, t], name=f"GenMin_u{i+1}_t{t+1}")
        model.addConstr(p[i, t] <= P_max[i] * u[i, t], name=f"GenMax_u{i+1}_t{t+1}")

# 3. Startup/shutdown logic constraints: v[i,t] - w[i,t] = u[i,t] - u[i,t-1]
print("Adding startup/shutdown logic constraints...")
for i in range(num_units):
    for t in range(num_periods):
        if t == 0:
            # First period: compare with initial status
            model.addConstr(v[i, t] - w[i, t] == u[i, t] - initial_status[i],
                            name=f"StartupShutdown_u{i+1}_t{t+1}")
        else:
            model.addConstr(v[i, t] - w[i, t] == u[i, t] - u[i, t-1],
                            name=f"StartupShutdown_u{i+1}_t{t+1}")

# 4. Minimum up-time constraints
print("Adding minimum up-time constraints...")
for i in range(num_units):
    if Min_Up_Time[i] > 0:
        # For periods at the beginning, consider initial status
        if initial_status[i] == 1:
            # Unit is initially ON
            # Must stay ON for remaining minimum up-time
            remaining_up_time = max(0, Min_Up_Time[i] - Initial_Up_Time[i])
            for t in range(min(remaining_up_time, num_periods)):
                model.addConstr(u[i, t] == 1, name=f"MinUp_Initial_u{i+1}_t{t+1}")
        
        # Standard minimum up-time constraint:
        # If unit started in any of the last Min_Up_Time periods, it must be ON at time t
        # Also account for initial ON status: if unit was initially ON and hasn't satisfied min up-time yet
        for t in range(num_periods):
            # Sum of startup indicators in the relevant time window
            startup_sum = gp.quicksum(v[i, t_start] 
                                      for t_start in range(max(0, t - Min_Up_Time[i] + 1), t + 1))
            
            # If unit was initially ON and period t is within the minimum up-time requirement
            if initial_status[i] == 1:
                # Check if we're still within the minimum up-time window from the start
                if t < Min_Up_Time[i]:
                    # Unit must stay ON (already enforced by initial constraints, but add for consistency)
                    # The constraint u[i,t] >= startup_sum still applies, but startup_sum = 0 for early periods
                    # So we rely on the initial constraints for these periods
                    pass
            
            model.addConstr(u[i, t] >= startup_sum, name=f"MinUp_u{i+1}_t{t+1}")

# 5. Minimum down-time constraints
print("Adding minimum down-time constraints...")
for i in range(num_units):
    if Min_Down_Time[i] > 0:
        # For periods at the beginning, consider initial status
        if initial_status[i] == 0:
            # Unit is initially OFF
            # Must stay OFF for remaining minimum down-time
            remaining_down_time = max(0, Min_Down_Time[i] - Initial_Down_Time[i])
            for t in range(min(remaining_down_time, num_periods)):
                model.addConstr(u[i, t] == 0, name=f"MinDown_Initial_u{i+1}_t{t+1}")
        
        # Standard minimum down-time constraint:
        # If unit shut down in any of the last Min_Down_Time periods, it must be OFF at time t
        # Formulation: u[i,t] <= 1 - sum(w[i, t-k+1] for k=1 to min(Min_Down_Time, t+1))
        for t in range(num_periods):
            # Sum of shutdown indicators in the relevant time window
            shutdown_sum = gp.quicksum(w[i, t_start] 
                                       for t_start in range(max(0, t - Min_Down_Time[i] + 1), t + 1))
            model.addConstr(u[i, t] <= 1 - shutdown_sum, name=f"MinDown_u{i+1}_t{t+1}")

# 6. Ramp-rate constraints
print("Adding ramp-rate constraints...")
for i in range(num_units):
    for t in range(num_periods):
        if t == 0:
            # First period: compare with initial power level
            # Initial power: if unit is ON initially, assume it's at P_min (conservative)
            initial_power = P_min[i] if initial_status[i] == 1 else 0
            # Ramp-up constraint
            model.addConstr(p[i, t] - initial_power <= Ramp_Up[i] * initial_status[i] + 
                            P_max[i] * (1 - initial_status[i]),
                            name=f"RampUp_u{i+1}_t{t+1}")
            # Ramp-down constraint
            model.addConstr(initial_power - p[i, t] <= Ramp_Down[i] * u[i, t] + 
                            P_max[i] * (1 - u[i, t]),
                            name=f"RampDown_u{i+1}_t{t+1}")
        else:
            # Ramp-up constraint: p[i,t] - p[i,t-1] ≤ RampUp[i]·u[i,t-1] + P_max[i]·(1-u[i,t-1])
            model.addConstr(p[i, t] - p[i, t-1] <= Ramp_Up[i] * u[i, t-1] + 
                            P_max[i] * (1 - u[i, t-1]),
                            name=f"RampUp_u{i+1}_t{t+1}")
            # Ramp-down constraint: p[i,t-1] - p[i,t] ≤ RampDown[i]·u[i,t] + P_max[i]·(1-u[i,t])
            model.addConstr(p[i, t-1] - p[i, t] <= Ramp_Down[i] * u[i, t] + 
                            P_max[i] * (1 - u[i, t]),
                            name=f"RampDown_u{i+1}_t{t+1}")

# Additional constraint: v[i,t] and w[i,t] cannot both be 1
print("Adding mutual exclusivity constraints for startup/shutdown...")
for i in range(num_units):
    for t in range(num_periods):
        model.addConstr(v[i, t] + w[i, t] <= 1, name=f"MutualExcl_u{i+1}_t{t+1}")

print("\nModel formulation complete!")
print(f"Total variables: {model.numVars}")
print(f"Total constraints: {model.numConstrs}")

# ============================================================================
# Solve the Model
# ============================================================================

print("\n" + "=" * 70)
print("Solving the Unit Commitment Problem...")
print("=" * 70)

model.optimize()

# ============================================================================
# Results Output
# ============================================================================

if model.status == GRB.OPTIMAL:
    print("\n" + "=" * 70)
    print("OPTIMAL SOLUTION FOUND")
    print("=" * 70)
    
    obj_value = model.ObjVal
    print(f"\nOptimal Total Cost: ${obj_value:.2f}")
    
    # Calculate cost breakdown
    fuel_cost_total = sum(a_coeff[i] * p[i, t].X * p[i, t].X + 
                          b_coeff[i] * p[i, t].X + 
                          c_coeff[i]
                          for i in range(num_units) for t in range(num_periods))
    
    startup_cost_total = sum(Startup_Cost[i] * v[i, t].X 
                              for i in range(num_units) for t in range(num_periods))
    
    shutdown_cost_total = sum(Shutdown_Cost[i] * w[i, t].X 
                               for i in range(num_units) for t in range(num_periods))
    
    print(f"\nCost Breakdown:")
    print(f"  Fuel Cost:        ${fuel_cost_total:.2f}")
    print(f"  Startup Cost:     ${startup_cost_total:.2f}")
    print(f"  Shutdown Cost:    ${shutdown_cost_total:.2f}")
    print(f"  Total Cost:       ${obj_value:.2f}")
    
    # Unit commitment schedule
    print("\n" + "=" * 70)
    print("Unit Commitment Schedule (1=ON, 0=OFF)")
    print("=" * 70)
    header = "Period | " + " | ".join([f"Unit {u:2d}" for u in units])
    print(header)
    print("-" * len(header))
    for t in range(num_periods):
        status_str = " | ".join([f"{int(u[i, t].X):5d}" for i in range(num_units)])
        print(f"  {t+1:2d}   | {status_str}")
    
    # Generation schedule
    print("\n" + "=" * 70)
    print("Generation Schedule (MW)")
    print("=" * 70)
    header = "Period | Load  | " + " | ".join([f"Unit {u:2d}" for u in units]) + " | Total"
    print(header)
    print("-" * len(header))
    for t in range(num_periods):
        gen_values = [p[i, t].X for i in range(num_units)]
        total_gen = sum(gen_values)
        gen_str = " | ".join([f"{val:6.2f}" for val in gen_values])
        print(f"  {t+1:2d}   | {load_demand[t]:5d} | {gen_str} | {total_gen:6.2f}")
    
    # Startup/Shutdown events
    print("\n" + "=" * 70)
    print("Startup/Shutdown Events")
    print("=" * 70)
    for i in range(num_units):
        startups = [t+1 for t in range(num_periods) if v[i, t].X > 0.5]
        shutdowns = [t+1 for t in range(num_periods) if w[i, t].X > 0.5]
        if startups or shutdowns:
            print(f"Unit {units[i]}:")
            if startups:
                print(f"  Startups at periods: {startups}")
            if shutdowns:
                print(f"  Shutdowns at periods: {shutdowns}")
    
    # Verify constraints
    print("\n" + "=" * 70)
    print("Constraint Verification")
    print("=" * 70)
    
    # Check power balance
    power_balance_ok = True
    for t in range(num_periods):
        total_gen = sum(p[i, t].X for i in range(num_units))
        if abs(total_gen - load_demand[t]) > 1e-3:
            print(f"WARNING: Power balance violation at period {t+1}: "
                  f"Generation={total_gen:.2f}, Load={load_demand[t]}")
            power_balance_ok = False
    if power_balance_ok:
        print("✓ Power balance constraints satisfied")
    
    # Check generation limits
    gen_limits_ok = True
    for i in range(num_units):
        for t in range(num_periods):
            gen_val = p[i, t].X
            u_val = u[i, t].X
            if u_val > 0.5:
                if gen_val < P_min[i] - 1e-3 or gen_val > P_max[i] + 1e-3:
                    print(f"WARNING: Generation limit violation for Unit {units[i]} at period {t+1}: "
                          f"Generation={gen_val:.2f}, Limits=[{P_min[i]}, {P_max[i]}]")
                    gen_limits_ok = False
            else:
                if gen_val > 1e-3:
                    print(f"WARNING: Unit {units[i]} is OFF but generating {gen_val:.2f} MW at period {t+1}")
                    gen_limits_ok = False
    if gen_limits_ok:
        print("✓ Generation limit constraints satisfied")
    
    print("\n" + "=" * 70)
    print("Solution Summary Complete")
    print("=" * 70)
    
    # ============================================================================
    # Data Saving
    # ============================================================================
    print("\n" + "=" * 70)
    print("Saving Results to Files...")
    print("=" * 70)
    
    # Create results directory (relative to project root)
    # Get project root directory (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(project_root, "results", "problem1")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save unit commitment schedule to CSV
    import csv
    uc_schedule_data = []
    for t in range(num_periods):
        row = {'Period': t + 1, 'Load_MW': load_demand[t]}
        for i, unit_id in enumerate(units):
            row[f'Unit_{unit_id}_Status'] = int(u[i, t].X)
            row[f'Unit_{unit_id}_Generation_MW'] = p[i, t].X
        row['Total_Generation_MW'] = sum(p[i, t].X for i in range(num_units))
        uc_schedule_data.append(row)
    
    uc_csv_path = os.path.join(results_dir, f"uc_schedule_{timestamp}.csv")
    if VISUALIZATION_AVAILABLE:
        uc_df = pd.DataFrame(uc_schedule_data)
        uc_df.to_csv(uc_csv_path, index=False)
    else:
        # Use standard csv module if pandas not available
        with open(uc_csv_path, 'w', newline='', encoding='utf-8') as f:
            if uc_schedule_data:
                writer = csv.DictWriter(f, fieldnames=uc_schedule_data[0].keys())
                writer.writeheader()
                writer.writerows(uc_schedule_data)
    print(f"✓ Unit commitment schedule saved to: {uc_csv_path}")
    
    # 2. Save generation schedule to CSV
    gen_schedule_data = []
    for t in range(num_periods):
        row = {'Period': t + 1, 'Load_MW': load_demand[t]}
        for i, unit_id in enumerate(units):
            row[f'Unit_{unit_id}_MW'] = p[i, t].X
        row['Total_Generation_MW'] = sum(p[i, t].X for i in range(num_units))
        gen_schedule_data.append(row)
    
    gen_csv_path = os.path.join(results_dir, f"generation_schedule_{timestamp}.csv")
    if VISUALIZATION_AVAILABLE:
        gen_df = pd.DataFrame(gen_schedule_data)
        gen_df.to_csv(gen_csv_path, index=False)
    else:
        # Use standard csv module if pandas not available
        with open(gen_csv_path, 'w', newline='', encoding='utf-8') as f:
            if gen_schedule_data:
                writer = csv.DictWriter(f, fieldnames=gen_schedule_data[0].keys())
                writer.writeheader()
                writer.writerows(gen_schedule_data)
    print(f"✓ Generation schedule saved to: {gen_csv_path}")
    
    # 3. Save summary statistics to JSON
    summary_data = {
        'optimization_info': {
            'timestamp': timestamp,
            'total_cost': float(obj_value),
            'fuel_cost': float(fuel_cost_total),
            'startup_cost': float(startup_cost_total),
            'shutdown_cost': float(shutdown_cost_total),
            'num_units': num_units,
            'num_periods': num_periods,
            'units': [int(u) for u in units],
            'min_up_time_table': MIN_UP_TIME_TABLE,
            'min_up_time_source': min_up_time_source
        },
        'unit_statistics': []
    }
    
    for i, unit_id in enumerate(units):
        unit_gen = [p[i, t].X for t in range(num_periods)]
        unit_status = [int(u[i, t].X) for t in range(num_periods)]
        startups = [t+1 for t in range(num_periods) if v[i, t].X > 0.5]
        shutdowns = [t+1 for t in range(num_periods) if w[i, t].X > 0.5]
        
        unit_stats = {
            'unit_id': int(unit_id),
            'total_generation_MWh': float(sum(unit_gen)),
            'average_generation_MW': float(np.mean(unit_gen)),
            'max_generation_MW': float(max(unit_gen)),
            'min_generation_MW': float(min(unit_gen)),
            'on_periods': int(sum(unit_status)),
            'off_periods': num_periods - int(sum(unit_status)),
            'utilization_rate': float(sum(unit_status) / num_periods),
            'startup_events': startups,
            'shutdown_events': shutdowns,
            'total_startup_cost': float(sum(Startup_Cost[i] * v[i, t].X for t in range(num_periods))),
            'total_shutdown_cost': float(sum(Shutdown_Cost[i] * w[i, t].X for t in range(num_periods)))
        }
        summary_data['unit_statistics'].append(unit_stats)
    
    summary_json_path = os.path.join(results_dir, f"summary_{timestamp}.json")
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Summary statistics saved to: {summary_json_path}")
    
    # ============================================================================
    # Data Visualization (至少3种图表)
    # ============================================================================
    if not VISUALIZATION_AVAILABLE:
        print("\n" + "=" * 70)
        print("Visualization skipped (libraries not available)")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("Generating Visualizations...")
        print("=" * 70)
    
    # Prepare data for visualization
    periods = np.arange(1, num_periods + 1)
    unit_generation = np.array([[p[i, t].X for t in range(num_periods)] for i in range(num_units)])
    unit_status_matrix = np.array([[u[i, t].X for t in range(num_periods)] for i in range(num_units)])
    
    # Visualization 1: 机组组合状态热力图 (Unit Commitment Status Heatmap)
    print("Creating Visualization 1: Unit Commitment Status Heatmap...")
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    im = ax1.imshow(unit_status_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax1.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Unit', fontsize=12, fontweight='bold')
    ax1.set_title('Unit Commitment Status Heatmap\n(Green=ON, Red=OFF)', fontsize=14, fontweight='bold')
    ax1.set_yticks(range(num_units))
    ax1.set_yticklabels([f'Unit {u}' for u in units])
    ax1.set_xticks(range(0, num_periods, 2))
    ax1.set_xticklabels(range(1, num_periods + 1, 2))
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Status (1=ON, 0=OFF)', rotation=270, labelpad=20)
    plt.tight_layout()
    heatmap_path = os.path.join(results_dir, f"1_unit_commitment_heatmap_{timestamp}.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {heatmap_path}")
    plt.close()
    
    # Visualization 2: 发电量随时间变化曲线 (Generation vs Time)
    print("Creating Visualization 2: Generation Schedule Over Time...")
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, num_units))
    for i, unit_id in enumerate(units):
        ax2.plot(periods, unit_generation[i], marker='o', linewidth=2, 
                markersize=4, label=f'Unit {unit_id}', color=colors[i])
    ax2.plot(periods, load_demand, 'k--', linewidth=2.5, label='Load Demand', alpha=0.7)
    ax2.fill_between(periods, load_demand, alpha=0.2, color='gray', label='Load Area')
    ax2.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Power Generation (MW)', fontsize=12, fontweight='bold')
    ax2.set_title('Generation Schedule and Load Demand Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', ncol=3, fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, num_periods + 0.5)
    plt.tight_layout()
    gen_curve_path = os.path.join(results_dir, f"2_generation_schedule_{timestamp}.png")
    plt.savefig(gen_curve_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {gen_curve_path}")
    plt.close()
    
    # Visualization 3: 成本分解饼图 (Cost Breakdown Pie Chart)
    print("Creating Visualization 3: Cost Breakdown Pie Chart...")
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart for cost breakdown
    cost_labels = ['Fuel Cost', 'Startup Cost', 'Shutdown Cost']
    cost_values = [fuel_cost_total, startup_cost_total, shutdown_cost_total]
    colors_pie = ['#ff9999', '#66b3ff', '#99ff99']
    # Only show non-zero costs
    non_zero_data = [(label, val) for label, val in zip(cost_labels, cost_values) if val > 1e-6]
    if non_zero_data:
        labels_pie, values_pie = zip(*non_zero_data)
        ax3a.pie(values_pie, labels=labels_pie, autopct='%1.2f%%', startangle=90,
                colors=colors_pie[:len(values_pie)], textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax3a.set_title('Total Cost Breakdown', fontsize=13, fontweight='bold', pad=20)
    else:
        ax3a.text(0.5, 0.5, 'No costs to display', ha='center', va='center', 
                 transform=ax3a.transAxes, fontsize=12)
        ax3a.set_title('Total Cost Breakdown', fontsize=13, fontweight='bold')
    
    # Bar chart for unit-wise fuel costs
    unit_fuel_costs = []
    unit_labels_bar = []
    for i, unit_id in enumerate(units):
        unit_cost = sum(a_coeff[i] * p[i, t].X * p[i, t].X + 
                       b_coeff[i] * p[i, t].X + c_coeff[i]
                       for t in range(num_periods))
        unit_fuel_costs.append(unit_cost)
        unit_labels_bar.append(f'Unit {unit_id}')
    
    bars = ax3b.bar(unit_labels_bar, unit_fuel_costs, color=colors[:num_units], alpha=0.7, edgecolor='black')
    ax3b.set_xlabel('Unit', fontsize=12, fontweight='bold')
    ax3b.set_ylabel('Fuel Cost ($)', fontsize=12, fontweight='bold')
    ax3b.set_title('Fuel Cost by Unit', fontsize=13, fontweight='bold')
    ax3b.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3b.text(bar.get_x() + bar.get_width()/2., height,
                 f'${height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    cost_pie_path = os.path.join(results_dir, f"3_cost_breakdown_{timestamp}.png")
    plt.savefig(cost_pie_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {cost_pie_path}")
    plt.close()
    
    # Visualization 4: 各机组出力分布堆叠面积图 (Stacked Area Chart)
    print("Creating Visualization 4: Stacked Generation Area Chart...")
    fig4, ax4 = plt.subplots(figsize=(14, 7))
    ax4.stackplot(periods, unit_generation, labels=[f'Unit {u}' for u in units],
                  colors=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.plot(periods, load_demand, 'k-', linewidth=3, label='Load Demand', marker='o', markersize=5)
    ax4.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Power Generation (MW)', fontsize=12, fontweight='bold')
    ax4.set_title('Stacked Generation Schedule vs Load Demand', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=10, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.5, num_periods + 0.5)
    plt.tight_layout()
    stacked_area_path = os.path.join(results_dir, f"4_stacked_generation_{timestamp}.png")
    plt.savefig(stacked_area_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {stacked_area_path}")
    plt.close()
    
    # Visualization 5: 机组利用率统计 (Unit Utilization Statistics)
    print("Creating Visualization 5: Unit Utilization Statistics...")
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Utilization rate (on periods / total periods)
    utilization_rates = [sum(unit_status_matrix[i]) / num_periods * 100 for i in range(num_units)]
    bars1 = ax5a.bar([f'Unit {u}' for u in units], utilization_rates, 
                     color=colors[:num_units], alpha=0.7, edgecolor='black')
    ax5a.set_ylabel('Utilization Rate (%)', fontsize=12, fontweight='bold')
    ax5a.set_xlabel('Unit', fontsize=12, fontweight='bold')
    ax5a.set_title('Unit Utilization Rate', fontsize=13, fontweight='bold')
    ax5a.set_ylim(0, 105)
    ax5a.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        height = bar.get_height()
        ax5a.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Total generation by unit
    total_gen_by_unit = [sum(unit_generation[i]) for i in range(num_units)]
    bars2 = ax5b.bar([f'Unit {u}' for u in units], total_gen_by_unit,
                     color=colors[:num_units], alpha=0.7, edgecolor='black')
    ax5b.set_ylabel('Total Generation (MWh)', fontsize=12, fontweight='bold')
    ax5b.set_xlabel('Unit', fontsize=12, fontweight='bold')
    ax5b.set_title('Total Generation by Unit (24 hours)', fontsize=13, fontweight='bold')
    ax5b.grid(True, alpha=0.3, axis='y')
    for bar in bars2:
        height = bar.get_height()
        ax5b.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    utilization_path = os.path.join(results_dir, f"5_unit_utilization_{timestamp}.png")
    plt.savefig(utilization_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {utilization_path}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("All basic visualizations saved successfully!")
    print(f"Results directory: {results_dir}")
    print("=" * 70)
    
    # ============================================================================
    # Advanced Visualizations (3D and High-Quality Plots)
    # ============================================================================
    try:
        from .advanced_visualizations import create_advanced_visualizations
        print("\n" + "=" * 70)
        print("Generating Advanced Visualizations (3D plots)...")
        print("=" * 70)
        
        create_advanced_visualizations(
            unit_generation=unit_generation,
            unit_status_matrix=unit_status_matrix,
            load_demand=np.array(load_demand),
            fuel_cost_total=fuel_cost_total,
            startup_cost_total=startup_cost_total,
            shutdown_cost_total=shutdown_cost_total,
            a_coeff=a_coeff,
            b_coeff=b_coeff,
            c_coeff=c_coeff,
            units=units,
            periods=periods,
            results_dir=results_dir,
            timestamp=timestamp
        )
    except ImportError:
        # Try absolute import if relative import fails
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from advanced_visualizations import create_advanced_visualizations
            print("\n" + "=" * 70)
            print("Generating Advanced Visualizations (3D plots)...")
            print("=" * 70)
            
            create_advanced_visualizations(
                unit_generation=unit_generation,
                unit_status_matrix=unit_status_matrix,
                load_demand=np.array(load_demand),
                fuel_cost_total=fuel_cost_total,
                startup_cost_total=startup_cost_total,
                shutdown_cost_total=shutdown_cost_total,
                a_coeff=a_coeff,
                b_coeff=b_coeff,
                c_coeff=c_coeff,
                units=units,
                periods=periods,
                results_dir=results_dir,
                timestamp=timestamp
            )
        except Exception as e:
            print(f"\nWarning: Could not generate advanced visualizations: {e}")
            print("Basic visualizations are still available.")
    
    # ============================================================================
    # Theoretical/Principle Visualizations (2D Principle Diagrams)
    # ============================================================================
    try:
        from .theoretical_visualizations import create_theoretical_visualizations
        print("\n" + "=" * 70)
        print("Generating Theoretical/Principle Visualizations (2D diagrams)...")
        print("=" * 70)
        
        create_theoretical_visualizations(
            a_coeff=a_coeff,
            b_coeff=b_coeff,
            c_coeff=c_coeff,
            P_min=P_min,
            P_max=P_max,
            Ramp_Up=Ramp_Up,
            Ramp_Down=Ramp_Down,
            Min_Up_Time=Min_Up_Time,
            Min_Down_Time=Min_Down_Time,
            Startup_Cost=Startup_Cost,
            Shutdown_Cost=Shutdown_Cost,
            units=units,
            unit_generation=unit_generation,
            unit_status_matrix=unit_status_matrix,
            load_demand=np.array(load_demand),
            results_dir=results_dir,
            timestamp=timestamp
        )
    except ImportError:
        # Try absolute import if relative import fails
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from theoretical_visualizations import create_theoretical_visualizations
            print("\n" + "=" * 70)
            print("Generating Theoretical/Principle Visualizations (2D diagrams)...")
            print("=" * 70)
            
            create_theoretical_visualizations(
                a_coeff=a_coeff,
                b_coeff=b_coeff,
                c_coeff=c_coeff,
                P_min=P_min,
                P_max=P_max,
                Ramp_Up=Ramp_Up,
                Ramp_Down=Ramp_Down,
                Min_Up_Time=Min_Up_Time,
                Min_Down_Time=Min_Down_Time,
                Startup_Cost=Startup_Cost,
                Shutdown_Cost=Shutdown_Cost,
                units=units,
                unit_generation=unit_generation,
                unit_status_matrix=unit_status_matrix,
                load_demand=np.array(load_demand),
                results_dir=results_dir,
                timestamp=timestamp
            )
        except Exception as e:
            print(f"\nWarning: Could not generate theoretical visualizations: {e}")
            print("Other visualizations are still available.")
    
else:
    print(f"\nOptimization failed. Status: {model.status}")
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        model.computeIIS()
        model.write("model_iis.ilp")
        print("IIS written to model_iis.ilp")

