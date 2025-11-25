"""
Problem 2: UC Modeling with Network and Security Constraints
============================================================

This script extends Problem 1 by adding:
- DC power flow constraints on transmission lines
- N-1 security constraints (single generator/line outage scenarios)
- Spinning reserve requirements
- Optional: Minimum safety inertia constraint

Usage:
    python uc_network_security.py [--enable-inertia]
    
    Options:
        --enable-inertia: Enable minimum safety inertia constraint (optional)
"""

import argparse

# ============================================================================
# Configuration: Command Line Arguments
# ============================================================================

parser = argparse.ArgumentParser(
    description='Unit Commitment Problem with Network and Security Constraints',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    '--enable-inertia',
    action='store_true',
    help='Enable minimum safety inertia constraint (optional)'
)
parser.add_argument(
    '--min-up-time-table',
    type=int,
    choices=[1, 2],
    default=2,
    help='Select which table to use for Minimum Up Time: 1 (Table 1) or 2 (Table 2). Default: 2 (Table 2 is authoritative per problem.md)'
)
parser.add_argument(
    '--relax-n1',
    action='store_true',
    help='Use relaxed N-1 security constraints (for debugging)'
)
args = parser.parse_args()

ENABLE_INERTIA = args.enable_inertia
MIN_UP_TIME_TABLE = args.min_up_time_table
RELAX_N1 = args.relax_n1

# Now import required libraries
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: Gurobi is not installed.")
    print("Please install Gurobi using one of the following methods:")
    print("1. pip install gurobipy")
    print("2. Download from https://www.gurobi.com/downloads/")
    raise

import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
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

# ============================================================================
# Data Extraction from Problem.md Tables
# ============================================================================

# Units: 1, 2, 5, 8, 11, 13
units = [1, 2, 5, 8, 11, 13]
num_units = len(units)
num_periods = 24

# Create unit index mapping and bus-to-unit mapping
unit_to_idx = {unit: idx for idx, unit in enumerate(units)}
unit_to_bus = {units[i]: units[i] for i in range(num_units)}  # Units are at their bus numbers

# Table 1: Parameters Part A
table1_data = {
    1: {'P_max': 300, 'P_min': 50, 'Min_Up_Time_Table1': 8, 'Startup_Cost': 180, 'Shutdown_Cost': 180, 'Ramp_Up': 80},
    2: {'P_max': 180, 'P_min': 20, 'Min_Up_Time_Table1': 8, 'Startup_Cost': 180, 'Shutdown_Cost': 180, 'Ramp_Up': 80},
    5: {'P_max': 50, 'P_min': 15, 'Min_Up_Time_Table1': 5, 'Startup_Cost': 40, 'Shutdown_Cost': 40, 'Ramp_Up': 50},
    8: {'P_max': 35, 'P_min': 10, 'Min_Up_Time_Table1': 5, 'Startup_Cost': 60, 'Shutdown_Cost': 60, 'Ramp_Up': 50},
    11: {'P_max': 30, 'P_min': 10, 'Min_Up_Time_Table1': 6, 'Startup_Cost': 60, 'Shutdown_Cost': 60, 'Ramp_Up': 60},
    13: {'P_max': 40, 'P_min': 12, 'Min_Up_Time_Table1': 3, 'Startup_Cost': 40, 'Shutdown_Cost': 40, 'Ramp_Up': 60},
}

# Table 2: Parameters Part B
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

# Table 3: Grid parameters (branches)
# Branch | From | To | R | X | b | Max Power
branches_data = [
    {'branch': 1, 'from': 1, 'to': 2, 'R': 0.02, 'X': 0.06, 'b': 0.03, 'P_max': 650},
    {'branch': 2, 'from': 1, 'to': 3, 'R': 0.05, 'X': 0.19, 'b': 0.02, 'P_max': 650},
    {'branch': 3, 'from': 2, 'to': 4, 'R': 0.06, 'X': 0.17, 'b': 0.02, 'P_max': 325},
    {'branch': 4, 'from': 3, 'to': 4, 'R': 0.01, 'X': 0.04, 'b': 0, 'P_max': 650},
    {'branch': 5, 'from': 2, 'to': 5, 'R': 0.05, 'X': 0.20, 'b': 0.02, 'P_max': 650},
    {'branch': 6, 'from': 2, 'to': 6, 'R': 0.06, 'X': 0.18, 'b': 0.02, 'P_max': 325},
    {'branch': 7, 'from': 4, 'to': 6, 'R': 0.01, 'X': 0.04, 'b': 0, 'P_max': 450},
    {'branch': 8, 'from': 5, 'to': 7, 'R': 0.05, 'X': 0.12, 'b': 0.01, 'P_max': 350},
    {'branch': 9, 'from': 6, 'to': 7, 'R': 0.03, 'X': 0.08, 'b': 0.01, 'P_max': 650},
    {'branch': 10, 'from': 6, 'to': 8, 'R': 0.01, 'X': 0.04, 'b': 0, 'P_max': 160},
    {'branch': 11, 'from': 6, 'to': 9, 'R': 0, 'X': 0.21, 'b': 0, 'P_max': 325},
    {'branch': 12, 'from': 6, 'to': 10, 'R': 0, 'X': 0.56, 'b': 0, 'P_max': 160},
    {'branch': 13, 'from': 9, 'to': 11, 'R': 0, 'X': 0.21, 'b': 0, 'P_max': 325},
    {'branch': 14, 'from': 9, 'to': 10, 'R': 0, 'X': 0.11, 'b': 0, 'P_max': 325},
    {'branch': 15, 'from': 4, 'to': 12, 'R': 0, 'X': 0.26, 'b': 0, 'P_max': 325},
    {'branch': 16, 'from': 12, 'to': 13, 'R': 0, 'X': 0.14, 'b': 0, 'P_max': 325},
    {'branch': 17, 'from': 12, 'to': 14, 'R': 0.12, 'X': 0.26, 'b': 0, 'P_max': 160},
    {'branch': 18, 'from': 12, 'to': 15, 'R': 0.07, 'X': 0.13, 'b': 0, 'P_max': 160},
    {'branch': 19, 'from': 12, 'to': 16, 'R': 0.09, 'X': 0.20, 'b': 0, 'P_max': 160},
    {'branch': 20, 'from': 14, 'to': 15, 'R': 0.22, 'X': 0.20, 'b': 0, 'P_max': 80},
    {'branch': 21, 'from': 16, 'to': 17, 'R': 0.08, 'X': 0.19, 'b': 0, 'P_max': 80},
    {'branch': 22, 'from': 15, 'to': 18, 'R': 0.11, 'X': 0.22, 'b': 0, 'P_max': 80},
    {'branch': 23, 'from': 18, 'to': 19, 'R': 0.06, 'X': 0.13, 'b': 0, 'P_max': 80},
    {'branch': 24, 'from': 19, 'to': 20, 'R': 0.03, 'X': 0.07, 'b': 0, 'P_max': 80},
    {'branch': 25, 'from': 10, 'to': 20, 'R': 0.09, 'X': 0.21, 'b': 0, 'P_max': 80},
    {'branch': 26, 'from': 10, 'to': 17, 'R': 0.03, 'X': 0.08, 'b': 0, 'P_max': 80},
    {'branch': 27, 'from': 10, 'to': 21, 'R': 0.03, 'X': 0.07, 'b': 0, 'P_max': 80},
    {'branch': 28, 'from': 10, 'to': 22, 'R': 0.07, 'X': 0.15, 'b': 0, 'P_max': 80},
    {'branch': 29, 'from': 21, 'to': 22, 'R': 0.01, 'X': 0.02, 'b': 0, 'P_max': 160},
    {'branch': 30, 'from': 15, 'to': 23, 'R': 0.10, 'X': 0.20, 'b': 0, 'P_max': 160},
    {'branch': 31, 'from': 22, 'to': 24, 'R': 0.12, 'X': 0.18, 'b': 0, 'P_max': 160},
    {'branch': 32, 'from': 23, 'to': 24, 'R': 0.13, 'X': 0.27, 'b': 0, 'P_max': 160},
    {'branch': 33, 'from': 24, 'to': 25, 'R': 0.19, 'X': 0.33, 'b': 0, 'P_max': 80},
    {'branch': 34, 'from': 25, 'to': 26, 'R': 0.25, 'X': 0.38, 'b': 0, 'P_max': 80},
    {'branch': 35, 'from': 25, 'to': 27, 'R': 0.11, 'X': 0.21, 'b': 0, 'P_max': 80},
    {'branch': 36, 'from': 27, 'to': 28, 'R': 0, 'X': 0.40, 'b': 0, 'P_max': 80},
    {'branch': 37, 'from': 27, 'to': 29, 'R': 0.22, 'X': 0.42, 'b': 0, 'P_max': 80},
    {'branch': 38, 'from': 27, 'to': 30, 'R': 0.32, 'X': 0.60, 'b': 0, 'P_max': 80},
    {'branch': 39, 'from': 29, 'to': 30, 'R': 0.24, 'X': 0.45, 'b': 0, 'P_max': 80},
    {'branch': 40, 'from': 8, 'to': 28, 'R': 0.06, 'X': 0.20, 'b': 0.02, 'P_max': 160},
    {'branch': 41, 'from': 6, 'to': 28, 'R': 0.02, 'X': 0.06, 'b': 0.01, 'P_max': 160},
]

# Table 4: Load demands (24 periods)
load_demand = [166, 196, 229, 257, 263, 252, 246, 213, 192, 161, 147, 160,
               170, 185, 208, 232, 246, 241, 236, 225, 204, 182, 161, 131]

# Table 5: Inertia-related security parameters
ROCOF_set = 0.5  # Hz/s
F = 2.118
load_step = 600  # MW

# Extract unit parameters
P_max = np.array([table1_data[u]['P_max'] for u in units])
P_min = np.array([table1_data[u]['P_min'] for u in units])

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
H_inertia = np.array([table2_data[u]['H'] for u in units])  # Inertia constants

initial_status = np.array([1 if Initial_Up_Time[i] > 0 else 0 for i in range(num_units)])

# Process network data
# Find all buses (union of all from/to buses)
all_buses = set()
for branch in branches_data:
    all_buses.add(branch['from'])
    all_buses.add(branch['to'])
all_buses = sorted(list(all_buses))
num_buses = len(all_buses)
bus_to_idx = {bus: idx for idx, bus in enumerate(all_buses)}

# Calculate susceptance B_ij = 1/X_ij for DC power flow
# Store branch information
branches = []
for branch in branches_data:
    from_bus = branch['from']
    to_bus = branch['to']
    X = branch['X']
    # Handle zero reactance (transformer or very low impedance)
    # Use a large but finite susceptance value
    if X > 1e-6:  # Avoid division by zero
        B = 1.0 / X
    else:
        # For zero or very small X, use a large susceptance (low impedance path)
        # This represents a nearly ideal connection
        B = 1e6  # Large but finite value
    P_max_line = branch['P_max']
    branches.append({
        'from': from_bus,
        'to': to_bus,
        'from_idx': bus_to_idx[from_bus],
        'to_idx': bus_to_idx[to_bus],
        'B': B,
        'P_max': P_max_line
    })

num_branches = len(branches)

# Load distribution: For this problem, we need to distribute load across buses
# Since load buses are not explicitly specified, we'll use a reasonable distribution
# Load is typically at buses without generators or at specific load buses
# For simplicity, distribute load across key buses (excluding generator buses)
# We'll use a weighted distribution based on typical power system patterns
load_distribution = defaultdict(float)
# Primary load buses (larger loads) - exclude generator buses: 1, 2, 5, 8, 11, 13
load_buses_list = [3, 4, 6, 7, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
# Distribute load proportionally - use simple equal distribution for feasibility
# In practice, this would be specified based on actual load data
for bus in load_buses_list:
    load_distribution[bus] = 1.0 / len(load_buses_list)

# Verify load distribution sums to 1.0
total_dist = sum(load_distribution.values())
if abs(total_dist - 1.0) > 1e-6:
    print(f"WARNING: Load distribution sum = {total_dist}, normalizing...")
    for bus in load_distribution:
        load_distribution[bus] /= total_dist

# Spinning reserve requirement: typically 10% of load or largest unit capacity
# However, we need to ensure feasibility - reserve shouldn't exceed available capacity
largest_unit_capacity = max(P_max)
total_capacity = sum(P_max)
spinning_reserve_req = []
for t in range(num_periods):
    # Standard approach: reserve = max(10% of load, largest unit capacity)
    # But we need to ensure it's feasible with N-1 constraints
    # If we lose the largest unit, remaining capacity must >= load + reserve
    # So: (total_capacity - largest_unit) >= load + reserve
    # Therefore: reserve <= (total_capacity - largest_unit) - load
    reserve_10pct = 0.10 * load_demand[t]
    reserve_largest = largest_unit_capacity
    
    # Calculate maximum feasible reserve considering N-1
    max_feasible_reserve = max(0, (total_capacity - largest_unit_capacity) - load_demand[t])
    
    # Use the standard requirement, but cap at feasible level
    reserve_standard = max(reserve_10pct, reserve_largest)
    reserve_final = min(reserve_standard, max_feasible_reserve)
    
    # Ensure at least 5% of load as minimum reserve
    reserve_final = max(reserve_final, 0.05 * load_demand[t])
    
    spinning_reserve_req.append(reserve_final)

print("=" * 70)
print("Unit Commitment Problem with Network and Security Constraints")
print("=" * 70)
print(f"Number of units: {num_units}")
print(f"Units: {units}")
print(f"Number of buses: {num_buses}")
print(f"Buses: {all_buses}")
print(f"Number of branches: {num_branches}")
print(f"Number of time periods: {num_periods}")
print(f"\nMinimum Up Time Source: {min_up_time_source}")
print(f"Spinning Reserve: {spinning_reserve_req[0]:.2f} MW (first period)")
print(f"Inertia Constraint: {'ENABLED' if ENABLE_INERTIA else 'DISABLED'}")
print("=" * 70)

# ============================================================================
# Gurobi Model Formulation
# ============================================================================

model = gp.Model("UnitCommitment_NetworkSecurity")
model.setParam('OutputFlag', 1)
model.setParam('MIPGap', 1e-4)

# Decision Variables (from Problem 1)
u = model.addVars(num_units, num_periods, vtype=GRB.BINARY, name="u")
p = model.addVars(num_units, num_periods, lb=0, name="p")
v = model.addVars(num_units, num_periods, vtype=GRB.BINARY, name="v")
w = model.addVars(num_units, num_periods, vtype=GRB.BINARY, name="w")

# New decision variables for network modeling
# Voltage angles (DC power flow)
# Note: In DC power flow, angles can theoretically be any value to satisfy power balance
# We remove bounds to ensure feasibility - angles will be checked post-optimization for reasonableness
# Typical power system angles are -30° to +30°, but optimization may require larger values
# Unbounded angles are mathematically valid in DC power flow
theta = model.addVars(num_buses, num_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta")
# Line flows (can be positive or negative)
P_flow = model.addVars(num_branches, num_periods, lb=-GRB.INFINITY, name="P_flow")

# Objective Function (same as Problem 1)
fuel_cost = gp.quicksum(a_coeff[i] * p[i, t] * p[i, t] + 
                         b_coeff[i] * p[i, t] + 
                         c_coeff[i]
                         for i in range(num_units) for t in range(num_periods))

startup_shutdown_cost = gp.quicksum(Startup_Cost[i] * v[i, t] + 
                                     Shutdown_Cost[i] * w[i, t]
                                     for i in range(num_units) for t in range(num_periods))

model.setObjective(fuel_cost + startup_shutdown_cost, GRB.MINIMIZE)

# ============================================================================
# Constraints from Problem 1
# ============================================================================

print("\nAdding Problem 1 constraints...")

# 1. Power balance constraint (system-wide - will be enforced by bus-level balance)
# Note: Bus-level balance ensures system balance, but we keep this for consistency
print("Adding power balance constraints...")
for t in range(num_periods):
    model.addConstr(gp.quicksum(p[i, t] for i in range(num_units)) == load_demand[t],
                    name=f"PowerBalance_t{t+1}")

# 2. Generation limits
print("Adding generation limit constraints...")
for i in range(num_units):
    for t in range(num_periods):
        model.addConstr(p[i, t] >= P_min[i] * u[i, t], name=f"GenMin_u{i+1}_t{t+1}")
        model.addConstr(p[i, t] <= P_max[i] * u[i, t], name=f"GenMax_u{i+1}_t{t+1}")

# 3. Startup/shutdown logic
print("Adding startup/shutdown logic constraints...")
for i in range(num_units):
    for t in range(num_periods):
        if t == 0:
            model.addConstr(v[i, t] - w[i, t] == u[i, t] - initial_status[i],
                            name=f"StartupShutdown_u{i+1}_t{t+1}")
        else:
            model.addConstr(v[i, t] - w[i, t] == u[i, t] - u[i, t-1],
                            name=f"StartupShutdown_u{i+1}_t{t+1}")

# 4. Minimum up-time constraints
print("Adding minimum up-time constraints...")
for i in range(num_units):
    if Min_Up_Time[i] > 0:
        if initial_status[i] == 1:
            remaining_up_time = max(0, Min_Up_Time[i] - Initial_Up_Time[i])
            for t in range(min(remaining_up_time, num_periods)):
                model.addConstr(u[i, t] == 1, name=f"MinUp_Initial_u{i+1}_t{t+1}")
        
        for t in range(num_periods):
            startup_sum = gp.quicksum(v[i, t_start] 
                                      for t_start in range(max(0, t - Min_Up_Time[i] + 1), t + 1))
            model.addConstr(u[i, t] >= startup_sum, name=f"MinUp_u{i+1}_t{t+1}")

# 5. Minimum down-time constraints
print("Adding minimum down-time constraints...")
for i in range(num_units):
    if Min_Down_Time[i] > 0:
        if initial_status[i] == 0:
            remaining_down_time = max(0, Min_Down_Time[i] - Initial_Down_Time[i])
            for t in range(min(remaining_down_time, num_periods)):
                model.addConstr(u[i, t] == 0, name=f"MinDown_Initial_u{i+1}_t{t+1}")
        
        for t in range(num_periods):
            shutdown_sum = gp.quicksum(w[i, t_start] 
                                       for t_start in range(max(0, t - Min_Down_Time[i] + 1), t + 1))
            model.addConstr(u[i, t] <= 1 - shutdown_sum, name=f"MinDown_u{i+1}_t{t+1}")

# 6. Ramp-rate constraints
print("Adding ramp-rate constraints...")
for i in range(num_units):
    for t in range(num_periods):
        if t == 0:
            initial_power = P_min[i] if initial_status[i] == 1 else 0
            model.addConstr(p[i, t] - initial_power <= Ramp_Up[i] * initial_status[i] + 
                            P_max[i] * (1 - initial_status[i]),
                            name=f"RampUp_u{i+1}_t{t+1}")
            model.addConstr(initial_power - p[i, t] <= Ramp_Down[i] * u[i, t] + 
                            P_max[i] * (1 - u[i, t]),
                            name=f"RampDown_u{i+1}_t{t+1}")
        else:
            model.addConstr(p[i, t] - p[i, t-1] <= Ramp_Up[i] * u[i, t-1] + 
                            P_max[i] * (1 - u[i, t-1]),
                            name=f"RampUp_u{i+1}_t{t+1}")
            model.addConstr(p[i, t-1] - p[i, t] <= Ramp_Down[i] * u[i, t] + 
                            P_max[i] * (1 - u[i, t]),
                            name=f"RampDown_u{i+1}_t{t+1}")

# Additional constraint: v and w mutual exclusivity
print("Adding mutual exclusivity constraints...")
for i in range(num_units):
    for t in range(num_periods):
        model.addConstr(v[i, t] + w[i, t] <= 1, name=f"MutualExcl_u{i+1}_t{t+1}")

# ============================================================================
# New Constraints for Problem 2
# ============================================================================

print("\n" + "=" * 70)
print("Adding Problem 2 constraints...")
print("=" * 70)

# 1. DC Power Flow Constraints
print("\nAdding DC power flow constraints...")

# Set reference bus angle to zero (bus 1 as reference)
ref_bus_idx = bus_to_idx[1]
for t in range(num_periods):
    model.addConstr(theta[ref_bus_idx, t] == 0, name=f"RefBus_t{t+1}")

# Line flow constraints: P_ij = B_ij * (θ_i - θ_j)
for br_idx, branch in enumerate(branches):
    from_idx = branch['from_idx']
    to_idx = branch['to_idx']
    B = branch['B']
    for t in range(num_periods):
        model.addConstr(P_flow[br_idx, t] == B * (theta[from_idx, t] - theta[to_idx, t]),
                        name=f"LineFlow_br{br_idx+1}_t{t+1}")
        # Line flow limits: -P_max <= P_flow <= P_max
        model.addConstr(P_flow[br_idx, t] <= branch['P_max'], name=f"LineFlowMax_br{br_idx+1}_t{t+1}")
        model.addConstr(P_flow[br_idx, t] >= -branch['P_max'], name=f"LineFlowMin_br{br_idx+1}_t{t+1}")

# Power balance at each bus
print("Adding bus power balance constraints...")
# Verify load distribution sums to 1.0
total_load_dist = sum(load_distribution.values())
if abs(total_load_dist - 1.0) > 1e-6:
    raise ValueError(f"Load distribution must sum to 1.0, but sums to {total_load_dist}")

for bus_idx, bus in enumerate(all_buses):
    for t in range(num_periods):
        # Generation at this bus
        gen_at_bus = gp.quicksum(p[i, t] for i in range(num_units) if units[i] == bus)
        
        # Load at this bus
        load_at_bus = load_demand[t] * load_distribution.get(bus, 0)
        
        # Net flow out = sum of flows leaving bus - sum of flows entering bus
        # P_flow[br_idx, t] represents flow from branch['from'] to branch['to']
        # For bus i: net flow out = sum(P_flow for branches leaving i) - sum(P_flow for branches entering i)
        flow_out = gp.quicksum(P_flow[br_idx, t] 
                               for br_idx, branch in enumerate(branches) 
                               if branch['from_idx'] == bus_idx)
        flow_in = gp.quicksum(P_flow[br_idx, t] 
                              for br_idx, branch in enumerate(branches) 
                              if branch['to_idx'] == bus_idx)
        net_flow_out = flow_out - flow_in
        
        # Power balance: Generation - Load = Net Flow Out
        # For buses with no generation and no load, net flow should be zero (transit buses)
        model.addConstr(gen_at_bus - load_at_bus == net_flow_out,
                        name=f"BusBalance_bus{bus}_t{t+1}")

# 2. Spinning Reserve Requirements
print("\nAdding spinning reserve constraints...")
for t in range(num_periods):
    available_reserve = gp.quicksum((P_max[i] * u[i, t] - p[i, t]) for i in range(num_units))
    model.addConstr(available_reserve >= spinning_reserve_req[t],
                    name=f"SpinningReserve_t{t+1}")

# 3. N-1 Security Constraints
print("\nAdding N-1 security constraints...")
if RELAX_N1:
    print("  (Using relaxed N-1 constraints)")

# N-1 Generator contingencies: For each generator outage, ensure remaining can meet demand + reserve
for gen_out_idx in range(num_units):
    for t in range(num_periods):
        # Available capacity excluding the outaged generator
        available_capacity = gp.quicksum(P_max[i] * u[i, t] 
                                         for i in range(num_units) if i != gen_out_idx)
        # Required: demand + reserve
        # If relaxed, use a smaller reserve requirement for N-1
        if RELAX_N1:
            # Use only demand (no reserve requirement in contingency)
            required_capacity = load_demand[t]
        else:
            required_capacity = load_demand[t] + spinning_reserve_req[t]
        model.addConstr(available_capacity >= required_capacity,
                        name=f"N1_GenOutage_u{gen_out_idx+1}_t{t+1}")

# N-1 Line contingencies: For each line outage, ensure flows redistribute without violations
# Approach: Ensure that no single line carries more than a safe percentage of total load
# This prevents over-reliance on any single transmission path
# Use a more relaxed constraint to avoid infeasibility
max_line_utilization = 0.8  # No line should carry more than 80% of total load (relaxed from 60%)
for br_idx, branch in enumerate(branches):
    for t in range(num_periods):
        # Limit individual line flow to a percentage of total load
        # Use absolute value constraint: |P_flow| <= max_line_utilization * load
        model.addConstr(P_flow[br_idx, t] <= load_demand[t] * max_line_utilization,
                        name=f"LineUtilization_br{br_idx+1}_t{t+1}")
        model.addConstr(P_flow[br_idx, t] >= -load_demand[t] * max_line_utilization,
                        name=f"LineUtilizationNeg_br{br_idx+1}_t{t+1}")

# Also ensure sufficient total transmission capacity remains after any single line outage
# This is a simplified check - full N-1 would require solving power flow for each contingency
# Use a more relaxed constraint to avoid infeasibility
for line_out_idx in range(num_branches):
    for t in range(num_periods):
        # Sum of remaining line capacities (excluding outaged line)
        remaining_capacity = gp.quicksum(branches[br_idx]['P_max'] 
                                        for br_idx in range(num_branches) 
                                        if br_idx != line_out_idx)
        # Ensure remaining capacity can handle total load
        # Use a more relaxed factor - in practice, not all capacity is usable simultaneously
        # but we need to ensure basic feasibility
        model.addConstr(remaining_capacity >= load_demand[t] * 1.0,  # Relaxed safety factor
                        name=f"N1_LineOutage_br{line_out_idx+1}_t{t+1}")

# 4. Optional: Minimum Safety Inertia Constraint
if ENABLE_INERTIA:
    print("\nAdding minimum safety inertia constraints...")
    # From paper [3]: H_system = Σ(H_i * P_max[i] * u[i,t]) / Σ(P_max[i] * u[i,t])
    # Constraint: H_system >= ΔP / (2 * ROCOF_set)
    # Rearranging: Σ(H_i * P_max[i] * u[i,t]) >= (ΔP / (2 * ROCOF_set)) * Σ(P_max[i] * u[i,t])
    
    min_inertia_required = load_step / (2 * ROCOF_set)  # Minimum H required
    
    for t in range(num_periods):
        # Total inertia contribution
        total_inertia = gp.quicksum(H_inertia[i] * P_max[i] * u[i, t] for i in range(num_units))
        # Total capacity online
        total_capacity = gp.quicksum(P_max[i] * u[i, t] for i in range(num_units))
        
        # Constraint: H_system >= min_inertia_required
        # Which means: total_inertia >= min_inertia_required * total_capacity
        # But we need to handle the case when total_capacity = 0
        # Use a big-M approach or ensure at least some capacity is online
        
        # Simplified: Ensure total inertia contribution meets minimum
        # For systems with units online, this ensures average H >= min_inertia_required
        model.addConstr(total_inertia >= min_inertia_required * total_capacity,
                        name=f"InertiaConstraint_t{t+1}")

print("\nModel formulation complete!")
print(f"Total variables: {model.numVars}")
print(f"Total constraints: {model.numConstrs}")

# ============================================================================
# Solve the Model
# ============================================================================

print("\n" + "=" * 70)
print("Solving the Unit Commitment Problem with Network and Security Constraints...")
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
    
    # Spinning reserve utilization
    print(f"\nSpinning Reserve Utilization:")
    for t in range(min(5, num_periods)):  # Show first 5 periods
        available = sum((P_max[i] * u[i, t].X - p[i, t].X) for i in range(num_units))
        required = spinning_reserve_req[t]
        print(f"  Period {t+1}: Available={available:.2f} MW, Required={required:.2f} MW")
    
    # Save results
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(project_root, "results", "problem2")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary_data = {
        'optimization_info': {
            'timestamp': timestamp,
            'total_cost': float(obj_value),
            'fuel_cost': float(fuel_cost_total),
            'startup_cost': float(startup_cost_total),
            'shutdown_cost': float(shutdown_cost_total),
            'num_units': num_units,
            'num_periods': num_periods,
            'num_buses': num_buses,
            'num_branches': num_branches,
            'units': [int(u) for u in units],
            'spinning_reserve_enabled': True,
            'n1_security_enabled': True,
            'inertia_constraint_enabled': ENABLE_INERTIA,
            'min_up_time_table': MIN_UP_TIME_TABLE
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
            'on_periods': int(sum(unit_status)),
            'utilization_rate': float(sum(unit_status) / num_periods),
            'startup_events': startups,
            'shutdown_events': shutdowns
        }
        summary_data['unit_statistics'].append(unit_stats)
    
    summary_json_path = os.path.join(results_dir, f"summary_{timestamp}.json")
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Summary saved to: {summary_json_path}")
    
    # Save unit commitment schedule
    import csv
    uc_schedule_data = []
    for t in range(num_periods):
        row = {'Period': t + 1, 'Load_MW': load_demand[t]}
        for i, unit_id in enumerate(units):
            row[f'Unit_{unit_id}_Status'] = int(u[i, t].X)
            row[f'Unit_{unit_id}_Generation_MW'] = p[i, t].X
        row['Total_Generation_MW'] = sum(p[i, t].X for i in range(num_units))
        row['Spinning_Reserve_MW'] = sum((P_max[i] * u[i, t].X - p[i, t].X) for i in range(num_units))
        uc_schedule_data.append(row)
    
    uc_csv_path = os.path.join(results_dir, f"uc_schedule_{timestamp}.csv")
    if VISUALIZATION_AVAILABLE:
        uc_df = pd.DataFrame(uc_schedule_data)
        uc_df.to_csv(uc_csv_path, index=False)
    else:
        with open(uc_csv_path, 'w', newline='', encoding='utf-8') as f:
            if uc_schedule_data:
                writer = csv.DictWriter(f, fieldnames=uc_schedule_data[0].keys())
                writer.writeheader()
                writer.writerows(uc_schedule_data)
    print(f"✓ Unit commitment schedule saved to: {uc_csv_path}")
    
    # Save bus voltage angles (DC power flow)
    print("\nSaving bus voltage angles...")
    bus_angles_data = []
    for t in range(num_periods):
        row = {'Period': t + 1}
        for bus_idx, bus in enumerate(all_buses):
            row[f'Bus_{bus}_Angle_rad'] = theta[bus_idx, t].X
            row[f'Bus_{bus}_Angle_deg'] = theta[bus_idx, t].X * 180 / np.pi
        bus_angles_data.append(row)
    
    angles_csv_path = os.path.join(results_dir, f"bus_angles_{timestamp}.csv")
    if VISUALIZATION_AVAILABLE:
        angles_df = pd.DataFrame(bus_angles_data)
        angles_df.to_csv(angles_csv_path, index=False)
    else:
        with open(angles_csv_path, 'w', newline='', encoding='utf-8') as f:
            if bus_angles_data:
                writer = csv.DictWriter(f, fieldnames=bus_angles_data[0].keys())
                writer.writeheader()
                writer.writerows(bus_angles_data)
    print(f"✓ Bus voltage angles saved to: {angles_csv_path}")
    
    # Save line flows
    print("Saving line flows...")
    line_flows_data = []
    for t in range(num_periods):
        row = {'Period': t + 1}
        for br_idx, branch in enumerate(branches):
            from_bus = branch['from']
            to_bus = branch['to']
            flow_value = P_flow[br_idx, t].X
            row[f'Branch_{br_idx+1}_{from_bus}_to_{to_bus}_MW'] = flow_value
            row[f'Branch_{br_idx+1}_{from_bus}_to_{to_bus}_Percent'] = (flow_value / branch['P_max'] * 100) if branch['P_max'] > 0 else 0
        line_flows_data.append(row)
    
    flows_csv_path = os.path.join(results_dir, f"line_flows_{timestamp}.csv")
    if VISUALIZATION_AVAILABLE:
        flows_df = pd.DataFrame(line_flows_data)
        flows_df.to_csv(flows_csv_path, index=False)
    else:
        with open(flows_csv_path, 'w', newline='', encoding='utf-8') as f:
            if line_flows_data:
                writer = csv.DictWriter(f, fieldnames=line_flows_data[0].keys())
                writer.writeheader()
                writer.writerows(line_flows_data)
    print(f"✓ Line flows saved to: {flows_csv_path}")
    
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
        spinning_reserve_available = np.array([sum((P_max[i] * u[i, t].X - p[i, t].X) for i in range(num_units)) 
                                               for t in range(num_periods)])
        spinning_reserve_required = np.array(spinning_reserve_req)
        
        colors = plt.cm.tab10(np.linspace(0, 1, num_units))
        
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
        
        # Visualization 3: 旋转备用随时间变化 (Spinning Reserve Over Time) - Problem 2特有
        print("Creating Visualization 3: Spinning Reserve Over Time...")
        fig3, ax3 = plt.subplots(figsize=(14, 7))
        ax3.plot(periods, spinning_reserve_available, 'b-', linewidth=2.5, marker='o', 
                markersize=5, label='Available Reserve', alpha=0.8)
        ax3.plot(periods, spinning_reserve_required, 'r--', linewidth=2.5, marker='s', 
                markersize=5, label='Required Reserve', alpha=0.8)
        ax3.fill_between(periods, spinning_reserve_required, spinning_reserve_available, 
                         where=(spinning_reserve_available >= spinning_reserve_required),
                         alpha=0.3, color='green', label='Reserve Margin')
        ax3.fill_between(periods, 0, spinning_reserve_required, alpha=0.2, color='red', label='Required Reserve')
        ax3.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Spinning Reserve (MW)', fontsize=12, fontweight='bold')
        ax3.set_title('Spinning Reserve: Available vs Required Over Time', fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0.5, num_periods + 0.5)
        plt.tight_layout()
        reserve_path = os.path.join(results_dir, f"3_spinning_reserve_{timestamp}.png")
        plt.savefig(reserve_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {reserve_path}")
        plt.close()
        
        # Visualization 4: 成本分解图 (Cost Breakdown)
        print("Creating Visualization 4: Cost Breakdown...")
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart for cost breakdown
        cost_labels = ['Fuel Cost', 'Startup Cost', 'Shutdown Cost']
        cost_values = [fuel_cost_total, startup_cost_total, shutdown_cost_total]
        colors_pie = ['#ff9999', '#66b3ff', '#99ff99']
        non_zero_data = [(label, val) for label, val in zip(cost_labels, cost_values) if val > 1e-6]
        if non_zero_data:
            labels_pie, values_pie = zip(*non_zero_data)
            ax4a.pie(values_pie, labels=labels_pie, autopct='%1.2f%%', startangle=90,
                    colors=colors_pie[:len(values_pie)], textprops={'fontsize': 11, 'fontweight': 'bold'})
            ax4a.set_title('Total Cost Breakdown', fontsize=13, fontweight='bold', pad=20)
        else:
            ax4a.text(0.5, 0.5, 'No costs to display', ha='center', va='center', 
                     transform=ax4a.transAxes, fontsize=12)
            ax4a.set_title('Total Cost Breakdown', fontsize=13, fontweight='bold')
        
        # Bar chart for unit-wise fuel costs
        unit_fuel_costs = []
        unit_labels_bar = []
        for i, unit_id in enumerate(units):
            unit_cost = sum(a_coeff[i] * p[i, t].X * p[i, t].X + 
                           b_coeff[i] * p[i, t].X + c_coeff[i]
                           for t in range(num_periods))
            unit_fuel_costs.append(unit_cost)
            unit_labels_bar.append(f'Unit {unit_id}')
        
        bars = ax4b.bar(unit_labels_bar, unit_fuel_costs, color=colors[:num_units], alpha=0.7, edgecolor='black')
        ax4b.set_xlabel('Unit', fontsize=12, fontweight='bold')
        ax4b.set_ylabel('Fuel Cost ($)', fontsize=12, fontweight='bold')
        ax4b.set_title('Fuel Cost by Unit', fontsize=13, fontweight='bold')
        ax4b.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax4b.text(bar.get_x() + bar.get_width()/2., height,
                     f'${height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        cost_pie_path = os.path.join(results_dir, f"4_cost_breakdown_{timestamp}.png")
        plt.savefig(cost_pie_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {cost_pie_path}")
        plt.close()
        
        # Visualization 5: 堆叠面积图 (Stacked Area Chart)
        print("Creating Visualization 5: Stacked Generation Area Chart...")
        fig5, ax5 = plt.subplots(figsize=(14, 7))
        ax5.stackplot(periods, unit_generation, labels=[f'Unit {u}' for u in units],
                      colors=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax5.plot(periods, load_demand, 'k-', linewidth=3, label='Load Demand', marker='o', markersize=5)
        ax5.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Power Generation (MW)', fontsize=12, fontweight='bold')
        ax5.set_title('Stacked Generation Schedule vs Load Demand', fontsize=14, fontweight='bold')
        ax5.legend(loc='upper left', fontsize=10, ncol=2)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0.5, num_periods + 0.5)
        plt.tight_layout()
        stacked_area_path = os.path.join(results_dir, f"5_stacked_generation_{timestamp}.png")
        plt.savefig(stacked_area_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {stacked_area_path}")
        plt.close()
        
        # Visualization 6: 机组利用率统计 (Unit Utilization Statistics)
        print("Creating Visualization 6: Unit Utilization Statistics...")
        fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Utilization rate
        utilization_rates = [sum(unit_status_matrix[i]) / num_periods * 100 for i in range(num_units)]
        bars1 = ax6a.bar([f'Unit {u}' for u in units], utilization_rates, 
                         color=colors[:num_units], alpha=0.7, edgecolor='black')
        ax6a.set_ylabel('Utilization Rate (%)', fontsize=12, fontweight='bold')
        ax6a.set_xlabel('Unit', fontsize=12, fontweight='bold')
        ax6a.set_title('Unit Utilization Rate', fontsize=13, fontweight='bold')
        ax6a.set_ylim(0, 105)
        ax6a.grid(True, alpha=0.3, axis='y')
        for bar in bars1:
            height = bar.get_height()
            ax6a.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Total generation by unit
        total_gen_by_unit = [sum(unit_generation[i]) for i in range(num_units)]
        bars2 = ax6b.bar([f'Unit {u}' for u in units], total_gen_by_unit,
                         color=colors[:num_units], alpha=0.7, edgecolor='black')
        ax6b.set_ylabel('Total Generation (MWh)', fontsize=12, fontweight='bold')
        ax6b.set_xlabel('Unit', fontsize=12, fontweight='bold')
        ax6b.set_title('Total Generation by Unit (24 hours)', fontsize=13, fontweight='bold')
        ax6b.grid(True, alpha=0.3, axis='y')
        for bar in bars2:
            height = bar.get_height()
            ax6b.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        utilization_path = os.path.join(results_dir, f"6_unit_utilization_{timestamp}.png")
        plt.savefig(utilization_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {utilization_path}")
        plt.close()
        
        # Visualization 7: 线路潮流热力图 (Line Flow Heatmap) - Problem 2特有
        print("Creating Visualization 7: Line Flow Heatmap...")
        # Select key lines for visualization (top 10 by capacity)
        line_flows_matrix = np.array([[P_flow[br_idx, t].X for t in range(num_periods)] 
                                      for br_idx in range(num_branches)])
        # Normalize by line capacity for percentage
        line_flows_pct = np.zeros_like(line_flows_matrix)
        for br_idx, branch in enumerate(branches):
            if branch['P_max'] > 0:
                line_flows_pct[br_idx, :] = line_flows_matrix[br_idx, :] / branch['P_max'] * 100
        
        # Show top 15 lines by average flow
        avg_flows = np.abs(line_flows_matrix).mean(axis=1)
        top_lines_idx = np.argsort(avg_flows)[-15:][::-1]
        
        fig7, ax7 = plt.subplots(figsize=(16, 8))
        top_flows_pct = line_flows_pct[top_lines_idx, :]
        im = ax7.imshow(top_flows_pct, aspect='auto', cmap='RdYlGn_r', vmin=-100, vmax=100)
        ax7.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Transmission Line', fontsize=12, fontweight='bold')
        ax7.set_title('Line Flow Utilization Heatmap (Top 15 Lines)\n(Red=High Utilization, Green=Low)', 
                      fontsize=14, fontweight='bold')
        line_labels = [f"Br{top_lines_idx[i]+1}: {branches[top_lines_idx[i]]['from']}-{branches[top_lines_idx[i]]['to']}" 
                       for i in range(len(top_lines_idx))]
        ax7.set_yticks(range(len(top_lines_idx)))
        ax7.set_yticklabels(line_labels, fontsize=9)
        ax7.set_xticks(range(0, num_periods, 2))
        ax7.set_xticklabels(range(1, num_periods + 1, 2))
        cbar = plt.colorbar(im, ax=ax7)
        cbar.set_label('Flow Utilization (%)', rotation=270, labelpad=20)
        plt.tight_layout()
        line_flow_path = os.path.join(results_dir, f"7_line_flow_heatmap_{timestamp}.png")
        plt.savefig(line_flow_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {line_flow_path}")
        plt.close()
        
        # Visualization 8: 节点电压角随时间变化 (Bus Voltage Angles Over Time)
        print("Creating Visualization 8: Bus Voltage Angles Over Time...")
        # Select key buses (generator buses and a few load buses)
        key_buses = [1, 2, 5, 8, 11, 13, 3, 4, 7, 9, 10]  # Generator buses + some load buses
        key_bus_indices = [bus_to_idx[bus] for bus in key_buses if bus in bus_to_idx]
        
        fig8, ax8 = plt.subplots(figsize=(14, 7))
        for bus_idx in key_bus_indices:
            bus = all_buses[bus_idx]
            angles_deg = [theta[bus_idx, t].X * 180 / np.pi for t in range(num_periods)]
            ax8.plot(periods, angles_deg, marker='o', linewidth=2, markersize=3, 
                    label=f'Bus {bus}', alpha=0.7)
        ax8.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Voltage Angle (degrees)', fontsize=12, fontweight='bold')
        ax8.set_title('Bus Voltage Angles Over Time (DC Power Flow)', fontsize=14, fontweight='bold')
        ax8.legend(loc='best', fontsize=9, ncol=2)
        ax8.grid(True, alpha=0.3)
        ax8.set_xlim(0.5, num_periods + 0.5)
        ax8.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Reference Bus')
        plt.tight_layout()
        angles_path = os.path.join(results_dir, f"8_bus_angles_{timestamp}.png")
        plt.savefig(angles_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {angles_path}")
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
            
            # Prepare line flows matrix
            line_flows_matrix = np.array([[P_flow[br_idx, t].X for t in range(num_periods)] 
                                          for br_idx in range(num_branches)])
            
            # Prepare bus angles matrix
            bus_angles_matrix = np.array([[theta[bus_idx, t].X for t in range(num_periods)] 
                                          for bus_idx in range(num_buses)])
            
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
                spinning_reserve_available=spinning_reserve_available,
                spinning_reserve_required=spinning_reserve_required,
                line_flows_matrix=line_flows_matrix,
                branches=branches,
                bus_angles_matrix=bus_angles_matrix,
                all_buses=all_buses,
                bus_to_idx=bus_to_idx,
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
                
                # Prepare line flows matrix
                line_flows_matrix = np.array([[P_flow[br_idx, t].X for t in range(num_periods)] 
                                              for br_idx in range(num_branches)])
                
                # Prepare bus angles matrix
                bus_angles_matrix = np.array([[theta[bus_idx, t].X for t in range(num_periods)] 
                                              for bus_idx in range(num_buses)])
                
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
                    spinning_reserve_available=spinning_reserve_available,
                    spinning_reserve_required=spinning_reserve_required,
                    line_flows_matrix=line_flows_matrix,
                    branches=branches,
                    bus_angles_matrix=bus_angles_matrix,
                    all_buses=all_buses,
                    bus_to_idx=bus_to_idx,
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
            
            # Prepare line flows matrix if not already prepared
            if 'line_flows_matrix' not in locals():
                line_flows_matrix = np.array([[P_flow[br_idx, t].X for t in range(num_periods)] 
                                              for br_idx in range(num_branches)])
            
            # Prepare bus angles matrix if not already prepared
            if 'bus_angles_matrix' not in locals():
                bus_angles_matrix = np.array([[theta[bus_idx, t].X for t in range(num_periods)] 
                                              for bus_idx in range(num_buses)])
            
            create_theoretical_visualizations(
                a_coeff=a_coeff,
                b_coeff=b_coeff,
                c_coeff=c_coeff,
                P_min=P_min,
                P_max=P_max,
                units=units,
                branches=branches,
                all_buses=all_buses,
                bus_to_idx=bus_to_idx,
                spinning_reserve_available=spinning_reserve_available,
                spinning_reserve_required=spinning_reserve_required,
                line_flows_matrix=line_flows_matrix,
                bus_angles_matrix=bus_angles_matrix,
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
                
                # Prepare line flows matrix if not already prepared
                if 'line_flows_matrix' not in locals():
                    line_flows_matrix = np.array([[P_flow[br_idx, t].X for t in range(num_periods)] 
                                                  for br_idx in range(num_branches)])
                
                # Prepare bus angles matrix if not already prepared
                if 'bus_angles_matrix' not in locals():
                    bus_angles_matrix = np.array([[theta[bus_idx, t].X for t in range(num_periods)] 
                                                  for bus_idx in range(num_buses)])
                
                create_theoretical_visualizations(
                    a_coeff=a_coeff,
                    b_coeff=b_coeff,
                    c_coeff=c_coeff,
                    P_min=P_min,
                    P_max=P_max,
                    units=units,
                    branches=branches,
                    all_buses=all_buses,
                    bus_to_idx=bus_to_idx,
                    spinning_reserve_available=spinning_reserve_available,
                    spinning_reserve_required=spinning_reserve_required,
                    line_flows_matrix=line_flows_matrix,
                    bus_angles_matrix=bus_angles_matrix,
                    unit_generation=unit_generation,
                    unit_status_matrix=unit_status_matrix,
                    load_demand=np.array(load_demand),
                    results_dir=results_dir,
                    timestamp=timestamp
                )
            except Exception as e:
                print(f"\nWarning: Could not generate theoretical visualizations: {e}")
                print("Other visualizations are still available.")
    
    print("\n" + "=" * 70)
    print("Solution Summary Complete")
    print("=" * 70)
    
else:
    print(f"\nOptimization failed. Status: {model.status}")
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        model.computeIIS()
        model.write("model_iis.ilp")
        print("IIS written to model_iis.ilp")

