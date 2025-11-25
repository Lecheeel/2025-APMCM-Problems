"""
Data Loader Module
==================

Loads and prepares data for Problem 3 QUBO transformation.
Based on Problem 2 results, all units remain online at all periods.
"""

import numpy as np
from typing import Dict, List, Tuple
import json
import os


class UCDataLoader:
    """Loads UC problem data from tables and Problem 2 results."""
    
    def __init__(self, results_dir: str = None):
        """
        Initialize data loader.
        
        Args:
            results_dir: Directory containing Problem 2 results (optional)
        """
        self.units = [1, 2, 5, 8, 11, 13]
        self.num_units = len(self.units)
        self.num_periods = 24
        self.unit_to_idx = {unit: idx for idx, unit in enumerate(self.units)}
        
        # Load unit parameters from Table 1 and Table 2
        self._load_unit_parameters()
        
        # Load network data
        self._load_network_data()
        
        # Load load demand
        self._load_demand_data()
        
        # Load Problem 2 results if available
        self.problem2_results = None
        if results_dir:
            self._load_problem2_results(results_dir)
    
    def _load_unit_parameters(self):
        """Load unit parameters from Table 1 and Table 2."""
        # Table 1: Parameters Part A
        table1_data = {
            1: {'P_max': 300, 'P_min': 50, 'Startup_Cost': 180, 'Shutdown_Cost': 180, 'Ramp_Up': 80},
            2: {'P_max': 180, 'P_min': 20, 'Startup_Cost': 180, 'Shutdown_Cost': 180, 'Ramp_Up': 80},
            5: {'P_max': 50, 'P_min': 15, 'Startup_Cost': 40, 'Shutdown_Cost': 40, 'Ramp_Up': 50},
            8: {'P_max': 35, 'P_min': 10, 'Startup_Cost': 60, 'Shutdown_Cost': 60, 'Ramp_Up': 50},
            11: {'P_max': 30, 'P_min': 10, 'Startup_Cost': 60, 'Shutdown_Cost': 60, 'Ramp_Up': 60},
            13: {'P_max': 40, 'P_min': 12, 'Startup_Cost': 40, 'Shutdown_Cost': 40, 'Ramp_Up': 60},
        }
        
        # Table 2: Parameters Part B (authoritative)
        table2_data = {
            1: {'Ramp_Down': 80, 'Min_Up_Time': 5, 'Min_Down_Time': 3, 
                'Initial_Up_Time': 5, 'Initial_Down_Time': 0,
                'a': 0.02, 'b': 2.00, 'c': 0, 'H': 7.0},
            2: {'Ramp_Down': 80, 'Min_Up_Time': 4, 'Min_Down_Time': 2,
                'Initial_Up_Time': 4, 'Initial_Down_Time': 0,
                'a': 0.0175, 'b': 1.75, 'c': 0, 'H': 4.5},
            5: {'Ramp_Down': 50, 'Min_Up_Time': 3, 'Min_Down_Time': 2,
                'Initial_Up_Time': 3, 'Initial_Down_Time': 0,
                'a': 0.0625, 'b': 1.00, 'c': 0, 'H': 4.5},
            8: {'Ramp_Down': 50, 'Min_Up_Time': 3, 'Min_Down_Time': 2,
                'Initial_Up_Time': 3, 'Initial_Down_Time': 0,
                'a': 0.00834, 'b': 3.25, 'c': 0, 'H': 3.2},
            11: {'Ramp_Down': 60, 'Min_Up_Time': 1, 'Min_Down_Time': 1,
                'Initial_Up_Time': 3, 'Initial_Down_Time': 0,
                'a': 0.025, 'b': 3.00, 'c': 0, 'H': 3.0},
            13: {'Ramp_Down': 60, 'Min_Up_Time': 4, 'Min_Down_Time': 2,
                'Initial_Up_Time': 4, 'Initial_Down_Time': 0,
                'a': 0.025, 'b': 3.00, 'c': 0, 'H': 3.0},
        }
        
        # Extract parameters (Table 2 takes precedence)
        self.P_max = np.array([table1_data[u]['P_max'] for u in self.units])
        self.P_min = np.array([table1_data[u]['P_min'] for u in self.units])
        self.Ramp_Up = np.array([table1_data[u]['Ramp_Up'] for u in self.units])
        self.Ramp_Down = np.array([table2_data[u]['Ramp_Down'] for u in self.units])
        self.a_coeff = np.array([table2_data[u]['a'] for u in self.units])
        self.b_coeff = np.array([table2_data[u]['b'] for u in self.units])
        self.c_coeff = np.array([table2_data[u]['c'] for u in self.units])
        self.Min_Up_Time = np.array([table2_data[u]['Min_Up_Time'] for u in self.units])
        self.Min_Down_Time = np.array([table2_data[u]['Min_Down_Time'] for u in self.units])
        self.Initial_Up_Time = np.array([table2_data[u]['Initial_Up_Time'] for u in self.units])
        self.Initial_Down_Time = np.array([table2_data[u]['Initial_Down_Time'] for u in self.units])
        self.Startup_Cost = np.array([table1_data[u]['Startup_Cost'] for u in self.units])
        self.Shutdown_Cost = np.array([table1_data[u]['Shutdown_Cost'] for u in self.units])
        self.H_inertia = np.array([table2_data[u]['H'] for u in self.units])
        
        # Initial status: all units start online (based on Problem 2 results)
        self.initial_status = np.ones(self.num_units, dtype=int)
        
        # Based on Problem 2 results: all units stay online at all periods
        # So u_{i,t} = 1 for all i, t (fixed)
        self.unit_status_fixed = np.ones((self.num_units, self.num_periods), dtype=int)
    
    def _load_network_data(self):
        """Load network parameters from Table 3."""
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
        
        # Process branches
        all_buses = set()
        for branch in branches_data:
            all_buses.add(branch['from'])
            all_buses.add(branch['to'])
        self.all_buses = sorted(list(all_buses))
        self.num_buses = len(self.all_buses)
        self.bus_to_idx = {bus: idx for idx, bus in enumerate(self.all_buses)}
        
        # Calculate susceptance B = 1/X for DC power flow
        self.branches = []
        for branch in branches_data:
            X = branch['X']
            if X > 1e-6:
                B = 1.0 / X
            else:
                B = 1e6  # Large but finite value for zero reactance
            self.branches.append({
                'from': branch['from'],
                'to': branch['to'],
                'from_idx': self.bus_to_idx[branch['from']],
                'to_idx': self.bus_to_idx[branch['to']],
                'B': B,
                'P_max': branch['P_max']
            })
        self.num_branches = len(self.branches)
        
        # Load distribution (simplified: equal distribution to load buses)
        load_buses = [b for b in self.all_buses if b not in self.units]
        self.load_distribution = {bus: 1.0 / len(load_buses) for bus in load_buses}
        for bus in self.units:
            self.load_distribution[bus] = 0.0
    
    def _load_demand_data(self):
        """Load demand data from Table 4."""
        self.load_demand = np.array([
            166, 196, 229, 257, 263, 252, 246, 213, 192, 161, 147, 160,
            170, 185, 208, 232, 246, 241, 236, 225, 204, 182, 161, 131
        ])
        
        # Calculate spinning reserve requirements
        largest_unit_capacity = np.max(self.P_max)
        total_capacity = np.sum(self.P_max)
        self.spinning_reserve_req = []
        for t in range(self.num_periods):
            reserve_10pct = 0.10 * self.load_demand[t]
            reserve_largest = largest_unit_capacity
            max_feasible_reserve = max(0, (total_capacity - largest_unit_capacity) - self.load_demand[t])
            reserve_standard = max(reserve_10pct, reserve_largest)
            reserve_final = min(reserve_standard, max_feasible_reserve)
            reserve_final = max(reserve_final, 0.05 * self.load_demand[t])
            self.spinning_reserve_req.append(reserve_final)
        self.spinning_reserve_req = np.array(self.spinning_reserve_req)
    
    def _load_problem2_results(self, results_dir: str):
        """Load Problem 2 optimization results."""
        try:
            # Find latest summary file
            summary_files = [f for f in os.listdir(results_dir) if f.startswith('summary_') and f.endswith('.json')]
            if summary_files:
                latest_file = max(summary_files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
                with open(os.path.join(results_dir, latest_file), 'r') as f:
                    self.problem2_results = json.load(f)
                print(f"Loaded Problem 2 results from: {latest_file}")
        except Exception as e:
            print(f"Warning: Could not load Problem 2 results: {e}")
    
    def get_summary(self) -> Dict:
        """Get summary of loaded data."""
        return {
            'num_units': self.num_units,
            'units': self.units,
            'num_periods': self.num_periods,
            'num_buses': self.num_buses,
            'num_branches': self.num_branches,
            'total_capacity': float(np.sum(self.P_max)),
            'load_range': [float(np.min(self.load_demand)), float(np.max(self.load_demand))],
            'all_units_online': True  # Based on Problem 2 results
        }

