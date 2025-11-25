"""
Data Loader Module for Problem 4
==================================

Loads and prepares data for reduced-scale QUBO transformation.
Based on Problem 1 and Problem 2 conclusions: all units stay online.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os


class ReducedDataLoader:
    """Loads UC problem data and supports reduction strategies."""
    
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
        
        # Based on Problem 1 and Problem 2: all units stay online at all periods
        self.unit_status_fixed = np.ones((self.num_units, self.num_periods), dtype=int)
    
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
            'total_capacity': float(np.sum(self.P_max)),
            'load_range': [float(np.min(self.load_demand)), float(np.max(self.load_demand))],
            'all_units_online': True  # Based on Problem 1 and Problem 2 results
        }

