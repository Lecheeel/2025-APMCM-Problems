"""
Reference Solution Module
=========================

Converts Problem 2 results to QUBO solution format.
"""

import numpy as np
import json
import os
from typing import Dict, Optional
from .data_loader import UCDataLoader
from .discretization import GenerationDiscretizer


class ReferenceSolutionConverter:
    """Converts Problem 2 results to QUBO solution."""
    
    def __init__(self, data_loader: UCDataLoader, discretizer: GenerationDiscretizer, results_dir: str = None):
        """
        Initialize converter.
        
        Args:
            data_loader: UCDataLoader instance
            discretizer: GenerationDiscretizer instance
            results_dir: Directory containing Problem 2 results (optional)
        """
        self.data = data_loader
        self.discretizer = discretizer
        self.results_dir = results_dir
    
    def load_problem2_generation(self) -> Optional[np.ndarray]:
        """
        Load generation schedule from Problem 2 CSV file.
        
        Returns:
            Generation matrix (num_units, num_periods) or None if not found
        """
        try:
            # Use provided results_dir or default
            if self.results_dir:
                results_dir = self.results_dir
            else:
                # Use the same results_dir logic as data_loader
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(script_dir))
                results_dir = os.path.join(project_root, 'results', 'problem2')
            
            # Check if directory exists
            if not os.path.exists(results_dir):
                return None
            
            # Find latest CSV file
            csv_files = [f for f in os.listdir(results_dir) 
                        if f.startswith('uc_schedule_') and f.endswith('.csv')]
            if not csv_files:
                return None
            
            latest_file = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
            csv_path = os.path.join(results_dir, latest_file)
            
            # Read CSV
            import csv
            generation = np.zeros((self.data.num_units, self.data.num_periods))
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    period = int(row['Period']) - 1  # 0-indexed
                    for i, unit_id in enumerate(self.data.units):
                        col_name = f'Unit_{unit_id}_Generation_MW'
                        if col_name in row:
                            generation[i, period] = float(row[col_name])
            
            return generation
        except Exception as e:
            print(f"Warning: Could not load Problem 2 generation: {e}")
            return None
    
    def generation_to_qubo_solution(self, generation: np.ndarray) -> Dict:
        """
        Convert generation matrix to QUBO solution dictionary.
        
        Args:
            generation: Generation matrix (num_units, num_periods)
            
        Returns:
            Solution dictionary compatible with QUBO solver
        """
        sol_dict = {}
        
        # Convert generation to binary variables
        for i in range(self.data.num_units):
            for t in range(self.data.num_periods):
                # Find closest binary representation
                p = generation[i, t]
                
                if self.discretizer.num_bits == 1:
                    # For 1-bit: choose P_min or P_max based on which is closer
                    dist_to_min = abs(p - self.data.P_min[i])
                    dist_to_max = abs(p - self.data.P_max[i])
                    
                    if dist_to_max < dist_to_min:
                        # Choose P_max (bit = 1)
                        var_name = self.discretizer.get_variable_name(i, t, 0)
                        sol_dict[var_name] = 1
                    else:
                        # Choose P_min (bit = 0)
                        var_name = self.discretizer.get_variable_name(i, t, 0)
                        sol_dict[var_name] = 0
                else:
                    # Multi-bit encoding: find closest discrete value
                    # For 2-bit: values are P_min, P_min+delta, P_min+2*delta, P_max
                    # where delta = (P_max - P_min) / 3
                    p_norm = (p - self.data.P_min[i]) / self.discretizer.delta[i]
                    p_norm = np.clip(p_norm, 0, 2**self.discretizer.num_bits - 1)
                    p_int = int(np.round(p_norm))
                    # Ensure p_int is within valid range [0, 2^num_bits - 1]
                    p_int = max(0, min(p_int, 2**self.discretizer.num_bits - 1))
                    
                    # Convert to binary representation
                    for k in range(self.discretizer.num_bits):
                        var_name = self.discretizer.get_variable_name(i, t, k)
                        sol_dict[var_name] = (p_int >> k) & 1
        
        return sol_dict
    
    def get_reference_solution(self) -> Optional[Dict]:
        """
        Get reference solution from Problem 2 results.
        
        Returns:
            Solution dictionary or None if not available
        """
        generation_continuous = self.load_problem2_generation()
        if generation_continuous is None:
            return None
        
        # Convert continuous generation to binary QUBO solution
        sol_dict = self.generation_to_qubo_solution(generation_continuous)
        
        # Convert binary solution back to discrete generation values
        # This reflects the actual discrete values used in QUBO
        generation_discrete = np.zeros_like(generation_continuous)
        for i in range(self.data.num_units):
            for t in range(self.data.num_periods):
                generation_discrete[i, t] = self.discretizer.get_generation_value(i, t, sol_dict)
        
        # Calculate objective value using DISCRETE generation values
        # This reflects the true cost of the discretized solution
        total_cost = 0.0
        for i in range(self.data.num_units):
            for t in range(self.data.num_periods):
                p = generation_discrete[i, t]
                fuel_cost = (self.data.a_coeff[i] * p**2 + 
                            self.data.b_coeff[i] * p + 
                            self.data.c_coeff[i])
                total_cost += fuel_cost
        
        return {
            'sol_dict': sol_dict,
            'objective': total_cost,
            'generation': generation_discrete,  # Use discrete values
            'generation_continuous': generation_continuous,  # Keep original for comparison
            'is_reference': True
        }

