"""
Verification Module
===================

Verifies solution quality and constraint satisfaction.
"""

import numpy as np
from typing import Dict, List, Tuple
from .data_loader import UCDataLoader


class SolutionVerifier:
    """Verifies UC solution quality and constraints."""
    
    def __init__( self, data_loader: UCDataLoader):
        """
        Initialize verifier.
        
        Args:
            data_loader: UCDataLoader instance
        """
        self.data = data_loader
    
    def verify_power_balance(self, generation: np.ndarray) -> Tuple[bool, Dict]:
        """
        Verify power balance constraints.
        
        Args:
            generation: Generation matrix (num_units, num_periods)
            
        Returns:
            (is_valid, violation_info)
        """
        violations = []
        max_violation = 0.0
        
        for t in range(self.data.num_periods):
            total_gen = np.sum(generation[:, t])
            demand = self.data.load_demand[t]
            violation = abs(total_gen - demand)
            
            if violation > 1e-3:  # Tolerance: 1 MW
                violations.append({
                    'period': t + 1,
                    'total_generation': float(total_gen),
                    'demand': float(demand),
                    'violation': float(violation)
                })
                max_violation = max(max_violation, violation)
        
        is_valid = len(violations) == 0
        
        return is_valid, {
            'is_valid': is_valid,
            'max_violation': float(max_violation),
            'violations': violations
        }
    
    def verify_generation_limits(self, generation: np.ndarray) -> Tuple[bool, Dict]:
        """
        Verify generation limits.
        
        Args:
            generation: Generation matrix (num_units, num_periods)
            
        Returns:
            (is_valid, violation_info)
        """
        violations = []
        
        for i in range(self.data.num_units):
            for t in range(self.data.num_periods):
                p = generation[i, t]
                
                if p < self.data.P_min[i] - 1e-3:
                    violations.append({
                        'unit': self.data.units[i],
                        'period': t + 1,
                        'generation': float(p),
                        'min_limit': float(self.data.P_min[i]),
                        'violation': float(self.data.P_min[i] - p)
                    })
                elif p > self.data.P_max[i] + 1e-3:
                    violations.append({
                        'unit': self.data.units[i],
                        'period': t + 1,
                        'generation': float(p),
                        'max_limit': float(self.data.P_max[i]),
                        'violation': float(p - self.data.P_max[i])
                    })
        
        is_valid = len(violations) == 0
        
        return is_valid, {
            'is_valid': is_valid,
            'violations': violations
        }
    
    def verify_ramp_constraints(self, generation: np.ndarray) -> Tuple[bool, Dict]:
        """
        Verify ramp constraints.
        
        Args:
            generation: Generation matrix (num_units, num_periods)
            
        Returns:
            (is_valid, violation_info)
        """
        violations = []
        
        for i in range(self.data.num_units):
            # Initial power (assume P_min)
            p_prev = self.data.P_min[i]
            
            for t in range(self.data.num_periods):
                p_curr = generation[i, t]
                
                # Ramp-up constraint
                ramp_up = p_curr - p_prev
                if ramp_up > self.data.Ramp_Up[i] + 1e-3:
                    violations.append({
                        'unit': self.data.units[i],
                        'period': t + 1,
                        'ramp_up': float(ramp_up),
                        'limit': float(self.data.Ramp_Up[i]),
                        'violation': float(ramp_up - self.data.Ramp_Up[i]),
                        'type': 'ramp_up'
                    })
                
                # Ramp-down constraint
                ramp_down = p_prev - p_curr
                if ramp_down > self.data.Ramp_Down[i] + 1e-3:
                    violations.append({
                        'unit': self.data.units[i],
                        'period': t + 1,
                        'ramp_down': float(ramp_down),
                        'limit': float(self.data.Ramp_Down[i]),
                        'violation': float(ramp_down - self.data.Ramp_Down[i]),
                        'type': 'ramp_down'
                    })
                
                p_prev = p_curr
        
        is_valid = len(violations) == 0
        
        return is_valid, {
            'is_valid': is_valid,
            'violations': violations
        }
    
    def verify_reserve_constraints(self, generation: np.ndarray) -> Tuple[bool, Dict]:
        """
        Verify spinning reserve constraints.
        
        Args:
            generation: Generation matrix (num_units, num_periods)
            
        Returns:
            (is_valid, violation_info)
        """
        violations = []
        
        for t in range(self.data.num_periods):
            available_reserve = np.sum(self.data.P_max - generation[:, t])
            required_reserve = self.data.spinning_reserve_req[t]
            
            if available_reserve < required_reserve - 1e-3:
                violations.append({
                    'period': t + 1,
                    'available_reserve': float(available_reserve),
                    'required_reserve': float(required_reserve),
                    'violation': float(required_reserve - available_reserve)
                })
        
        is_valid = len(violations) == 0
        
        return is_valid, {
            'is_valid': is_valid,
            'violations': violations
        }
    
    def verify_all(self, generation: np.ndarray) -> Dict:
        """
        Verify all constraints.
        
        Args:
            generation: Generation matrix (num_units, num_periods)
            
        Returns:
            Verification results dictionary
        """
        results = {}
        
        # Power balance
        pb_valid, pb_info = self.verify_power_balance(generation)
        results['power_balance'] = pb_info
        
        # Generation limits
        gl_valid, gl_info = self.verify_generation_limits(generation)
        results['generation_limits'] = gl_info
        
        # Ramp constraints
        ramp_valid, ramp_info = self.verify_ramp_constraints(generation)
        results['ramp_constraints'] = ramp_info
        
        # Reserve constraints
        reserve_valid, reserve_info = self.verify_reserve_constraints(generation)
        results['reserve_constraints'] = reserve_info
        
        # Overall validity
        results['all_valid'] = pb_valid and gl_valid and ramp_valid and reserve_valid
        
        return results
    
    def compare_with_problem2(self, generation: np.ndarray, problem2_results: Dict) -> Dict:
        """
        Compare solution with Problem 2 results.
        
        Args:
            generation: Generation matrix (num_units, num_periods)
            problem2_results: Problem 2 results dictionary
            
        Returns:
            Comparison results
        """
        if problem2_results is None:
            return {'error': 'Problem 2 results not available'}
        
        # Calculate costs
        qubo_cost = self.calculate_cost(generation)
        problem2_cost = problem2_results.get('optimization_info', {}).get('total_cost', None)
        
        # Calculate generation differences
        if 'unit_statistics' in problem2_results:
            # Would need to load actual generation from CSV for detailed comparison
            pass
        
        return {
            'qubo_cost': float(qubo_cost),
            'problem2_cost': float(problem2_cost) if problem2_cost else None,
            'cost_difference': float(qubo_cost - problem2_cost) if problem2_cost else None,
            'cost_ratio': float(qubo_cost / problem2_cost) if problem2_cost else None
        }
    
    def calculate_cost(self, generation: np.ndarray) -> float:
        """
        Calculate total cost from generation.
        
        Args:
            generation: Generation matrix (num_units, num_periods)
            
        Returns:
            Total cost
        """
        total_cost = 0.0
        
        for i in range(self.data.num_units):
            for t in range(self.data.num_periods):
                p = generation[i, t]
                fuel_cost = (self.data.a_coeff[i] * p**2 + 
                            self.data.b_coeff[i] * p + 
                            self.data.c_coeff[i])
                total_cost += fuel_cost
        
        return total_cost

