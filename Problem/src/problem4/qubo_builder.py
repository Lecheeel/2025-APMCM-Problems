"""
QUBO Builder Module for Problem 4
==================================

Builds QUBO model from reduced UC problem with constraints converted to penalty terms.
Based on Problem 3 QUBO builder but adapted for reduced problem size.

Kaiwu SDK Documentation:
- English: https://kaiwu-sdk-docs.qboson.com/en/latest/
- Chinese: https://kaiwu-sdk-docs.qboson.com/zh/v1.2.0/source/getting_started/index.html
"""

import numpy as np
from typing import Dict, List, Tuple
import kaiwu as kw
from .discretization import GenerationDiscretizer


class ReducedQUBOBuilder:
    """
    Builds QUBO model for reduced UC problem.
    
    Key simplification: All units stay online (u_{i,t} = 1 fixed),
    so we only optimize generation levels p_{i,t}.
    """
    
    def __init__(self, reduced_data: Dict, discretizer: GenerationDiscretizer,
                 penalty_power_balance: float = None,
                 penalty_ramp: float = None,
                 penalty_reserve: float = None,
                 penalty_n1: float = None):
        """
        Initialize QUBO builder.
        
        Args:
            reduced_data: Dictionary with reduced problem data
            discretizer: GenerationDiscretizer instance
            penalty_power_balance: Penalty coefficient for power balance constraint
            penalty_ramp: Penalty coefficient for ramp constraints
            penalty_reserve: Penalty coefficient for reserve constraints
            penalty_n1: Penalty coefficient for N-1 constraints
        """
        self.data = reduced_data
        self.discretizer = discretizer
        self.num_units = reduced_data['num_units']
        self.num_periods = reduced_data['num_periods']
        self.num_bits = discretizer.num_bits
        
        # Create binary variables once
        self.p_vars = {}
        for i in range(self.num_units):
            for t in range(self.num_periods):
                for k in range(self.num_bits):
                    var_name = self.discretizer.get_variable_name(i, t, k)
                    self.p_vars[(i, t, k)] = kw.core.Binary(var_name)
        
        # Auto-calculate penalty coefficients if not provided
        if penalty_power_balance is None:
            penalty_power_balance = self._estimate_power_balance_penalty()
        if penalty_ramp is None:
            penalty_ramp = self._estimate_ramp_penalty()
        if penalty_reserve is None:
            penalty_reserve = self._estimate_reserve_penalty()
        if penalty_n1 is None:
            penalty_n1 = self._estimate_n1_penalty()
        
        self.penalty_power_balance = penalty_power_balance
        self.penalty_ramp = penalty_ramp
        self.penalty_reserve = penalty_reserve
        self.penalty_n1 = penalty_n1
        
        print(f"Penalty coefficients:")
        print(f"  Power balance: {penalty_power_balance:.2e}")
        print(f"  Ramp: {penalty_ramp:.2e}")
        print(f"  Reserve: {penalty_reserve:.2e}")
        print(f"  N-1: {penalty_n1:.2e}")
    
    def _get_power_expression(self, i: int, t: int) -> kw.core.BinaryExpression:
        """Get power expression for unit i at period t."""
        p_expr = kw.core.BinaryExpression(offset=self.data['P_min'][i])
        for k in range(self.num_bits):
            coeff = self.discretizer.get_coefficient_for_bit(i, k)
            p_expr = p_expr + coeff * self.p_vars[(i, t, k)]
        return p_expr
    
    def _estimate_power_balance_penalty(self) -> float:
        """Estimate penalty coefficient for power balance constraint."""
        max_obj = np.sum(self.data['a_coeff'] * self.data['P_max']**2 + 
                        self.data['b_coeff'] * self.data['P_max'])
        min_obj = np.sum(self.data['a_coeff'] * self.data['P_min']**2 + 
                        self.data['b_coeff'] * self.data['P_min'])
        max_cost_diff = max_obj - min_obj
        
        if self.num_bits == 1:
            max_load = np.max(self.data['load_demand'])
            min_total_gen = np.sum(self.data['P_min'])
            max_violation = max_load - min_total_gen
            # Reduce penalty to keep coefficients smaller for CIM compatibility
            penalty_base = max_cost_diff * 100.0 / (max_violation**2) if max_violation > 0 else 1e4
        else:
            avg_delta = np.mean(self.discretizer.delta)
            typical_violation = avg_delta * 4
            penalty_base = max_cost_diff * 20.0 / (typical_violation**2) if typical_violation > 0 else 1e4
        
        # Scale down to keep coefficients in reasonable range
        return max(penalty_base, 1e3)
    
    def _estimate_ramp_penalty(self) -> float:
        """Estimate penalty coefficient for ramp constraints."""
        max_obj = np.sum(self.data['a_coeff'] * self.data['P_max']**2)
        
        if self.num_bits == 1:
            max_ramp_violation = np.max(self.data['P_max'] - self.data['P_min'])
            # Reduce multiplier to keep coefficients smaller
            multiplier = 100.0
            penalty = max_obj * multiplier / max_ramp_violation**2
        else:
            avg_delta = np.mean(self.discretizer.delta)
            typical_ramp_violation = avg_delta * 2
            multiplier = 50.0
            penalty = max_obj * multiplier / (typical_ramp_violation**2) if typical_ramp_violation > 0 else 1e4
        
        # Scale down to keep coefficients in reasonable range
        return max(penalty, 1e2)
    
    def _estimate_reserve_penalty(self) -> float:
        """Estimate penalty coefficient for reserve constraints."""
        max_reserve_violation = np.max(self.data['load_demand'])
        max_obj = np.sum(self.data['a_coeff'] * self.data['P_max']**2)
        # Reduce multiplier significantly to keep coefficients smaller
        multiplier = 5.0 if self.num_bits == 1 else 0.5
        penalty = max_obj * multiplier / max_reserve_violation**2
        return max(penalty, 1e-1)
    
    def _estimate_n1_penalty(self) -> float:
        """Estimate penalty coefficient for N-1 constraints."""
        max_n1_violation = np.max(self.data['load_demand'])
        max_obj = np.sum(self.data['a_coeff'] * self.data['P_max']**2)
        # Reduce multiplier to keep coefficients smaller
        penalty = max_obj * 0.5 / max_n1_violation**2
        return max(penalty, 1e-2)
    
    def build_objective(self) -> kw.core.BinaryExpression:
        """
        Build objective function: fuel cost.
        
        Since u_{i,t} = 1 (fixed), startup/shutdown costs are zero.
        
        Note: We scale down the objective to keep QUBO coefficients smaller
        for CIM 8-bit compatibility. The scaling factor will be accounted for
        when extracting the solution.
        """
        # Scale factor to reduce coefficient magnitudes
        # This helps meet CIM 8-bit limit without losing too much precision
        scale_factor = 0.01  # Scale down by 100x
        
        objective = kw.core.BinaryExpression(offset=0.0)
        
        for i in range(self.num_units):
            for t in range(self.num_periods):
                p_expr = self._get_power_expression(i, t)
                
                # Fuel cost: a_i * p^2 + b_i * p + c_i
                # Scale down to keep coefficients smaller
                fuel_cost = scale_factor * (self.data['a_coeff'][i] * p_expr * p_expr + 
                            self.data['b_coeff'][i] * p_expr + 
                            self.data['c_coeff'][i])
                objective = objective + fuel_cost
        
        return objective
    
    def build_power_balance_constraints(self) -> kw.core.BinaryExpression:
        """
        Build power balance constraints as penalty terms.
        
        Constraint: sum_i p_{i,t} = D_t for all t
        Penalty: lambda * (sum_i p_{i,t} - D_t)^2
        """
        penalty = kw.core.BinaryExpression(offset=0.0)
        
        for t in range(self.num_periods):
            # Total generation: sum_i p_{i,t}
            total_gen = kw.core.BinaryExpression(offset=0.0)
            for i in range(self.num_units):
                p_expr = self._get_power_expression(i, t)
                total_gen = total_gen + p_expr
            
            # Penalty: lambda * (total_gen - D_t)^2
            balance_error = total_gen - self.data['load_demand'][t]
            penalty = penalty + self.penalty_power_balance * balance_error * balance_error
        
        return penalty
    
    def build_ramp_constraints(self) -> kw.core.BinaryExpression:
        """
        Build ramp constraints as penalty terms.
        
        Constraints:
        - p_{i,t} - p_{i,t-1} <= Ramp_Up[i] (for t >= 1)
        - p_{i,t-1} - p_{i,t} <= Ramp_Down[i] (for t >= 1)
        """
        penalty = kw.core.BinaryExpression(offset=0.0)
        
        for i in range(self.num_units):
            # Initial power: use P_min
            initial_power = self.data['P_min'][i]
            p_prev = kw.core.BinaryExpression(offset=initial_power)
            
            for t in range(self.num_periods):
                # Current power
                p_curr = self._get_power_expression(i, t)
                
                # Ramp-up constraint: p_curr - p_prev <= Ramp_Up
                ramp_up_error = p_curr - p_prev - self.data['Ramp_Up'][i]
                penalty = penalty + self.penalty_ramp * ramp_up_error * ramp_up_error
                
                # Ramp-down constraint: p_prev - p_curr <= Ramp_Down
                ramp_down_error = p_prev - p_curr - self.data['Ramp_Down'][i]
                penalty = penalty + self.penalty_ramp * ramp_down_error * ramp_down_error
                
                p_prev = p_curr
        
        return penalty
    
    def build_reserve_constraints(self) -> kw.core.BinaryExpression:
        """
        Build spinning reserve constraints as penalty terms.
        
        Constraint: sum_i (P_max[i] - p_{i,t}) >= R_t
        Penalty: lambda * max(0, R_t - available_reserve)^2
        """
        penalty = kw.core.BinaryExpression(offset=0.0)
        
        for t in range(self.num_periods):
            # Available reserve: sum_i (P_max[i] - p_{i,t})
            available_reserve = kw.core.BinaryExpression(offset=0.0)
            for i in range(self.num_units):
                p_expr = self._get_power_expression(i, t)
                available_reserve = available_reserve + (self.data['P_max'][i] - p_expr)
            
            # Reserve constraint violation: R_t - available_reserve
            reserve_error = self.data['spinning_reserve_req'][t] - available_reserve
            penalty = penalty + self.penalty_reserve * reserve_error * reserve_error
        
        return penalty
    
    def build_n1_constraints(self) -> kw.core.BinaryExpression:
        """
        Build N-1 security constraints as penalty terms.
        
        Constraint: sum_{j != i} P_max[j] >= D_t + R_t for all i, t
        Since all units are online, this simplifies to checking capacity.
        """
        penalty_offset = 0.0
        
        # For each generator outage scenario
        for gen_out_idx in range(self.num_units):
            # Remaining capacity (excluding outaged generator)
            remaining_capacity = np.sum(self.data['P_max']) - self.data['P_max'][gen_out_idx]
            
            for t in range(self.num_periods):
                # Required capacity: D_t + R_t
                required_capacity = self.data['load_demand'][t] + self.data['spinning_reserve_req'][t]
                
                # Constraint violation: required - remaining
                n1_error = required_capacity - remaining_capacity
                if n1_error > 0:
                    penalty_offset += self.penalty_n1 * n1_error * n1_error
        
        # Create penalty expression with offset
        penalty = kw.core.BinaryExpression(offset=penalty_offset)
        return penalty
    
    def build_qubo_model(self) -> kw.qubo.QuboModel:
        """
        Build complete QUBO model.
        
        Returns:
            QuboModel instance
        """
        print("Building reduced QUBO model...")
        
        # Build objective function
        objective = self.build_objective()
        print("  ✓ Objective function built")
        
        # Build constraint penalties
        power_balance_penalty = self.build_power_balance_constraints()
        print("  ✓ Power balance constraints built")
        
        ramp_penalty = self.build_ramp_constraints()
        print("  ✓ Ramp constraints built")
        
        reserve_penalty = self.build_reserve_constraints()
        print("  ✓ Reserve constraints built")
        
        n1_penalty = self.build_n1_constraints()
        print("  ✓ N-1 constraints built")
        
        # Combine objective and penalties
        qubo_expr = objective + power_balance_penalty + ramp_penalty + reserve_penalty + n1_penalty
        
        # Create QUBO model
        qubo_model = kw.qubo.QuboModel()
        qubo_model.set_objective(qubo_expr)
        
        print(f"  ✓ QUBO model built")
        print(f"  Total binary variables: {self.discretizer.get_num_binary_vars(self.num_periods)}")
        
        return qubo_model

