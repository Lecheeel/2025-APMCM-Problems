"""
QUBO Builder Module
===================

Builds QUBO model from UC problem with constraints converted to penalty terms.
"""

import numpy as np
from typing import Dict, List, Tuple
import kaiwu as kw
from .data_loader import UCDataLoader
from .discretization import GenerationDiscretizer


class QUBOBuilder:
    """
    Builds QUBO model for UC problem.
    
    Since all units stay online (u_{i,t} = 1 fixed), we only need to
    optimize generation levels p_{i,t}, which are discretized to binary variables.
    """
    
    def __init__(self, data_loader: UCDataLoader, discretizer: GenerationDiscretizer,
                 penalty_power_balance: float = None,
                 penalty_ramp: float = None,
                 penalty_reserve: float = None,
                 penalty_n1: float = None):
        """
        Initialize QUBO builder.
        
        Args:
            data_loader: UCDataLoader instance
            discretizer: GenerationDiscretizer instance
            penalty_power_balance: Penalty coefficient for power balance constraint
            penalty_ramp: Penalty coefficient for ramp constraints
            penalty_reserve: Penalty coefficient for reserve constraints
            penalty_n1: Penalty coefficient for N-1 constraints
        """
        self.data = data_loader
        self.discretizer = discretizer
        self.num_units = data_loader.num_units
        self.num_periods = data_loader.num_periods
        self.num_bits = discretizer.num_bits
        
        # Create binary variables once (shared across all expressions)
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
        # Power expression: p = P_min + sum_k (2^k * delta * bit_k)
        # For 1-bit: p = P_min + bit0 * (P_max - P_min)
        # For 2-bit: p = P_min + (bit0 + 2*bit1) * delta, where delta = (P_max - P_min) / 3
        # For 3-bit: p = P_min + (bit0 + 2*bit1 + 4*bit2) * delta, where delta = (P_max - P_min) / 7
        
        p_expr = kw.core.BinaryExpression(offset=self.data.P_min[i])
        for k in range(self.num_bits):
            coeff = self.discretizer.get_coefficient_for_bit(i, k)
            p_expr = p_expr + coeff * self.p_vars[(i, t, k)]
        return p_expr
    
    def _estimate_power_balance_penalty(self) -> float:
        """Estimate penalty coefficient for power balance constraint."""
        # Estimate objective function scale
        max_obj = np.sum(self.data.a_coeff * self.data.P_max**2 + 
                        self.data.b_coeff * self.data.P_max)
        
        # Maximum cost difference: using all P_min vs all P_max
        min_obj = np.sum(self.data.a_coeff * self.data.P_min**2 + 
                        self.data.b_coeff * self.data.P_min)
        max_cost_diff = max_obj - min_obj
        
        # For 2-bit discretization, typical violation is much smaller
        # Step size: delta = (P_max - P_min) / 3
        # Typical violation: a few delta values, much smaller than 1-bit case
        if self.num_bits == 2:
            # With 2-bit, max step per unit is about (P_max - P_min) / 3
            # Typical violation: sum of a few delta values across units
            avg_delta = np.mean(self.discretizer.delta)
            # Conservative estimate: worst case violation ~ 3-5 delta values
            typical_violation = avg_delta * 4  # ~4 steps across all units
            # But we still need to ensure constraint satisfaction
            # Use moderate penalty that balances precision and constraint satisfaction
            penalty_base = max_cost_diff * 200.0 / (typical_violation**2) if typical_violation > 0 else 1e6
        elif self.num_bits == 1:
            # 1-bit: large violations possible
            max_load = np.max(self.data.load_demand)
            min_total_gen = np.sum(self.data.P_min)
            max_violation = max_load - min_total_gen
            penalty_base = max_cost_diff * 100.0 / (max_violation**2) if max_violation > 0 else 1e6
            penalty_base *= 1000.0  # Extra multiplier for 1-bit case
        else:
            # Multi-bit: use moderate penalty
            avg_delta = np.mean(self.discretizer.delta)
            typical_violation = avg_delta * 3
            penalty_base = max_cost_diff * 100.0 / (typical_violation**2) if typical_violation > 0 else 1e6
        
        # Ensure minimum penalty, but not too large to avoid precision loss
        if self.num_bits == 2:
            # For 2-bit, need larger penalty to ensure constraint satisfaction
            # but balance with precision requirements
            return max(penalty_base, 1e5)  # Increased minimum for 2-bit
        else:
            return max(penalty_base, 1e5)  # Higher minimum for 1-bit
    
    def _estimate_ramp_penalty(self) -> float:
        """Estimate penalty coefficient for ramp constraints."""
        max_obj = np.sum(self.data.a_coeff * self.data.P_max**2)
        
        if self.num_bits == 2:
            # With 2-bit, max step per unit is about 2*delta = 2/3 * (P_max - P_min)
            # Typical violation: about 1-2 steps = 1-2*delta
            avg_delta = np.mean(self.discretizer.delta)
            typical_ramp_violation = avg_delta * 2  # ~2 steps
            # Use larger penalty for 2-bit to ensure constraint satisfaction
            multiplier = 500.0  # Increased from 200.0
            penalty = max_obj * multiplier / (typical_ramp_violation**2) if typical_ramp_violation > 0 else 1e6
        elif self.num_bits == 1:
            # 1-bit: large jumps possible (P_min <-> P_max)
            max_ramp_violation = np.max(self.data.P_max - self.data.P_min)
            multiplier = 1000.0
            penalty = max_obj * multiplier / max_ramp_violation**2
        else:
            # Multi-bit: moderate penalty
            avg_delta = np.mean(self.discretizer.delta)
            typical_ramp_violation = avg_delta * 1.5
            multiplier = 100.0
            penalty = max_obj * multiplier / (typical_ramp_violation**2) if typical_ramp_violation > 0 else 1e6
        
        if self.num_bits == 2:
            return max(penalty, 1e4)  # Increased minimum for 2-bit
        else:
            return max(penalty, 1e4)  # Higher minimum for 1-bit
    
    def _estimate_reserve_penalty(self) -> float:
        """Estimate penalty coefficient for reserve constraints."""
        max_reserve_violation = np.max(self.data.load_demand)
        max_obj = np.sum(self.data.a_coeff * self.data.P_max**2)
        # Increase penalty for 1-bit case
        multiplier = 50.0 if self.num_bits == 1 else 5.0
        return max_obj * multiplier / max_reserve_violation**2
    
    def _estimate_n1_penalty(self) -> float:
        """Estimate penalty coefficient for N-1 constraints."""
        max_n1_violation = np.max(self.data.load_demand)
        max_obj = np.sum(self.data.a_coeff * self.data.P_max**2)
        return max_obj * 5.0 / max_n1_violation**2
    
    def build_objective(self) -> kw.core.BinaryExpression:
        """
        Build objective function: fuel cost.
        
        Since u_{i,t} = 1 (fixed), startup/shutdown costs are zero.
        """
        # Objective: sum_i sum_t [a_i * p_{i,t}^2 + b_i * p_{i,t} + c_i]
        objective = kw.core.BinaryExpression(offset=0.0)
        
        for i in range(self.num_units):
            for t in range(self.num_periods):
                p_expr = self._get_power_expression(i, t)
                
                # Fuel cost: a_i * p^2 + b_i * p + c_i
                fuel_cost = (self.data.a_coeff[i] * p_expr * p_expr + 
                            self.data.b_coeff[i] * p_expr + 
                            self.data.c_coeff[i])
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
            balance_error = total_gen - self.data.load_demand[t]
            penalty = penalty + self.penalty_power_balance * balance_error * balance_error
        
        return penalty
    
    def build_ramp_constraints(self) -> kw.core.BinaryExpression:
        """
        Build ramp constraints as penalty terms.
        
        Constraints:
        - p_{i,t} - p_{i,t-1} <= Ramp_Up[i] (for t >= 1)
        - p_{i,t-1} - p_{i,t} <= Ramp_Down[i] (for t >= 1)
        
        Penalties: lambda * max(0, violation)^2
        """
        penalty = kw.core.BinaryExpression(offset=0.0)
        
        for i in range(self.num_units):
            # Initial power: use P_min (or could use actual initial from Problem 2)
            # For 1-bit discretization, we need to be careful about initial state
            # If initial is closer to P_max, start with P_max to avoid ramp violations
            initial_power = self.data.P_min[i]  # Default to P_min
            p_prev = kw.core.BinaryExpression(offset=initial_power)
            
            for t in range(self.num_periods):
                # Current power
                p_curr = self._get_power_expression(i, t)
                
                # Ramp-up constraint: p_curr - p_prev <= Ramp_Up
                ramp_up_error = p_curr - p_prev - self.data.Ramp_Up[i]
                # Use quadratic penalty: max(0, error)^2 approximated as error^2 when error > 0
                # For QUBO, we use: error^2 (will be penalized when positive)
                # For 1-bit, violations can be large (P_max - P_min), so use larger penalty
                ramp_penalty_coeff = self.penalty_ramp
                if self.num_bits == 1:
                    # Extra penalty for large violations (P_min <-> P_max jumps)
                    # Check if this would be a large jump
                    max_jump = self.data.P_max[i] - self.data.P_min[i]
                    if max_jump > self.data.Ramp_Up[i] * 2:
                        ramp_penalty_coeff *= 10.0  # Extra penalty for large jumps
                
                penalty = penalty + ramp_penalty_coeff * ramp_up_error * ramp_up_error
                
                # Ramp-down constraint: p_prev - p_curr <= Ramp_Down
                ramp_down_error = p_prev - p_curr - self.data.Ramp_Down[i]
                penalty = penalty + ramp_penalty_coeff * ramp_down_error * ramp_down_error
                
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
                available_reserve = available_reserve + (self.data.P_max[i] - p_expr)
            
            # Reserve constraint violation: R_t - available_reserve
            reserve_error = self.data.spinning_reserve_req[t] - available_reserve
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
            remaining_capacity = np.sum(self.data.P_max) - self.data.P_max[gen_out_idx]
            
            for t in range(self.num_periods):
                # Required capacity: D_t + R_t
                required_capacity = self.data.load_demand[t] + self.data.spinning_reserve_req[t]
                
                # Constraint violation: required - remaining
                n1_error = required_capacity - remaining_capacity
                if n1_error > 0:
                    # This is a hard constraint - if violated, system is infeasible
                    # Use large penalty (constant term)
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
        print("Building QUBO model...")
        
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

