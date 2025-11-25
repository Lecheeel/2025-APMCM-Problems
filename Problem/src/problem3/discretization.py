"""
Discretization Module
=====================

Converts continuous generation variables p_{i,t} to binary variables
for QUBO formulation.
"""

import numpy as np
from typing import Tuple, Dict, List


class GenerationDiscretizer:
    """
    Discretizes continuous generation variables into binary variables.
    
    Uses binary encoding: p_{i,t} = P_min[i] + sum_k (2^k * delta[i] * x_{i,t,k})
    where delta[i] is the discretization step size for unit i.
    """
    
    def __init__(self, P_min: np.ndarray, P_max: np.ndarray, num_bits: int = None):
        """
        Initialize discretizer.
        
        Args:
            P_min: Minimum generation for each unit (MW)
            P_max: Maximum generation for each unit (MW)
            num_bits: Number of bits per unit per period (if None, auto-calculate)
        """
        self.P_min = P_min
        self.P_max = P_max
        self.num_units = len(P_min)
        self.generation_ranges = P_max - P_min
        
        # Set num_bits (should be provided, default handled by caller)
        if num_bits is None:
            num_bits = 1  # Default to 1-bit
        
        self.num_bits = num_bits
        
        # Calculate discretization step for each unit
        # delta[i] = (P_max[i] - P_min[i]) / (2^num_bits - 1)
        # 
        # For num_bits=1: delta = (P_max - P_min) / 1 = full range
        #   - Can represent: P_min (bit=0) or P_max (bit=1)
        #   - Value = P_min + bit * (P_max - P_min)
        #
        # For num_bits=2: delta = (P_max - P_min) / 3
        #   - Can represent: P_min, P_min+delta, P_min+2*delta, P_max
        #   - Value = P_min + (bit0 + 2*bit1) * delta
        #
        # For num_bits=3: delta = (P_max - P_min) / 7
        #   - Can represent: 8 discrete values
        #   - Value = P_min + (bit0 + 2*bit1 + 4*bit2) * delta
        
        if self.num_bits == 1:
            # With 1 bit: value = P_min + bit * (P_max - P_min)
            self.delta = self.generation_ranges  # Full range per bit
        else:
            # Multi-bit: delta = range / (2^num_bits - 1)
            # This gives us 2^num_bits discrete values
            self.delta = self.generation_ranges / (2**self.num_bits - 1)
        
        print(f"Discretization: {self.num_bits} bits per unit per period")
        print(f"Step sizes (MW): {self.delta}")
    
    def get_num_binary_vars(self, num_periods: int) -> int:
        """
        Get total number of binary variables needed.
        
        Args:
            num_periods: Number of time periods
            
        Returns:
            Total number of binary variables
        """
        return self.num_units * num_periods * self.num_bits
    
    def generation_to_binary(self, p: np.ndarray) -> np.ndarray:
        """
        Convert continuous generation to binary representation.
        
        Args:
            p: Generation matrix (num_units, num_periods)
            
        Returns:
            Binary matrix (num_units, num_periods, num_bits)
        """
        num_periods = p.shape[1]
        binary = np.zeros((self.num_units, num_periods, self.num_bits), dtype=int)
        
        for i in range(self.num_units):
            for t in range(num_periods):
                # Normalize: p_norm = (p[i,t] - P_min[i]) / delta[i]
                p_norm = (p[i, t] - self.P_min[i]) / self.delta[i]
                p_norm = np.clip(p_norm, 0, 2**self.num_bits - 1)
                p_int = int(np.round(p_norm))
                
                # Convert to binary
                for k in range(self.num_bits):
                    binary[i, t, k] = (p_int >> k) & 1
        
        return binary
    
    def binary_to_generation(self, binary: np.ndarray) -> np.ndarray:
        """
        Convert binary representation back to continuous generation.
        
        Args:
            binary: Binary matrix (num_units, num_periods, num_bits)
            
        Returns:
            Generation matrix (num_units, num_periods)
        """
        num_periods = binary.shape[1]
        p = np.zeros((self.num_units, num_periods))
        
        for i in range(self.num_units):
            for t in range(num_periods):
                # Convert binary to integer
                p_int = 0
                for k in range(self.num_bits):
                    p_int += binary[i, t, k] * (2**k)
                
                # Convert back to generation: p = P_min + p_int * delta
                p[i, t] = self.P_min[i] + p_int * self.delta[i]
        
        return p
    
    def get_generation_value(self, i: int, t: int, binary_vars: Dict[str, int]) -> float:
        """
        Get generation value from binary variables.
        
        Args:
            i: Unit index
            t: Time period
            binary_vars: Dictionary mapping variable names to values
            
        Returns:
            Generation value (MW)
        """
        if self.num_bits == 1:
            # For 1-bit: bit=0 -> P_min, bit=1 -> P_max
            var_name = f"p_{i}_{t}_0"
            bit_value = binary_vars.get(var_name, 0)
            if bit_value == 1:
                return self.P_max[i]
            else:
                return self.P_min[i]
        else:
            # Multi-bit encoding: p = P_min + p_int * delta
            # where p_int = bit0 + 2*bit1 + 4*bit2 + ...
            p_int = 0
            for k in range(self.num_bits):
                var_name = f"p_{i}_{t}_{k}"
                bit_value = binary_vars.get(var_name, 0)
                p_int += bit_value * (2**k)
            
            # Clamp p_int to valid range [0, 2^num_bits - 1]
            max_int = 2**self.num_bits - 1
            p_int = min(p_int, max_int)
            
            return self.P_min[i] + p_int * self.delta[i]
    
    def get_variable_name(self, i: int, t: int, k: int) -> str:
        """Get variable name for binary variable."""
        return f"p_{i}_{t}_{k}"
    
    def get_variable_index(self, i: int, t: int, k: int, num_periods: int) -> int:
        """
        Get linear index for binary variable.
        
        Args:
            i: Unit index
            t: Time period
            k: Bit index
            num_periods: Number of time periods
            
        Returns:
            Linear index
        """
        return (i * num_periods * self.num_bits) + (t * self.num_bits) + k
    
    def get_coefficient_for_bit(self, i: int, k: int) -> float:
        """
        Get coefficient for bit k in generation representation.
        
        Args:
            i: Unit index
            k: Bit index
            
        Returns:
            Coefficient (MW)
        """
        return (2**k) * self.delta[i]

