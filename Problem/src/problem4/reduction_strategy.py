"""
Reduction Strategy Module for Problem 4
=========================================

Implements various reduction strategies to reduce QUBO problem size
while maintaining solution quality within CIM bit capacity limits.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .data_loader import ReducedDataLoader


class ReductionStrategy:
    """
    Implements reduction strategies for UC QUBO problem.
    
    Key insights from Problem 1 and Problem 2:
    - All units stay online at all periods (u_{i,t} = 1 fixed)
    - Only generation levels p_{i,t} need to be optimized
    - This significantly reduces the problem size
    """
    
    def __init__(self, data_loader: ReducedDataLoader):
        """
        Initialize reduction strategy.
        
        Args:
            data_loader: ReducedDataLoader instance
        """
        self.data = data_loader
        self.original_num_units = data_loader.num_units
        self.original_num_periods = data_loader.num_periods
        
        # Reduction parameters (will be set by apply_reduction)
        self.selected_units = None
        self.selected_periods = None
        self.num_bits = None
        self.period_mapping = None  # Maps reduced periods to original periods
    
    def apply_reduction(self, 
                       max_binary_vars: int = 200,
                       num_bits: int = 1,
                       time_aggregation: Optional[str] = None,
                       unit_selection: Optional[List[int]] = None,
                       force_no_reduction: bool = False) -> Dict:
        """
        Apply reduction strategy to meet bit capacity constraints.
        
        Args:
            max_binary_vars: Maximum number of binary variables allowed
            num_bits: Number of bits per unit per period (1 or 2)
            time_aggregation: Time aggregation method ('uniform', 'peak_valley', None)
            unit_selection: List of unit indices to keep (None = keep all)
        
        Returns:
            Dictionary with reduction information
        """
        self.num_bits = num_bits
        
        # Step 1: Select units (if specified)
        if unit_selection is not None:
            self.selected_units = unit_selection
        else:
            # Default: keep all units (but can be reduced later)
            self.selected_units = list(range(self.original_num_units))
        
        # Step 2: Aggregate time periods
        if time_aggregation is not None:
            self.selected_periods, self.period_mapping = self._aggregate_periods(
                time_aggregation, max_binary_vars
            )
        else:
            # No aggregation: use all periods
            self.selected_periods = list(range(self.original_num_periods))
            self.period_mapping = {t: t for t in range(self.original_num_periods)}
        
        # Step 3: Calculate reduced problem size
        num_reduced_units = len(self.selected_units)
        num_reduced_periods = len(self.selected_periods)
        total_binary_vars = num_reduced_units * num_reduced_periods * num_bits
        
        # Skip automatic reduction if force_no_reduction is True
        if force_no_reduction:
            # Just return the current configuration without further reduction
            return {
                'selected_units': self.selected_units,
                'selected_periods': self.selected_periods,
                'num_bits': self.num_bits,
                'total_binary_vars': total_binary_vars,
                'period_mapping': self.period_mapping,
                'reduction_ratio': {
                    'units': len(self.selected_units) / self.original_num_units,
                    'periods': len(self.selected_periods) / self.original_num_periods,
                    'total': total_binary_vars / (self.original_num_units * self.original_num_periods * 2)
                }
            }
        
        # Force reduction if still too large (before further checks)
        if total_binary_vars > max_binary_vars:
            # Calculate how many periods we can actually fit
            max_periods = max(1, max_binary_vars // (num_reduced_units * num_bits))
            if max_periods < num_reduced_periods:
                print(f"  Forcing period reduction: {num_reduced_periods} -> {max_periods}")
                self.selected_periods = self._select_key_periods(max_periods)
                self.period_mapping = self._create_period_mapping()
                num_reduced_periods = len(self.selected_periods)
                total_binary_vars = num_reduced_units * num_reduced_periods * num_bits
        
        # Step 4: If still too large, reduce further
        if total_binary_vars > max_binary_vars:
            # Reduce number of periods
            reduction_factor = max_binary_vars / total_binary_vars
            target_periods = max(1, int(num_reduced_periods * reduction_factor))
            self.selected_periods = self._select_key_periods(target_periods)
            self.period_mapping = self._create_period_mapping()
            num_reduced_periods = len(self.selected_periods)
            total_binary_vars = num_reduced_units * num_reduced_periods * num_bits
        
        # Step 5: If still too large, reduce units
        if total_binary_vars > max_binary_vars:
            reduction_factor = max_binary_vars / total_binary_vars
            target_units = max(1, int(num_reduced_units * reduction_factor))
            self.selected_units = self._select_key_units(target_units)
            num_reduced_units = len(self.selected_units)
            total_binary_vars = num_reduced_units * num_reduced_periods * num_bits
        
        # Final check
        if total_binary_vars > max_binary_vars:
            # Last resort: reduce bits
            if num_bits > 1:
                self.num_bits = 1
                total_binary_vars = num_reduced_units * num_reduced_periods * 1
        
        return {
            'selected_units': self.selected_units,
            'selected_periods': self.selected_periods,
            'num_bits': self.num_bits,
            'total_binary_vars': total_binary_vars,
            'period_mapping': self.period_mapping,
            'reduction_ratio': {
                'units': len(self.selected_units) / self.original_num_units,
                'periods': len(self.selected_periods) / self.original_num_periods,
                'total': total_binary_vars / (self.original_num_units * self.original_num_periods * 2)
            }
        }
    
    def _aggregate_periods(self, method: str, max_binary_vars: int) -> Tuple[List[int], Dict[int, int]]:
        """
        Aggregate time periods to reduce problem size.
        
        Args:
            method: Aggregation method ('uniform', 'peak_valley')
            max_binary_vars: Maximum binary variables allowed
        
        Returns:
            Tuple of (selected_periods, period_mapping)
        """
        if method == 'uniform':
            # Uniform aggregation: divide 24 periods into groups
            # Calculate target periods based on max_binary_vars constraint
            max_vars_per_period = len(self.selected_units) * self.num_bits
            target_periods = max(1, max_binary_vars // max_vars_per_period)
            target_periods = min(target_periods, self.original_num_periods)
            
            # Ensure we actually reduce if possible
            if target_periods >= self.original_num_periods and max_binary_vars < self.original_num_periods * max_vars_per_period:
                # Force reduction: calculate how many periods we can fit
                target_periods = max(1, max_binary_vars // max_vars_per_period)
            
            # Select evenly spaced periods
            step = self.original_num_periods / target_periods
            selected = []
            mapping = {}
            for i in range(target_periods):
                period_idx = int(i * step)
                if period_idx >= self.original_num_periods:
                    period_idx = self.original_num_periods - 1
                if period_idx not in selected:
                    selected.append(period_idx)
                # Map all periods in this group to the selected period
                start_period = int(i * step)
                end_period = int((i + 1) * step) if i < target_periods - 1 else self.original_num_periods
                for p in range(start_period, end_period):
                    mapping[p] = period_idx
            
            return selected, mapping
        
        elif method == 'peak_valley':
            # Peak-valley aggregation: select peak and valley periods
            load = self.data.load_demand
            sorted_indices = np.argsort(load)
            
            # Select peak periods (highest load)
            num_peaks = max(1, max_binary_vars // (len(self.selected_units) * self.num_bits * 2))
            peak_periods = sorted_indices[-num_peaks:].tolist()
            
            # Select valley periods (lowest load)
            valley_periods = sorted_indices[:num_peaks].tolist()
            
            # Combine and sort
            selected = sorted(list(set(peak_periods + valley_periods)))
            mapping = {}
            
            # Map each period to nearest selected period
            for p in range(self.original_num_periods):
                if p in selected:
                    mapping[p] = p
                else:
                    # Find nearest selected period
                    distances = [abs(p - sp) for sp in selected]
                    nearest_idx = np.argmin(distances)
                    mapping[p] = selected[nearest_idx]
            
            return selected, mapping
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def _select_key_periods(self, target_num: int) -> List[int]:
        """
        Select key periods based on load variation.
        
        Args:
            target_num: Target number of periods
        
        Returns:
            List of selected period indices
        """
        load = self.data.load_demand
        
        # Select periods with significant load changes
        load_changes = np.abs(np.diff(load))
        sorted_by_change = np.argsort(load_changes)[::-1]
        
        # Select periods around significant changes
        selected = set()
        for idx in sorted_by_change[:target_num]:
            selected.add(idx)
            selected.add(idx + 1)
        
        # Ensure we have target_num periods
        if len(selected) < target_num:
            # Add evenly spaced periods
            step = self.original_num_periods / (target_num - len(selected))
            for i in range(target_num - len(selected)):
                period = int(i * step)
                selected.add(min(period, self.original_num_periods - 1))
        
        # Limit to target_num
        selected = sorted(list(selected))[:target_num]
        return selected
    
    def _select_key_units(self, target_num: int) -> List[int]:
        """
        Select key units based on capacity and cost.
        
        Args:
            target_num: Target number of units
        
        Returns:
            List of selected unit indices
        """
        # Priority: large capacity units first
        capacities = self.data.P_max
        sorted_by_capacity = np.argsort(capacities)[::-1]
        
        selected = sorted_by_capacity[:target_num].tolist()
        return sorted(selected)
    
    def _create_period_mapping(self) -> Dict[int, int]:
        """Create mapping from original periods to reduced periods."""
        mapping = {}
        for orig_period in range(self.original_num_periods):
            # Find nearest selected period
            distances = [abs(orig_period - sp) for sp in self.selected_periods]
            nearest_idx = np.argmin(distances)
            mapping[orig_period] = self.selected_periods[nearest_idx]
        return mapping
    
    def get_reduced_data(self) -> Dict:
        """
        Get reduced data arrays.
        
        Returns:
            Dictionary with reduced data arrays
        """
        # Reduced unit parameters
        unit_indices = self.selected_units
        reduced_data = {
            'units': [self.data.units[i] for i in unit_indices],
            'P_max': self.data.P_max[unit_indices],
            'P_min': self.data.P_min[unit_indices],
            'Ramp_Up': self.data.Ramp_Up[unit_indices],
            'Ramp_Down': self.data.Ramp_Down[unit_indices],
            'a_coeff': self.data.a_coeff[unit_indices],
            'b_coeff': self.data.b_coeff[unit_indices],
            'c_coeff': self.data.c_coeff[unit_indices],
        }
        
        # Reduced load demand (aggregate if needed)
        if len(self.selected_periods) < self.original_num_periods:
            # Aggregate load for selected periods
            reduced_load = []
            reduced_reserve = []
            for period in self.selected_periods:
                # Average load over periods mapped to this period
                mapped_periods = [p for p, mapped in self.period_mapping.items() if mapped == period]
                avg_load = np.mean([self.data.load_demand[p] for p in mapped_periods])
                avg_reserve = np.mean([self.data.spinning_reserve_req[p] for p in mapped_periods])
                reduced_load.append(avg_load)
                reduced_reserve.append(avg_reserve)
            reduced_data['load_demand'] = np.array(reduced_load)
            reduced_data['spinning_reserve_req'] = np.array(reduced_reserve)
        else:
            reduced_data['load_demand'] = self.data.load_demand[self.selected_periods]
            reduced_data['spinning_reserve_req'] = self.data.spinning_reserve_req[self.selected_periods]
        
        reduced_data['num_units'] = len(unit_indices)
        reduced_data['num_periods'] = len(self.selected_periods)
        
        return reduced_data

