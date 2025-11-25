"""
Solver Module for Problem 4
============================

Solves reduced QUBO model using Kaiwu SDK.

Kaiwu SDK Documentation:
- English: https://kaiwu-sdk-docs.qboson.com/en/latest/
- Chinese: https://kaiwu-sdk-docs.qboson.com/zh/v1.2.0/source/getting_started/index.html
"""

import numpy as np
import kaiwu as kw
from typing import Dict, List, Tuple, Optional
from .discretization import GenerationDiscretizer


class ReducedQUBOSolver:
    """Solves reduced QUBO model for UC problem."""
    
    def __init__(self, qubo_model: kw.qubo.QuboModel, discretizer: GenerationDiscretizer,
                 reduced_data: Dict, optimizer_type: str = 'simulated_annealing'):
        """
        Initialize solver.
        
        Args:
            qubo_model: QUBO model to solve
            discretizer: GenerationDiscretizer instance
            reduced_data: Reduced data dictionary
            optimizer_type: Type of optimizer ('simulated_annealing' or 'cim')
        """
        self.qubo_model = qubo_model
        self.discretizer = discretizer
        self.reduced_data = reduced_data
        self.optimizer_type = optimizer_type
        
        # Create optimizer (optimized based on problem size)
        num_vars = discretizer.get_num_binary_vars(reduced_data['num_periods'])
        
        if optimizer_type == 'simulated_annealing':
            if num_vars > 100:
                # For larger reduced problems
                self.optimizer = kw.classical.SimulatedAnnealingOptimizer(
                    initial_temperature=3e6,
                    alpha=0.97,
                    cutoff_temperature=0.001,
                    iterations_per_t=600,
                    size_limit=80
                )
            else:
                # For smaller reduced problems
                self.optimizer = kw.classical.SimulatedAnnealingOptimizer(
                    initial_temperature=2e6,
                    alpha=0.95,
                    cutoff_temperature=0.01,
                    iterations_per_t=400,
                    size_limit=50
                )
        elif optimizer_type == 'cim':
            # CIM optimizer (requires hardware connection)
            # Wrap with PrecisionReducer to handle 8-bit limit
            base_optimizer = kw.cim.CIMOptimizer()
            self.optimizer = kw.preprocess.PrecisionReducer(base_optimizer, bit_width=8)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Create solver
        self.solver = kw.solver.SimpleSolver(self.optimizer)
    
    def solve(self, num_solutions: int = 3) -> List[Dict]:
        """
        Solve QUBO model.
        
        Args:
            num_solutions: Number of solutions to return
            
        Returns:
            List of solution dictionaries
        """
        print(f"\nSolving reduced QUBO model using {self.optimizer_type}...")
        
        # Get QUBO matrix
        qubo_matrix = self.qubo_model.get_matrix()
        print(f"QUBO matrix shape: {qubo_matrix.shape}")
        print(f"QUBO matrix size: {qubo_matrix.size}")
        
        # Check bit width (CIM hardware limit: 8-bit INT [-128, 127])
        try:
            kw.qubo.check_qubo_matrix_bit_width(qubo_matrix, bit_width=8)
            print("✓ QUBO matrix passes bit width check (8-bit)")
        except ValueError as e:
            print(f"Warning: QUBO matrix bit width issue: {e}")
            print("Attempting to adjust precision...")
            try:
                qubo_matrix = kw.qubo.adjust_qubo_matrix_precision(qubo_matrix, bit_width=8)
                print("✓ QUBO matrix precision adjusted")
                
                # Verify adjustment worked
                try:
                    kw.qubo.check_qubo_matrix_bit_width(qubo_matrix, bit_width=8)
                    print("✓ Adjusted matrix passes bit width check")
                except ValueError as e2:
                    print(f"⚠ Warning: Adjusted matrix still fails bit width check: {e2}")
                    print("  Matrix coefficients may have been scaled too aggressively")
            except Exception as adjust_error:
                print(f"✗ Error adjusting precision: {adjust_error}")
                print("  This may indicate the problem is too large for CIM hardware")
                raise
        
        # Solve
        solutions = []
        
        for i in range(num_solutions):
            try:
                result = self.solver.solve_qubo(self.qubo_model)
                # SimpleSolver.solve_qubo returns (sol_dict, objective_value) tuple
                if isinstance(result, tuple) and len(result) == 2:
                    sol_dict, objective = result
                    solutions.append({'sol_dict': sol_dict, 'objective': objective})
                elif isinstance(result, dict):
                    solutions.append(result)
                else:
                    solutions.append({'sol_dict': result, 'objective': None})
            except Exception as e:
                print(f"  Warning: Failed to get solution {i+1}: {e}")
                if i == 0:
                    raise
                break
        
        print(f"✓ Found {len(solutions)} solution(s)")
        
        return solutions
    
    def extract_generation(self, solution) -> np.ndarray:
        """
        Extract generation values from solution.
        
        Args:
            solution: Solution from solver
            
        Returns:
            Generation matrix (num_units, num_periods) for reduced problem
        """
        num_units = self.discretizer.num_units
        num_periods = self.reduced_data['num_periods']
        
        # Get solution dictionary
        sol_dict = None
        
        if isinstance(solution, tuple):
            sol_dict = solution[0]
        elif isinstance(solution, dict):
            if 'sol_dict' in solution:
                sol_dict = solution['sol_dict']
            else:
                sol_dict = solution
        else:
            sol_dict = solution
        
        if not isinstance(sol_dict, dict):
            raise ValueError(f"Expected dict or tuple, got {type(sol_dict)}")
        
        # Extract binary variables and convert to generation
        generation = np.zeros((num_units, num_periods))
        
        for i in range(num_units):
            for t in range(num_periods):
                generation[i, t] = self.discretizer.get_generation_value(i, t, sol_dict)
        
        return generation
    
    def calculate_objective(self, generation: np.ndarray) -> float:
        """
        Calculate objective function value from generation.
        
        Args:
            generation: Generation matrix (num_units, num_periods)
            
        Returns:
            Objective value (total cost)
        """
        total_cost = 0.0
        
        for i in range(self.discretizer.num_units):
            for t in range(self.reduced_data['num_periods']):
                p = generation[i, t]
                fuel_cost = (self.reduced_data['a_coeff'][i] * p**2 + 
                            self.reduced_data['b_coeff'][i] * p + 
                            self.reduced_data['c_coeff'][i])
                total_cost += fuel_cost
        
        return total_cost

