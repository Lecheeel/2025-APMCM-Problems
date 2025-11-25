"""
Solver Module
=============

Solves QUBO model using Kaiwu SDK.
"""

import numpy as np
import kaiwu as kw
from typing import Dict, List, Tuple, Optional
from .qubo_builder import QUBOBuilder
from .discretization import GenerationDiscretizer


class QUBOSolver:
    """Solves QUBO model for UC problem."""
    
    def __init__(self, qubo_model: kw.qubo.QuboModel, discretizer: GenerationDiscretizer,
                 data_loader, optimizer_type: str = 'simulated_annealing'):
        """
        Initialize solver.
        
        Args:
            qubo_model: QUBO model to solve
            discretizer: GenerationDiscretizer instance
            data_loader: UCDataLoader instance
            optimizer_type: Type of optimizer ('simulated_annealing' or 'cim')
        """
        self.qubo_model = qubo_model
        self.discretizer = discretizer
        self.data_loader = data_loader
        self.optimizer_type = optimizer_type
        
        # Create optimizer (optimized based on problem size)
        num_vars = discretizer.get_num_binary_vars(data_loader.num_periods)
        
        if optimizer_type == 'simulated_annealing':
            if num_vars > 200:
                # For larger problems (2-bit), balance quality and speed
                # Reduced iterations to avoid long runtime while maintaining quality
                self.optimizer = kw.classical.SimulatedAnnealingOptimizer(
                    initial_temperature=3e7,  # Moderate temperature
                    alpha=0.98,  # Slower cooling for better convergence
                    cutoff_temperature=0.001,  # Reasonable cutoff
                    iterations_per_t=800,  # Reduced from 2000 to improve speed
                    size_limit=100  # Reduced from 200 to improve speed
                )
            else:
                # For smaller problems (1-bit), optimize for speed
                self.optimizer = kw.classical.SimulatedAnnealingOptimizer(
                    initial_temperature=3e6,  # Moderate temperature
                    alpha=0.96,  # Faster cooling for speed
                    cutoff_temperature=0.01,  # Higher cutoff for speed
                    iterations_per_t=500,  # Moderate iterations
                    size_limit=50  # Smaller solution space for speed
                )
        elif optimizer_type == 'cim':
            # CIM optimizer (requires hardware connection)
            self.optimizer = kw.cim.CIMOptimizer()
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Create solver
        self.solver = kw.solver.SimpleSolver(self.optimizer)
    
    def solve(self, num_solutions: int = 1) -> List[Dict]:
        """
        Solve QUBO model.
        
        Args:
            num_solutions: Number of solutions to return
            
        Returns:
            List of solution dictionaries
        """
        print(f"\nSolving QUBO model using {self.optimizer_type}...")
        
        # Make QUBO model
        qubo_expr = self.qubo_model.make()
        
        # Get QUBO matrix
        qubo_matrix = self.qubo_model.get_matrix()
        print(f"QUBO matrix shape: {qubo_matrix.shape}")
        print(f"QUBO matrix size: {qubo_matrix.size}")
        
        # Check bit width
        try:
            kw.qubo.check_qubo_matrix_bit_width(qubo_matrix, bit_width=8)
            print("✓ QUBO matrix passes bit width check (8-bit)")
        except ValueError as e:
            print(f"Warning: QUBO matrix bit width issue: {e}")
            print("Attempting to adjust precision...")
            qubo_matrix = kw.qubo.adjust_qubo_matrix_precision(qubo_matrix, bit_width=8)
            print("✓ QUBO matrix precision adjusted")
        
        # Solve - SimpleSolver doesn't have solve_qubo_multi_results
        # So we call solve_qubo multiple times to get multiple solutions
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
                    # Fallback: assume it's a sol_dict directly
                    solutions.append({'sol_dict': result, 'objective': None})
            except Exception as e:
                print(f"  Warning: Failed to get solution {i+1}: {e}")
                if i == 0:
                    # If first solution fails, raise the error
                    raise
                # Otherwise, continue with solutions we have
                break
        
        print(f"✓ Found {len(solutions)} solution(s)")
        
        return solutions
    
    def extract_generation(self, solution) -> np.ndarray:
        """
        Extract generation values from solution.
        
        Args:
            solution: Solution from solver (can be dict, tuple, or sol_dict directly)
            
        Returns:
            Generation matrix (num_units, num_periods)
        """
        num_units = self.discretizer.num_units
        num_periods = self.data_loader.num_periods
        
        # Get solution dictionary - handle different return formats
        sol_dict = None
        
        if isinstance(solution, tuple):
            # Tuple format: (sol_dict, objective_value)
            sol_dict = solution[0]
        elif isinstance(solution, dict):
            if 'sol_dict' in solution:
                sol_dict = solution['sol_dict']
            else:
                # Assume it's already a sol_dict
                sol_dict = solution
        else:
            # Try to use solution directly as sol_dict
            sol_dict = solution
        
        # Ensure sol_dict is a dictionary
        if not isinstance(sol_dict, dict):
            raise ValueError(f"Expected dict or tuple, got {type(sol_dict)}: {sol_dict}")
        
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
            for t in range(self.data_loader.num_periods):
                p = generation[i, t]
                fuel_cost = (self.data_loader.a_coeff[i] * p**2 + 
                            self.data_loader.b_coeff[i] * p + 
                            self.data_loader.c_coeff[i])
                total_cost += fuel_cost
        
        return total_cost

