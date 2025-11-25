"""
Problem 3: QUBO Transformation and Quantum Solution
====================================================

This package implements the transformation of the UC model from Problem 2 
into a QUBO representation and solves it using the Kaiwu SDK.

Modules:
- data_loader: Load and prepare problem data
- discretization: Convert continuous variables to binary variables
- qubo_builder: Build QUBO model from UC constraints
- constraint_handler: Convert constraints to penalty terms
- solver: Solve QUBO using Kaiwu SDK
- verifier: Verify solution quality and constraint satisfaction
"""

__version__ = "1.0.0"

