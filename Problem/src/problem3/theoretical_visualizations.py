"""
Theoretical Visualizations for Problem 3
==========================================
Generates high-quality 2D principle diagrams and flowcharts for QUBO transformation.
These diagrams illustrate the theoretical foundations and transformation process.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as path_effects
from typing import Dict, List, Optional, Tuple
import os
import json


def create_theoretical_visualizations(results_dir: str, timestamp: Optional[str] = None):
    """
    Generate theoretical principle diagrams and flowcharts.
    
    Args:
        results_dir: Directory containing Problem 3 results
        timestamp: Specific timestamp to use (if None, uses latest)
    """
    print("\n" + "=" * 70)
    print("Generating Theoretical Principle Diagrams...")
    print("=" * 70)
    
    # Load data if available
    try:
        if timestamp is None:
            summary_files = [f for f in os.listdir(results_dir) 
                           if f.startswith('summary_') and f.endswith('.json')]
            if summary_files:
                latest_file = max(summary_files, key=lambda f: 
                                 os.path.getmtime(os.path.join(results_dir, f)))
                timestamp = latest_file.replace('summary_', '').replace('.json', '')
        
        summary_path = os.path.join(results_dir, f"summary_{timestamp}.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            opt_info = summary_data['optimization_info']
            num_units = opt_info['num_units']
            num_periods = opt_info['num_periods']
            num_bits = opt_info.get('num_bits_per_unit', 2)
        else:
            # Use defaults
            num_units = 6
            num_periods = 24
            num_bits = 2
    except Exception as e:
        print(f"Warning: Could not load results, using defaults: {e}")
        num_units = 6
        num_periods = 24
        num_bits = 2
    
    # Set style for publication-quality figures
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.0,
    })
    
    timestamp_str = timestamp if timestamp else "latest"
    output_dir = results_dir
    
    # ============================================================================
    # Diagram 1: QUBO Transformation Flowchart
    # ============================================================================
    print("Creating Diagram 1: QUBO Transformation Flowchart...")
    create_qubo_transformation_flowchart(output_dir, timestamp_str)
    
    # ============================================================================
    # Diagram 2: Discretization Principle Diagram
    # ============================================================================
    print("Creating Diagram 2: Discretization Principle Diagram...")
    create_discretization_principle_diagram(output_dir, timestamp_str, num_bits)
    
    # ============================================================================
    # Diagram 3: Constraint to Penalty Conversion Diagram
    # ============================================================================
    print("Creating Diagram 3: Constraint to Penalty Conversion Diagram...")
    create_constraint_penalty_diagram(output_dir, timestamp_str)
    
    # ============================================================================
    # Diagram 4: QUBO Matrix Structure Diagram
    # ============================================================================
    print("Creating Diagram 4: QUBO Matrix Structure Diagram...")
    create_qubo_matrix_structure_diagram(output_dir, timestamp_str, num_units, num_periods, num_bits)
    
    # ============================================================================
    # Diagram 5: Binary Variable Mapping Diagram
    # ============================================================================
    print("Creating Diagram 5: Binary Variable Mapping Diagram...")
    create_binary_variable_mapping_diagram(output_dir, timestamp_str, num_units, num_periods, num_bits)
    
    # ============================================================================
    # Diagram 6: Objective Function Construction Diagram
    # ============================================================================
    print("Creating Diagram 6: Objective Function Construction Diagram...")
    create_objective_function_diagram(output_dir, timestamp_str)
    
    # ============================================================================
    # Diagram 7: Penalty Term Construction Diagram
    # ============================================================================
    print("Creating Diagram 7: Penalty Term Construction Diagram...")
    create_penalty_term_diagram(output_dir, timestamp_str)
    
    # ============================================================================
    # Diagram 8: Solving Process Flowchart
    # ============================================================================
    print("Creating Diagram 8: Solving Process Flowchart...")
    create_solving_process_flowchart(output_dir, timestamp_str)
    
    # ============================================================================
    # Diagram 9: Discretization Error Analysis Diagram
    # ============================================================================
    print("Creating Diagram 9: Discretization Error Analysis Diagram...")
    create_discretization_error_diagram(output_dir, timestamp_str, num_bits)
    
    # ============================================================================
    # Diagram 10: Complete System Architecture Diagram
    # ============================================================================
    print("Creating Diagram 10: Complete System Architecture Diagram...")
    create_system_architecture_diagram(output_dir, timestamp_str)
    
    print("\n" + "=" * 70)
    print("All theoretical visualizations generated successfully!")
    print(f"Results directory: {output_dir}")
    print("=" * 70)


def create_qubo_transformation_flowchart(output_dir: str, timestamp: str):
    """Create flowchart showing QUBO transformation process."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    color_start = '#2ECC71'  # Green
    color_process = '#3498DB'  # Blue
    color_transform = '#E74C3C'  # Red
    color_output = '#9B59B6'  # Purple
    
    # Start box
    start_box = FancyBboxPatch((3.5, 10.5), 3, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_start, edgecolor='black', linewidth=2)
    ax.add_patch(start_box)
    ax.text(5, 11, 'UC Problem\n(Problem 2)', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='white')
    
    # Step 1: Discretization
    step1_box = FancyBboxPatch((1, 8.5), 3, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_process, edgecolor='black', linewidth=2)
    ax.add_patch(step1_box)
    ax.text(2.5, 9.1, 'Step 1:\nDiscretization', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='white')
    
    # Step 2: Constraint Conversion
    step2_box = FancyBboxPatch((4, 8.5), 3, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_process, edgecolor='black', linewidth=2)
    ax.add_patch(step2_box)
    ax.text(5.5, 9.1, 'Step 2:\nConstraint\nConversion', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='white')
    
    # Step 3: QUBO Construction
    step3_box = FancyBboxPatch((7, 8.5), 3, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_process, edgecolor='black', linewidth=2)
    ax.add_patch(step3_box)
    ax.text(8.5, 9.1, 'Step 3:\nQUBO Matrix\nConstruction', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='white')
    
    # Transformation details
    transform_box = FancyBboxPatch((2, 6), 6, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=color_transform, edgecolor='black', linewidth=2)
    ax.add_patch(transform_box)
    ax.text(5, 6.75, 'QUBO Form: min x^T Q x + c^T x', ha='center', va='center', 
           fontsize=13, fontweight='bold', color='white')
    
    # Details boxes
    detail1 = FancyBboxPatch((0.5, 4), 2.5, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#ECF0F1', edgecolor='black', linewidth=1.5)
    ax.add_patch(detail1)
    ax.text(1.75, 4.75, 'Binary Variables:\nx_{i,t,k} ∈ {0,1}', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    detail2 = FancyBboxPatch((3.75, 4), 2.5, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#ECF0F1', edgecolor='black', linewidth=1.5)
    ax.add_patch(detail2)
    ax.text(5, 4.75, 'Penalty Terms:\nλ × (violation)²', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    detail3 = FancyBboxPatch((7, 4), 2.5, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#ECF0F1', edgecolor='black', linewidth=1.5)
    ax.add_patch(detail3)
    ax.text(8.25, 4.75, 'Q Matrix:\nSymmetric\nN×N', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output box
    output_box = FancyBboxPatch((3.5, 1.5), 3, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_output, edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 2, 'QUBO Model\nReady for Solving', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='white')
    
    # Arrows
    arrows = [
        ((5, 10.5), (5, 9.7)),  # Start to steps
        ((2.5, 9.1), (4, 9.1)),  # Step1 to Step2
        ((5.5, 9.1), (7, 9.1)),  # Step2 to Step3
        ((2.5, 8.5), (2, 7.5)),  # Step1 to transform
        ((5.5, 8.5), (5, 7.5)),  # Step2 to transform
        ((8.5, 8.5), (8, 7.5)),  # Step3 to transform
        ((1.75, 6), (1.75, 5.5)),  # Transform to detail1
        ((5, 6), (5, 5.5)),  # Transform to detail2
        ((8.25, 6), (8.25, 5.5)),  # Transform to detail3
        ((5, 4), (5, 2.5)),  # Details to output
    ]
    
    for (start, end) in arrows:
        arrow = FancyArrowPatch(start, end, 
                               arrowstyle='->', lw=2.5, 
                               color='#34495E', zorder=1)
        ax.add_patch(arrow)
    
    ax.set_title('QUBO Transformation Flowchart', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    path = os.path.join(output_dir, f"theoretical_1_qubo_flowchart_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()


def create_discretization_principle_diagram(output_dir: str, timestamp: str, num_bits: int):
    """Create diagram showing discretization principle."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: Continuous to discrete mapping
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Title
    ax1.text(5, 9.5, f'{num_bits}-bit Discretization Principle', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Continuous range
    ax1.add_patch(Rectangle((1, 6), 8, 1.5, facecolor='#E8F8F5', 
                           edgecolor='#1ABC9C', linewidth=2))
    ax1.text(5, 6.75, 'Continuous Range: [P_min, P_max]', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Discrete values
    num_levels = 2 ** num_bits
    step = 7 / (num_levels - 1) if num_levels > 1 else 0
    
    for i in range(num_levels):
        x_pos = 1.5 + i * step
        # Calculate value
        if num_bits == 1:
            value = i * 1  # 0 or 1
        else:
            value = i / (num_levels - 1) if num_levels > 1 else 0
        
        # Draw discrete point
        circle = Circle((x_pos, 4), 0.3, facecolor='#E74C3C', 
                       edgecolor='black', linewidth=2, zorder=3)
        ax1.add_patch(circle)
        
        # Binary representation
        binary_str = format(i, f'0{num_bits}b')
        ax1.text(x_pos, 3.2, binary_str, ha='center', va='center', 
                fontsize=11, fontweight='bold')
        
        # Value label
        if num_bits == 1:
            label = 'P_min' if i == 0 else 'P_max'
        else:
            label = f'P_min+{i}δ' if i < num_levels - 1 else 'P_max'
        ax1.text(x_pos, 2.5, label, ha='center', va='center', 
                fontsize=9, style='italic')
    
    # Arrow from continuous to discrete
    arrow = FancyArrowPatch((5, 6), (5, 4.3), 
                           arrowstyle='->', lw=3, color='#E74C3C')
    ax1.add_patch(arrow)
    ax1.text(6.5, 5, 'Discretization', ha='left', va='center', 
            fontsize=11, fontweight='bold', color='#E74C3C')
    
    # Formula
    formula_box = FancyBboxPatch((0.5, 0.5), 9, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFF9E6', edgecolor='#F39C12', linewidth=2)
    ax1.add_patch(formula_box)
    if num_bits == 1:
        formula = 'p = P_min + bit × (P_max - P_min)'
    elif num_bits == 2:
        formula = 'p = P_min + (bit₀ + 2×bit₁) × δ\nwhere δ = (P_max - P_min) / 3'
    else:
        formula = f'p = P_min + Σ(2ᵏ × bitₖ) × δ\nwhere δ = (P_max - P_min) / (2^{num_bits} - 1)'
    ax1.text(5, 1.25, formula, ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # Right: Example with actual values
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    ax2.text(5, 9.5, 'Example: Unit 1 (P_min=50MW, P_max=300MW)', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Draw continuous line
    x_cont = np.linspace(1, 9, 100)
    y_cont = 7 + 1.5 * (x_cont - 1) / 8
    ax2.plot(x_cont, y_cont, 'b-', linewidth=3, alpha=0.5, label='Continuous')
    
    # Draw discrete points
    delta = (300 - 50) / (num_levels - 1) if num_levels > 1 else 250
    for i in range(num_levels):
        x_pos = 1 + i * 8 / (num_levels - 1) if num_levels > 1 else 5
        y_pos = 7 + i * 1.5 / (num_levels - 1) if num_levels > 1 else 7
        value = 50 + i * delta
        
        circle = Circle((x_pos, y_pos), 0.25, facecolor='#E74C3C', 
                       edgecolor='black', linewidth=2, zorder=3)
        ax2.add_patch(circle)
        
        binary_str = format(i, f'0{num_bits}b')
        ax2.text(x_pos, y_pos - 0.6, binary_str, ha='center', va='center', 
                fontsize=10, fontweight='bold')
        ax2.text(x_pos, y_pos - 1.2, f'{value:.1f}MW', ha='center', va='center', 
                fontsize=9)
    
    # Axis labels
    ax2.text(5, 5.5, 'Generation Level (MW)', ha='center', va='center', 
            fontsize=12, fontweight='bold', rotation=90)
    ax2.text(5, 0.5, 'Binary Encoding', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # Step size annotation
    if num_bits == 2:
        ax2.annotate('', xy=(2.5, 7.5), xytext=(2.5, 7), 
                    arrowprops=dict(arrowstyle='<->', lw=2, color='#27AE60'))
        ax2.text(2.8, 7.25, f'δ = {delta:.1f}MW', ha='left', va='center', 
                fontsize=10, fontweight='bold', color='#27AE60')
    
    plt.suptitle('Discretization Principle: Continuous to Binary Encoding', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"theoretical_2_discretization_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()


def create_constraint_penalty_diagram(output_dir: str, timestamp: str):
    """Create diagram showing constraint to penalty conversion."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    ax.text(6, 13, 'Constraint to Penalty Conversion', 
           ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Constraint types
    constraints = [
        ('Power Balance', 'Σᵢ pᵢₜ = Dₜ', 'λ_PB × (Σᵢ pᵢₜ - Dₜ)²', 1, '#E74C3C'),
        ('Ramp Up', 'pᵢₜ - pᵢₜ₋₁ ≤ RampUpᵢ', 'λ_Ramp × max(0, pᵢₜ - pᵢₜ₋₁ - RampUpᵢ)²', 2, '#3498DB'),
        ('Ramp Down', 'pᵢₜ₋₁ - pᵢₜ ≤ RampDownᵢ', 'λ_Ramp × max(0, pᵢₜ₋₁ - pᵢₜ - RampDownᵢ)²', 3, '#3498DB'),
        ('Reserve', 'Σᵢ(P_maxᵢ - pᵢₜ) ≥ Rₜ', 'λ_Res × max(0, Rₜ - Σᵢ(P_maxᵢ - pᵢₜ))²', 4, '#9B59B6'),
        ('N-1 Security', 'Σⱼ≠ᵢ P_maxⱼ ≥ Dₜ + Rₜ', 'λ_N1 × max(0, Dₜ + Rₜ - Σⱼ≠ᵢ P_maxⱼ)²', 5, '#F39C12'),
    ]
    
    y_start = 11
    y_spacing = 2
    
    for idx, (name, constraint, penalty, row, color) in enumerate(constraints):
        y_pos = y_start - idx * y_spacing
        
        # Constraint box
        constraint_box = FancyBboxPatch((0.5, y_pos - 0.6), 4.5, 1.2, 
                                       boxstyle="round,pad=0.1", 
                                       facecolor='#ECF0F1', edgecolor='black', linewidth=2)
        ax.add_patch(constraint_box)
        ax.text(2.75, y_pos, f'{name} Constraint', ha='center', va='center', 
               fontsize=12, fontweight='bold')
        ax.text(2.75, y_pos - 0.3, constraint, ha='center', va='center', 
               fontsize=10, style='italic')
        
        # Arrow
        arrow = FancyArrowPatch((5, y_pos), (7, y_pos), 
                               arrowstyle='->', lw=3, color=color)
        ax.add_patch(arrow)
        ax.text(6, y_pos + 0.4, 'Convert to', ha='center', va='center', 
               fontsize=9, style='italic', color=color)
        
        # Penalty box
        penalty_box = FancyBboxPatch((7, y_pos - 0.6), 4.5, 1.2, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(penalty_box)
        ax.text(9.25, y_pos, f'{name} Penalty', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='white')
        ax.text(9.25, y_pos - 0.3, penalty, ha='center', va='center', 
               fontsize=9, color='white', style='italic')
    
    # Penalty coefficients box
    coeff_box = FancyBboxPatch((1, 0.5), 10, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#FFF9E6', edgecolor='#F39C12', linewidth=2)
    ax.add_patch(coeff_box)
    ax.text(6, 1.25, 'Typical Penalty Coefficients:\n' +
           'λ_PB ≈ 10⁵, λ_Ramp ≈ 10⁴, λ_Res ≈ 10⁻¹, λ_N1 ≈ 10⁻¹', 
           ha='center', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"theoretical_3_constraint_penalty_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()


def create_qubo_matrix_structure_diagram(output_dir: str, timestamp: str, 
                                        num_units: int, num_periods: int, num_bits: int):
    """Create diagram showing QUBO matrix structure."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    # Left: Matrix structure
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    ax1.text(5, 9.5, 'QUBO Matrix Structure', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    # Draw matrix representation
    matrix_size = num_units * num_periods * num_bits
    matrix_box = FancyBboxPatch((2, 4), 6, 4, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E8F8F5', edgecolor='#1ABC9C', linewidth=3)
    ax1.add_patch(matrix_box)
    
    # Draw grid lines
    for i in range(4):
        # Horizontal lines
        ax1.plot([2, 8], [5 + i, 5 + i], 'k-', linewidth=0.5, alpha=0.3)
        # Vertical lines
        ax1.plot([2 + i * 1.5, 2 + i * 1.5], [4, 8], 'k-', linewidth=0.5, alpha=0.3)
    
    ax1.text(5, 6, f'Q Matrix\n{matrix_size} × {matrix_size}', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(5, 5.3, f'Symmetric Matrix\nQᵢⱼ = Qⱼᵢ', 
            ha='center', va='center', fontsize=11, style='italic')
    
    # Variable mapping
    var_box = FancyBboxPatch((1, 1.5), 8, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#FFF9E6', edgecolor='#F39C12', linewidth=2)
    ax1.add_patch(var_box)
    ax1.text(5, 2.25, f'Variables: xᵢₜₖ (i={num_units} units, t={num_periods} periods, k={num_bits} bits)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(5, 1.8, f'Total: {matrix_size} binary variables', 
            ha='center', va='center', fontsize=10)
    
    # Right: Matrix blocks
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    ax2.text(5, 9.5, 'Matrix Block Structure', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    # Draw block structure
    block_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']
    
    # Objective blocks (diagonal)
    for i in range(min(3, num_units)):
        x = 1 + i * 2.5
        y = 6.5
        block = Rectangle((x, y), 1.5, 1.5, facecolor=block_colors[i % len(block_colors)], 
                         edgecolor='black', linewidth=2)
        ax2.add_patch(block)
        ax2.text(x + 0.75, y + 0.75, f'Obj\n{i+1}', ha='center', va='center', 
                fontsize=9, fontweight='bold', color='white')
    
    # Constraint blocks (off-diagonal)
    for i in range(min(2, num_units)):
        for j in range(min(2, num_periods)):
            if i != j:
                x = 1 + i * 2.5
                y = 4 - j * 1.5
                block = Rectangle((x, y), 1.5, 1.5, facecolor='#95A5A6', 
                                 edgecolor='black', linewidth=1)
                ax2.add_patch(block)
                ax2.text(x + 0.75, y + 0.75, 'Pen', ha='center', va='center', 
                        fontsize=8, style='italic')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=block_colors[0], label='Objective Terms'),
        mpatches.Patch(facecolor='#95A5A6', label='Penalty Terms'),
    ]
    ax2.legend(handles=legend_elements, loc='lower center', fontsize=10)
    
    # Formula
    formula_box = FancyBboxPatch((1, 0.5), 8, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#ECF0F1', edgecolor='black', linewidth=2)
    ax2.add_patch(formula_box)
    ax2.text(5, 1, 'QUBO Form: min x^T Q x + c^T x', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.suptitle('QUBO Matrix Structure and Block Decomposition', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"theoretical_4_qubo_matrix_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()


def create_binary_variable_mapping_diagram(output_dir: str, timestamp: str, 
                                          num_units: int, num_periods: int, num_bits: int):
    """Create diagram showing binary variable mapping."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    ax.text(6, 11, 'Binary Variable Mapping', ha='center', va='center', 
           fontsize=18, fontweight='bold')
    
    # Draw unit-period grid
    unit_width = 1.5
    period_width = 0.8
    start_x = 1
    start_y = 8
    
    # Draw grid
    for i in range(min(3, num_units)):
        for t in range(min(6, num_periods)):
            x = start_x + t * period_width
            y = start_y - i * unit_width
            
            # Draw cell
            cell = Rectangle((x, y - 0.6), period_width, 0.6, 
                           facecolor='#E8F8F5', edgecolor='#1ABC9C', linewidth=1)
            ax.add_patch(cell)
            
            # Draw bits
            for k in range(num_bits):
                bit_x = x + k * (period_width / num_bits)
                bit_cell = Rectangle((bit_x, y - 0.6), period_width / num_bits, 0.6, 
                                   facecolor='#FFF9E6', edgecolor='#F39C12', linewidth=0.5)
                ax.add_patch(bit_cell)
                ax.text(bit_x + period_width / (2 * num_bits), y - 0.3, 
                       f'x_{i}_{t}_{k}', ha='center', va='center', 
                       fontsize=7, fontweight='bold')
    
    # Labels
    ax.text(0.5, start_y - 1, 'Unit', ha='center', va='center', 
           fontsize=12, fontweight='bold', rotation=90)
    ax.text(start_x + 3 * period_width, start_y + 0.5, 'Time Period', 
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Mapping formula
    formula_box = FancyBboxPatch((1, 4), 10, 2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#ECF0F1', edgecolor='black', linewidth=2)
    ax.add_patch(formula_box)
    ax.text(6, 5.5, 'Variable Index Mapping:', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(6, 4.8, f'idx = i × T × K + t × K + k', 
           ha='center', va='center', fontsize=11, style='italic')
    ax.text(6, 4.3, f'where i ∈ [0, {num_units-1}], t ∈ [0, {num_periods-1}], k ∈ [0, {num_bits-1}]', 
           ha='center', va='center', fontsize=10)
    
    # Example
    example_box = FancyBboxPatch((1, 1), 10, 2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFF9E6', edgecolor='#F39C12', linewidth=2)
    ax.add_patch(example_box)
    ax.text(6, 2.5, 'Example: Unit 1, Period 5, Bit 0', 
           ha='center', va='center', fontsize=11, fontweight='bold')
    example_idx = 1 * num_periods * num_bits + 5 * num_bits + 0
    ax.text(6, 1.8, f'Variable: x_1_5_0 → Index: {example_idx}', 
           ha='center', va='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"theoretical_5_binary_mapping_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()


def create_objective_function_diagram(output_dir: str, timestamp: str):
    """Create diagram showing objective function construction."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    ax.text(6, 11, 'Objective Function Construction', ha='center', va='center', 
           fontsize=18, fontweight='bold')
    
    # Fuel cost components
    components = [
        ('Quadratic Term', 'aᵢ × p²ᵢₜ', 9, '#E74C3C'),
        ('Linear Term', 'bᵢ × pᵢₜ', 7, '#3498DB'),
        ('Constant Term', 'cᵢ', 5, '#2ECC71'),
    ]
    
    for idx, (name, formula, y_pos, color) in enumerate(components):
        # Component box
        comp_box = FancyBboxPatch((1, y_pos - 0.6), 4, 1.2, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(comp_box)
        ax.text(3, y_pos, name, ha='center', va='center', 
               fontsize=12, fontweight='bold', color='white')
        ax.text(3, y_pos - 0.3, formula, ha='center', va='center', 
               fontsize=10, color='white', style='italic')
        
        # Plus sign
        if idx < len(components) - 1:
            ax.text(5.5, y_pos, '+', ha='center', va='center', 
                   fontsize=20, fontweight='bold')
        
        # Arrow to sum
        arrow = FancyArrowPatch((5, y_pos), (7, 8), 
                               arrowstyle='->', lw=2, color=color)
        ax.add_patch(arrow)
    
    # Sum box
    sum_box = FancyBboxPatch((7, 7), 4, 2, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#9B59B6', edgecolor='black', linewidth=3)
    ax.add_patch(sum_box)
    ax.text(9, 8.5, 'Fuel Cost', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='white')
    ax.text(9, 7.8, 'Σᵢₜ [aᵢp²ᵢₜ + bᵢpᵢₜ + cᵢ]', 
           ha='center', va='center', fontsize=11, color='white', style='italic')
    
    # Binary substitution
    sub_box = FancyBboxPatch((1, 3.5), 10, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='#FFF9E6', edgecolor='#F39C12', linewidth=2)
    ax.add_patch(sub_box)
    ax.text(6, 4.5, 'Substitute: pᵢₜ = P_minᵢ + Σₖ(2ᵏ × δᵢ × xᵢₜₖ)', 
           ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6, 3.9, 'Expand and collect terms → Quadratic form in binary variables', 
           ha='center', va='center', fontsize=10, style='italic')
    
    # Final form
    final_box = FancyBboxPatch((1, 1), 10, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#ECF0F1', edgecolor='black', linewidth=2)
    ax.add_patch(final_box)
    ax.text(6, 2, 'QUBO Objective: min x^T Q_obj x + c_obj^T x', 
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrow from sum to substitution
    arrow = FancyArrowPatch((9, 7), (6, 5), 
                           arrowstyle='->', lw=3, color='#34495E')
    ax.add_patch(arrow)
    
    # Arrow from substitution to final
    arrow = FancyArrowPatch((6, 3.5), (6, 2.5), 
                           arrowstyle='->', lw=3, color='#34495E')
    ax.add_patch(arrow)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"theoretical_6_objective_function_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()


def create_penalty_term_diagram(output_dir: str, timestamp: str):
    """Create diagram showing penalty term construction."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    penalty_types = [
        ('Power Balance', 'Σᵢ pᵢₜ = Dₜ', 'λ_PB × (Σᵢ pᵢₜ - Dₜ)²', 
         'Quadratic in p → Quadratic in x', '#E74C3C'),
        ('Ramp Constraint', '|pᵢₜ - pᵢₜ₋₁| ≤ Ramp', 'λ_Ramp × max(0, violation)²', 
         'Piecewise quadratic → Quadratic in x', '#3498DB'),
        ('Reserve Constraint', 'Σᵢ(P_max - pᵢₜ) ≥ Rₜ', 'λ_Res × max(0, Rₜ - reserve)²', 
         'Linear in p → Quadratic in x', '#9B59B6'),
        ('N-1 Constraint', 'Σⱼ≠ᵢ P_maxⱼ ≥ Dₜ + Rₜ', 'λ_N1 × max(0, violation)²', 
         'Constant check → Offset term', '#F39C12'),
    ]
    
    for idx, (name, constraint, penalty, note, color) in enumerate(penalty_types):
        ax = axes[idx]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9, name, ha='center', va='center', 
               fontsize=14, fontweight='bold')
        
        # Constraint
        constraint_box = FancyBboxPatch((1, 7), 8, 1, 
                                      boxstyle="round,pad=0.1", 
                                      facecolor='#ECF0F1', edgecolor='black', linewidth=2)
        ax.add_patch(constraint_box)
        ax.text(5, 7.5, constraint, ha='center', va='center', 
               fontsize=11, style='italic')
        
        # Arrow
        arrow = FancyArrowPatch((5, 7), (5, 5.5), 
                               arrowstyle='->', lw=3, color=color)
        ax.add_patch(arrow)
        ax.text(6.5, 6.25, 'Convert', ha='left', va='center', 
               fontsize=10, style='italic', color=color)
        
        # Penalty
        penalty_box = FancyBboxPatch((1, 4), 8, 1.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(penalty_box)
        ax.text(5, 4.9, 'Penalty Term', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='white')
        ax.text(5, 4.3, penalty, ha='center', va='center', 
               fontsize=10, color='white', style='italic')
        
        # Note
        note_box = FancyBboxPatch((1, 1.5), 8, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='#FFF9E6', edgecolor='#F39C12', linewidth=1.5)
        ax.add_patch(note_box)
        ax.text(5, 2, note, ha='center', va='center', 
               fontsize=9, style='italic')
    
    plt.suptitle('Penalty Term Construction for Each Constraint Type', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"theoretical_7_penalty_terms_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()


def create_solving_process_flowchart(output_dir: str, timestamp: str):
    """Create flowchart showing solving process."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    ax.text(6, 13, 'QUBO Solving Process', ha='center', va='center', 
           fontsize=18, fontweight='bold')
    
    # Steps
    steps = [
        ('QUBO Model', 'Q Matrix\nc Vector', 6, 11, '#2ECC71'),
        ('Matrix Check', 'Bit-width\nPrecision', 2, 9, '#3498DB'),
        ('Adjustment', 'Scale/Quantize', 6, 9, '#F39C12'),
        ('Optimizer', 'Simulated\nAnnealing', 10, 9, '#9B59B6'),
        ('Initialization', 'Random\nSolution', 2, 7, '#E74C3C'),
        ('Iteration', 'Temperature\nCooling', 6, 7, '#1ABC9C'),
        ('Solution', 'Binary\nVector', 10, 7, '#E67E22'),
        ('Decoding', 'Binary →\nGeneration', 6, 5, '#34495E'),
        ('Verification', 'Constraints\nCost', 2, 3, '#95A5A6'),
        ('Output', 'Schedule\nResults', 10, 3, '#2ECC71'),
    ]
    
    boxes = {}
    for name, detail, x, y, color in steps:
        box = FancyBboxPatch((x - 1.2, y - 0.6), 2.4, 1.2, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y + 0.2, name, ha='center', va='center', 
               fontsize=11, fontweight='bold', color='white')
        ax.text(x, y - 0.2, detail, ha='center', va='center', 
               fontsize=9, color='white')
        boxes[name] = (x, y)
    
    # Arrows
    arrows = [
        (boxes['QUBO Model'], boxes['Matrix Check']),
        (boxes['QUBO Model'], boxes['Adjustment']),
        (boxes['Matrix Check'], boxes['Adjustment']),
        (boxes['Adjustment'], boxes['Optimizer']),
        (boxes['Optimizer'], boxes['Initialization']),
        (boxes['Initialization'], boxes['Iteration']),
        (boxes['Iteration'], boxes['Solution']),
        (boxes['Solution'], boxes['Decoding']),
        (boxes['Decoding'], boxes['Verification']),
        (boxes['Decoding'], boxes['Output']),
        (boxes['Verification'], boxes['Output']),
    ]
    
    for (start, end) in arrows:
        arrow = FancyArrowPatch(start, end, 
                               arrowstyle='->', lw=2.5, 
                               color='#34495E', zorder=1)
        ax.add_patch(arrow)
    
    # Algorithm details
    algo_box = FancyBboxPatch((0.5, 0.5), 11, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='#ECF0F1', edgecolor='black', linewidth=2)
    ax.add_patch(algo_box)
    ax.text(6, 1.25, 'Simulated Annealing: T₀ → T_final, α = 0.98, iterations = 800/temp', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"theoretical_8_solving_process_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()


def create_discretization_error_diagram(output_dir: str, timestamp: str, num_bits: int):
    """Create diagram showing discretization error analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: Error sources
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    ax1.text(5, 9.5, 'Discretization Error Sources', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    # Error types
    error_types = [
        ('Quantization Error', 'Continuous → Discrete', 8, '#E74C3C'),
        ('Power Balance Error', 'Cannot match load exactly', 6, '#3498DB'),
        ('Ramp Approximation', 'Step size limitations', 4, '#9B59B6'),
    ]
    
    for name, desc, y_pos, color in error_types:
        error_box = FancyBboxPatch((1, y_pos - 0.6), 8, 1.2, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=color, edgecolor='black', linewidth=2)
        ax1.add_patch(error_box)
        ax1.text(5, y_pos + 0.2, name, ha='center', va='center', 
               fontsize=12, fontweight='bold', color='white')
        ax1.text(5, y_pos - 0.2, desc, ha='center', va='center', 
               fontsize=10, color='white', style='italic')
    
    # Error formula
    formula_box = FancyBboxPatch((1, 1.5), 8, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFF9E6', edgecolor='#F39C12', linewidth=2)
    ax1.add_patch(formula_box)
    ax1.text(5, 2.5, 'Maximum Error per Unit:', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    if num_bits == 1:
        error_formula = 'max_error = (P_max - P_min) / 2'
    else:
        error_formula = f'max_error = δ/2 = (P_max - P_min) / (2 × (2^{num_bits} - 1))'
    ax1.text(5, 1.9, error_formula, ha='center', va='center', 
            fontsize=10, style='italic')
    
    # Right: Error vs bits
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    ax2.text(5, 9.5, 'Error vs Discretization Bits', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    # Draw error curve
    bits_range = np.array([1, 2, 3, 4])
    error_relative = 1 / (2 ** bits_range - 1)  # Relative error
    
    # Scale for visualization
    y_scale = 6
    x_positions = 2 + bits_range * 1.5
    
    for i, (bits, error, x_pos) in enumerate(zip(bits_range, error_relative, x_positions)):
        y_pos = 2 + error * y_scale
        
        # Draw bar
        bar = Rectangle((x_pos - 0.4, 2), 0.8, y_pos - 2, 
                       facecolor='#E74C3C', edgecolor='black', linewidth=2)
        ax2.add_patch(bar)
        
        # Label
        ax2.text(x_pos, y_pos + 0.3, f'{error:.3f}', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
        ax2.text(x_pos, 1.2, f'{bits}-bit', ha='center', va='center', 
               fontsize=11, fontweight='bold')
    
    # Connect with line
    ax2.plot(x_positions, 2 + error_relative * y_scale, 'b--', linewidth=2, alpha=0.5)
    
    # Axis labels
    ax2.text(5, 0.5, 'Number of Bits', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax2.text(0.5, 5, 'Relative Error', ha='center', va='center', 
            fontsize=12, fontweight='bold', rotation=90)
    
    plt.suptitle('Discretization Error Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"theoretical_9_error_analysis_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()


def create_system_architecture_diagram(output_dir: str, timestamp: str):
    """Create complete system architecture diagram."""
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    ax.text(7, 13, 'Complete QUBO System Architecture', ha='center', va='center', 
           fontsize=18, fontweight='bold')
    
    # Input layer
    input_box = FancyBboxPatch((1, 11), 12, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#2ECC71', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(7, 11.75, 'Input: UC Problem Data (Problem 2 Results)', 
           ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    
    # Processing layers
    layers = [
        ('Data Loader', 'Load UC parameters\nand constraints', 2, 9, '#3498DB'),
        ('Discretizer', 'Convert continuous\nto binary', 5, 9, '#E74C3C'),
        ('QUBO Builder', 'Build Q matrix\nand penalty terms', 8, 9, '#9B59B6'),
        ('Solver', 'Optimize QUBO\nmodel', 11, 9, '#F39C12'),
    ]
    
    layer_boxes = {}
    for name, desc, x, y, color in layers:
        box = FancyBboxPatch((x - 1, y - 0.7), 2, 1.4, 
                            boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y + 0.2, name, ha='center', va='center', 
               fontsize=11, fontweight='bold', color='white')
        ax.text(x, y - 0.3, desc, ha='center', va='center', 
               fontsize=9, color='white')
        layer_boxes[name] = (x, y)
    
    # Arrows between layers
    layer_names = ['Data Loader', 'Discretizer', 'QUBO Builder', 'Solver']
    for i in range(len(layer_names) - 1):
        start = layer_boxes[layer_names[i]]
        end = layer_boxes[layer_names[i + 1]]
        arrow = FancyArrowPatch((start[0] + 1, start[1]), (end[0] - 1, end[1]), 
                               arrowstyle='->', lw=3, color='#34495E')
        ax.add_patch(arrow)
    
    # Components
    components = [
        ('Binary Variables', 'xᵢₜₖ ∈ {0,1}', 2, 6.5, '#ECF0F1'),
        ('Q Matrix', 'Symmetric N×N', 5, 6.5, '#ECF0F1'),
        ('Penalty Terms', 'Constraint violations', 8, 6.5, '#ECF0F1'),
        ('Solution Vector', 'Binary assignment', 11, 6.5, '#ECF0F1'),
    ]
    
    for name, desc, x, y, color in components:
        comp_box = FancyBboxPatch((x - 1, y - 0.5), 2, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(comp_box)
        ax.text(x, y + 0.15, name, ha='center', va='center', 
               fontsize=10, fontweight='bold')
        ax.text(x, y - 0.2, desc, ha='center', va='center', 
               fontsize=8, style='italic')
    
    # Output layer
    output_box = FancyBboxPatch((4, 4), 6, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#2ECC71', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7, 4.75, 'Output: Generation Schedule & Verification', 
           ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    
    # Arrows to output
    arrow1 = FancyArrowPatch((11, 9), (10, 5.5), 
                             arrowstyle='->', lw=3, color='#34495E')
    ax.add_patch(arrow1)
    
    # Verification
    verify_box = FancyBboxPatch((1, 1.5), 12, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#FFF9E6', edgecolor='#F39C12', linewidth=2)
    ax.add_patch(verify_box)
    ax.text(7, 2.25, 'Verification: Power Balance | Ramp | Reserve | N-1 | Cost Comparison', 
           ha='center', va='center', fontsize=11, fontweight='bold')
    
    arrow2 = FancyArrowPatch((7, 4), (7, 3), 
                             arrowstyle='->', lw=3, color='#34495E')
    ax.add_patch(arrow2)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"theoretical_10_system_architecture_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate theoretical visualizations for Problem 3'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory containing Problem 3 results'
    )
    parser.add_argument(
        '--timestamp',
        type=str,
        default=None,
        help='Specific timestamp to use'
    )
    
    args = parser.parse_args()
    
    if args.results_dir is None:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        args.results_dir = os.path.join(project_root, 'results', 'problem3')
    
    create_theoretical_visualizations(
        results_dir=args.results_dir,
        timestamp=args.timestamp
    )

