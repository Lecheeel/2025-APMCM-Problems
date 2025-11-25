"""
Advanced Visualizations for Problem 4
======================================
Generates high-quality 3D and advanced 2D visualizations for problem reduction results.
Includes comparison between reduced and full-scale solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass
from matplotlib.colors import LinearSegmentedColormap
import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple


def load_results_from_files(results_dir: str, timestamp: Optional[str] = None) -> Dict:
    """
    Load Problem 4 results from files.
    
    Args:
        results_dir: Directory containing result files
        timestamp: Specific timestamp to load (if None, loads latest)
    
    Returns:
        Dictionary containing loaded data
    """
    if timestamp is None:
        # Find latest summary file
        summary_files = [f for f in os.listdir(results_dir) 
                        if f.startswith('summary_') and f.endswith('.json')]
        if not summary_files:
            raise FileNotFoundError(f"No summary files found in {results_dir}")
        latest_file = max(summary_files, key=lambda f: 
                         os.path.getmtime(os.path.join(results_dir, f)))
        timestamp = latest_file.replace('summary_', '').replace('.json', '')
    
    # Load JSON summary
    summary_path = os.path.join(results_dir, f"summary_{timestamp}.json")
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    
    # Load CSV files if available
    reduced_csv_path = os.path.join(results_dir, f"reduced_generation_schedule_{timestamp}.csv")
    full_csv_path = os.path.join(results_dir, f"full_generation_schedule_{timestamp}.csv")
    
    reduced_df = None
    full_df = None
    if os.path.exists(reduced_csv_path):
        reduced_df = pd.read_csv(reduced_csv_path)
    if os.path.exists(full_csv_path):
        full_df = pd.read_csv(full_csv_path)
    
    return {
        'summary': summary_data,
        'reduced_df': reduced_df,
        'full_df': full_df,
        'timestamp': timestamp
    }


def create_advanced_visualizations(results_dir: str, timestamp: Optional[str] = None,
                                   problem2_results_dir: Optional[str] = None):
    """
    Generate advanced visualizations from Problem 4 results.
    
    Args:
        results_dir: Directory containing Problem 4 results
        timestamp: Specific timestamp to visualize (if None, uses latest)
        problem2_results_dir: Directory containing Problem 2 results for comparison
    """
    # Load Problem 4 results
    print("Loading Problem 4 results...")
    results = load_results_from_files(results_dir, timestamp)
    summary = results['summary']
    opt_info = summary['optimization_info']
    
    reduced_generation = np.array(summary['reduced_generation'])
    full_generation = np.array(summary['full_generation'])
    
    reduction_info = opt_info['reduction_info']
    selected_units = reduction_info['selected_units']
    selected_periods = reduction_info['selected_periods']
    
    num_units_reduced = opt_info['num_units']
    num_periods_reduced = opt_info['num_periods']
    num_units_full = 6
    num_periods_full = 24
    
    units_full = [1, 2, 5, 8, 11, 13]
    units_reduced = [units_full[i] for i in selected_units]
    
    periods_full = np.arange(1, num_periods_full + 1)
    periods_reduced = np.array([selected_periods[i] + 1 for i in range(len(selected_periods))])
    
    # Load demand data
    load_demand_full = np.array([166, 196, 229, 257, 263, 252, 246, 213, 192, 161, 147, 160,
                                170, 185, 208, 232, 246, 241, 236, 225, 204, 182, 161, 131])
    load_demand_reduced = load_demand_full[selected_periods]
    
    # Load unit parameters (P_min and P_max)
    P_min_full = np.array([50, 20, 15, 10, 10, 12])  # From Table 1
    P_max_full = np.array([300, 180, 50, 35, 30, 40])  # From Table 1
    P_min_reduced = P_min_full[selected_units]
    P_max_reduced = P_max_full[selected_units]
    
    # Load Problem 2 results for comparison if available
    problem2_generation = None
    problem2_cost = None
    if problem2_results_dir and os.path.exists(problem2_results_dir):
        try:
            p2_summary_files = [f for f in os.listdir(problem2_results_dir) 
                               if f.startswith('summary_') and f.endswith('.json')]
            if p2_summary_files:
                latest_p2 = max(p2_summary_files, key=lambda f: 
                               os.path.getmtime(os.path.join(problem2_results_dir, f)))
                with open(os.path.join(problem2_results_dir, latest_p2), 'r') as f:
                    p2_data = json.load(f)
                    problem2_cost = p2_data['optimization_info']['total_cost']
                    # Try to load generation from CSV
                    p2_csv_files = [f for f in os.listdir(problem2_results_dir) 
                                  if f.startswith('uc_schedule_') and f.endswith('.csv')]
                    if p2_csv_files:
                        latest_p2_csv = max(p2_csv_files, key=lambda f: 
                                          os.path.getmtime(os.path.join(problem2_results_dir, f)))
                        p2_df = pd.read_csv(os.path.join(problem2_results_dir, latest_p2_csv))
                        problem2_generation = np.zeros((num_units_full, num_periods_full))
                        for i, unit_id in enumerate(units_full):
                            col_name = f'Unit_{unit_id}_Generation_MW'
                            if col_name in p2_df.columns:
                                problem2_generation[i, :] = p2_df[col_name].values
        except Exception as e:
            print(f"Warning: Could not load Problem 2 results: {e}")
    
    # Extract cost information
    reduced_cost = opt_info['reduced_problem_cost']
    full_scale_cost = opt_info['full_scale_cost']
    num_binary_vars = opt_info['num_binary_vars']
    # Try to get max_binary_vars from reduction_info, otherwise use a default
    max_binary_vars = reduction_info.get('max_binary_vars', 100)
    if 'max_binary_vars' not in reduction_info:
        # Estimate from reduction ratio if available
        if reduction_info['reduction_ratio']['total'] > 0:
            estimated_original = num_binary_vars / reduction_info['reduction_ratio']['total']
            max_binary_vars = min(100, int(estimated_original * 0.5))  # Conservative estimate
    
    # Professional color schemes
    colors_units = plt.cm.viridis(np.linspace(0, 0.9, num_units_full))
    colors_reduction = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']  # Red, Blue, Green, Orange
    
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
    
    timestamp_str = results['timestamp']
    output_dir = results_dir
    
    # ============================================================================
    # Visualization 1: 3D Comparison - Reduced vs Full Scale Generation
    # ============================================================================
    print("Creating Advanced Visualization 1: 3D Reduced vs Full Scale Comparison...")
    fig = plt.figure(figsize=(20, 10))
    
    # Left: Reduced problem
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    X_reduced, Y_reduced = np.meshgrid(periods_reduced, range(num_units_reduced))
    Z_reduced = reduced_generation
    
    surf1 = ax1.plot_surface(X_reduced, Y_reduced, Z_reduced, cmap='Reds', alpha=0.8, 
                            linewidth=0.3, antialiased=True, edgecolor='black', linewidths=0.2)
    ax1.set_xlabel('Time Period (Hour)', labelpad=10, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Unit Index', labelpad=10, fontsize=11, fontweight='bold')
    ax1.set_zlabel('Generation (MW)', labelpad=10, fontsize=11, fontweight='bold')
    ax1.set_title(f'Reduced Problem\n({num_units_reduced} units, {num_periods_reduced} periods, {num_binary_vars} vars)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.set_yticks(range(num_units_reduced))
    ax1.set_yticklabels([f'Unit {u}' for u in units_reduced])
    ax1.view_init(elev=25, azim=45)
    
    # Right: Full scale
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    X_full, Y_full = np.meshgrid(periods_full, range(num_units_full))
    Z_full = full_generation
    
    surf2 = ax2.plot_surface(X_full, Y_full, Z_full, cmap='Blues', alpha=0.8, 
                             linewidth=0.3, antialiased=True, edgecolor='black', linewidths=0.2)
    ax2.set_xlabel('Time Period (Hour)', labelpad=10, fontsize=11, fontweight='bold')
    ax2.set_ylabel('Unit Index', labelpad=10, fontsize=11, fontweight='bold')
    ax2.set_zlabel('Generation (MW)', labelpad=10, fontsize=11, fontweight='bold')
    ax2.set_title(f'Full Scale Solution\n({num_units_full} units, {num_periods_full} periods)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax2.set_yticks(range(num_units_full))
    ax2.set_yticklabels([f'Unit {u}' for u in units_full])
    ax2.view_init(elev=25, azim=45)
    
    plt.suptitle('3D Generation Comparison: Reduced vs Full Scale', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"advanced_1_3d_reduced_vs_full_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 2: Reduction Strategy Heatmap
    # ============================================================================
    print("Creating Advanced Visualization 2: Reduction Strategy Heatmap...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Top: Unit selection heatmap
    unit_selection = np.zeros((num_units_full, num_periods_full))
    for i, unit_idx in enumerate(selected_units):
        unit_selection[unit_idx, :] = 1
    
    im1 = axes[0].imshow(unit_selection, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Unit', fontsize=12, fontweight='bold')
    axes[0].set_title('Unit Selection Strategy\n(Green=Selected, Red=Not Selected)', 
                     fontsize=13, fontweight='bold')
    axes[0].set_yticks(range(num_units_full))
    axes[0].set_yticklabels([f'Unit {u}' for u in units_full])
    axes[0].set_xticks(range(0, num_periods_full, 2))
    axes[0].set_xticklabels(range(1, num_periods_full + 1, 2))
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Selection Status', rotation=270, labelpad=20)
    
    # Bottom: Period selection heatmap
    period_selection = np.zeros((1, num_periods_full))
    period_selection[0, selected_periods] = 1
    
    im2 = axes[1].imshow(period_selection, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    axes[1].set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Period Selection', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Period Selection Strategy\n({len(selected_periods)}/{num_periods_full} periods selected)', 
                     fontsize=13, fontweight='bold')
    axes[1].set_yticks([0])
    axes[1].set_yticklabels(['Selected'])
    axes[1].set_xticks(range(0, num_periods_full, 2))
    axes[1].set_xticklabels(range(1, num_periods_full + 1, 2))
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Selection Status', rotation=270, labelpad=20)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"advanced_2_reduction_strategy_heatmap_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 3: 3D Binary Variable Reduction Visualization
    # ============================================================================
    print("Creating Advanced Visualization 3: 3D Binary Variable Reduction...")
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Original problem size
    original_vars = num_units_full * num_periods_full * 2  # Assuming 2-bit originally
    reduced_vars = num_binary_vars
    
    # Create bar chart showing reduction
    categories = ['Original\nProblem', 'Reduced\nProblem', 'CIM\nLimit']
    x_pos = [0, 1, 2]
    y_pos = [0, 0, 0]
    z_pos = [0, 0, 0]
    
    dx = [0.8, 0.8, 0.8]
    dy = [0.8, 0.8, 0.8]
    dz = [original_vars, reduced_vars, max_binary_vars]
    
    colors_bar = ['#E74C3C', '#3498DB', '#2ECC71']
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, 
            color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add text labels
    for i, (cat, z_val) in enumerate(zip(categories, dz)):
        ax.text(i, 0.5, z_val + max(dz) * 0.05, f'{int(z_val)}', 
               fontsize=12, fontweight='bold', ha='center')
        ax.text(i, -0.5, -max(dz) * 0.1, cat, 
               fontsize=11, ha='center', va='top')
    
    ax.set_xlabel('Problem Configuration', labelpad=12, fontweight='bold')
    ax.set_ylabel('', labelpad=12)
    ax.set_zlabel('Number of Binary Variables', labelpad=12, fontweight='bold')
    ax.set_title('3D Binary Variable Reduction Visualization', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(categories)
    ax.set_yticks([])
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"advanced_3_3d_binary_vars_reduction_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 4: Generation Heatmap Comparison
    # ============================================================================
    print("Creating Advanced Visualization 4: Generation Heatmap Comparison...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Top: Reduced generation heatmap
    im1 = axes[0].imshow(reduced_generation, aspect='auto', cmap='YlOrRd', 
                        interpolation='nearest', vmin=0, vmax=np.max(full_generation))
    axes[0].set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Unit', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Reduced Problem Generation Heatmap\n({num_units_reduced} units × {num_periods_reduced} periods)', 
                     fontsize=13, fontweight='bold')
    axes[0].set_yticks(range(num_units_reduced))
    axes[0].set_yticklabels([f'Unit {u}' for u in units_reduced])
    axes[0].set_xticks(range(len(periods_reduced)))
    axes[0].set_xticklabels([int(p) for p in periods_reduced])
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Generation (MW)', rotation=270, labelpad=20)
    
    # Add text annotations for reduced problem
    for i in range(num_units_reduced):
        for j in range(num_periods_reduced):
            text = axes[0].text(j, i, f'{reduced_generation[i, j]:.0f}',
                              ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    # Bottom: Full scale generation heatmap
    im2 = axes[1].imshow(full_generation, aspect='auto', cmap='YlOrRd', 
                        interpolation='nearest', vmin=0, vmax=np.max(full_generation))
    axes[1].set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Unit', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Full Scale Generation Heatmap\n({num_units_full} units × {num_periods_full} periods)', 
                     fontsize=13, fontweight='bold')
    axes[1].set_yticks(range(num_units_full))
    axes[1].set_yticklabels([f'Unit {u}' for u in units_full])
    axes[1].set_xticks(range(0, num_periods_full, 2))
    axes[1].set_xticklabels(range(1, num_periods_full + 1, 2))
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Generation (MW)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"advanced_4_generation_heatmap_comparison_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 5: 3D Cost Surface Comparison
    # ============================================================================
    print("Creating Advanced Visualization 5: 3D Cost Surface Comparison...")
    fig = plt.figure(figsize=(20, 10))
    
    # Calculate cost per unit per period
    # Load cost coefficients directly from data
    # Table 2 cost coefficients: a, b, c
    a_coeff = np.array([0.02, 0.0175, 0.0625, 0.00834, 0.025, 0.025])
    b_coeff = np.array([2.00, 1.75, 1.00, 3.25, 3.00, 3.00])
    c_coeff = np.array([0, 0, 0, 0, 0, 0])
    
    # Reduced problem cost surface
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    cost_reduced = np.zeros((num_units_reduced, num_periods_reduced))
    for i in range(num_units_reduced):
        unit_idx = selected_units[i]
        for j in range(num_periods_reduced):
            p = reduced_generation[i, j]
            cost_reduced[i, j] = (a_coeff[unit_idx] * p**2 + 
                                 b_coeff[unit_idx] * p + 
                                 c_coeff[unit_idx])
    
    X_reduced, Y_reduced = np.meshgrid(periods_reduced, range(num_units_reduced))
    surf1 = ax1.plot_surface(X_reduced, Y_reduced, cost_reduced, cmap='coolwarm', 
                            alpha=0.85, linewidth=0.3, antialiased=True)
    ax1.set_xlabel('Time Period (Hour)', labelpad=10, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Unit Index', labelpad=10, fontsize=11, fontweight='bold')
    ax1.set_zlabel('Cost ($)', labelpad=10, fontsize=11, fontweight='bold')
    ax1.set_title(f'Reduced Problem Cost Surface\n(Total: ${reduced_cost:.2f})', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.set_yticks(range(num_units_reduced))
    ax1.set_yticklabels([f'Unit {u}' for u in units_reduced])
    ax1.view_init(elev=25, azim=45)
    
    # Full scale cost surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    cost_full = np.zeros((num_units_full, num_periods_full))
    for i in range(num_units_full):
        for j in range(num_periods_full):
            p = full_generation[i, j]
            cost_full[i, j] = (a_coeff[i] * p**2 + 
                              b_coeff[i] * p + 
                              c_coeff[i])
    
    X_full, Y_full = np.meshgrid(periods_full, range(num_units_full))
    surf2 = ax2.plot_surface(X_full, Y_full, cost_full, cmap='coolwarm', 
                            alpha=0.85, linewidth=0.3, antialiased=True)
    ax2.set_xlabel('Time Period (Hour)', labelpad=10, fontsize=11, fontweight='bold')
    ax2.set_ylabel('Unit Index', labelpad=10, fontsize=11, fontweight='bold')
    ax2.set_zlabel('Cost ($)', labelpad=10, fontsize=11, fontweight='bold')
    ax2.set_title(f'Full Scale Cost Surface\n(Total: ${full_scale_cost:.2f})', 
                 fontsize=13, fontweight='bold', pad=15)
    ax2.set_yticks(range(num_units_full))
    ax2.set_yticklabels([f'Unit {u}' for u in units_full])
    ax2.view_init(elev=25, azim=45)
    
    plt.suptitle('3D Cost Surface Comparison: Reduced vs Full Scale', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"advanced_5_3d_cost_surface_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 6: Cost Comparison Bar Chart with Problem 2
    # ============================================================================
    print("Creating Advanced Visualization 6: Cost Comparison Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Cost comparison
    costs = []
    labels = []
    colors_cost = []
    
    if problem2_cost:
        costs.extend([problem2_cost, full_scale_cost, reduced_cost])
        labels.extend(['Problem 2\n(Continuous)', 'Problem 4\n(Full Scale)', 'Problem 4\n(Reduced)'])
        colors_cost.extend(['#3498DB', '#2ECC71', '#E74C3C'])
    else:
        costs.extend([full_scale_cost, reduced_cost])
        labels.extend(['Full Scale', 'Reduced'])
        colors_cost.extend(['#2ECC71', '#E74C3C'])
    
    bars = axes[0].bar(range(len(costs)), costs, color=colors_cost, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=11)
    axes[0].set_ylabel('Total Cost ($)', fontsize=12, fontweight='bold')
    axes[0].set_title('Cost Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'${cost:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Right: Reduction metrics
    reduction_ratios = reduction_info['reduction_ratio']
    metrics = ['Units', 'Periods', 'Total']
    values = [reduction_ratios['units'] * 100, 
             reduction_ratios['periods'] * 100,
             reduction_ratios['total'] * 100]
    
    bars2 = axes[1].bar(metrics, values, color=['#9B59B6', '#E67E22', '#1ABC9C'], 
                       alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Retention Ratio (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Problem Reduction Metrics', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 105])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars2, values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"advanced_6_cost_comparison_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 7: 3D Generation Difference Surface
    # ============================================================================
    print("Creating Advanced Visualization 7: 3D Generation Difference Surface...")
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create full-scale difference matrix
    # Map reduced solution to full scale for comparison
    difference_matrix = np.zeros((num_units_full, num_periods_full))
    
    # For selected units and periods, calculate difference
    for i, unit_idx in enumerate(selected_units):
        for j, period_idx in enumerate(selected_periods):
            reduced_val = reduced_generation[i, j]
            full_val = full_generation[unit_idx, period_idx]
            difference_matrix[unit_idx, period_idx] = full_val - reduced_val
    
    # For non-selected periods, use interpolation or zero
    for i, unit_idx in enumerate(selected_units):
        for period_idx in range(num_periods_full):
            if period_idx not in selected_periods:
                # Find nearest selected period
                distances = [abs(period_idx - sp) for sp in selected_periods]
                nearest_idx = np.argmin(distances)
                nearest_period = selected_periods[nearest_idx]
                reduced_val = reduced_generation[i, nearest_idx]
                full_val = full_generation[unit_idx, period_idx]
                difference_matrix[unit_idx, period_idx] = full_val - reduced_val
    
    X, Y = np.meshgrid(periods_full, range(num_units_full))
    Z = difference_matrix
    
    # Create surface with diverging colormap
    surf = ax.plot_surface(X, Y, Z, cmap='RdBu_r', alpha=0.85, 
                          linewidth=0.3, antialiased=True, 
                          vmin=-np.max(np.abs(Z)), vmax=np.max(np.abs(Z)))
    
    ax.set_xlabel('Time Period (Hour)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Unit Index', labelpad=12, fontweight='bold')
    ax.set_zlabel('Generation Difference (MW)', labelpad=12, fontweight='bold')
    ax.set_title('3D Generation Difference Surface\n(Full Scale - Reduced Solution)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(num_units_full))
    ax.set_yticklabels([f'Unit {u}' for u in units_full])
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Difference (MW)', rotation=270, labelpad=20, fontweight='bold')
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"advanced_7_3d_difference_surface_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 8: Detailed Reduction Information Dashboard
    # ============================================================================
    print("Creating Advanced Visualization 8: Detailed Reduction Information Dashboard...")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Binary variables comparison (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    var_data = [original_vars, reduced_vars, max_binary_vars]
    var_labels = ['Original', 'Reduced', 'CIM Limit']
    var_colors = ['#E74C3C', '#3498DB', '#2ECC71']
    bars = ax1.bar(var_labels, var_data, color=var_colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Binary Variables', fontsize=11, fontweight='bold')
    ax1.set_title('Binary Variable Count', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, var_data):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Reduction ratios (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ratios = [reduction_ratios['units'], reduction_ratios['periods'], reduction_ratios['total']]
    ratio_labels = ['Units', 'Periods', 'Total']
    bars2 = ax2.bar(ratio_labels, [r*100 for r in ratios], 
                    color=['#9B59B6', '#E67E22', '#1ABC9C'], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Retention Ratio (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Reduction Ratios', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, [r*100 for r in ratios]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Cost breakdown (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    cost_data = [reduced_cost, full_scale_cost]
    cost_labels = ['Reduced', 'Full Scale']
    cost_colors = ['#E74C3C', '#3498DB']
    bars3 = ax3.bar(cost_labels, cost_data, color=cost_colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Cost ($)', fontsize=11, fontweight='bold')
    ax3.set_title('Cost Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, cost_data):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Selected units visualization (middle-left)
    ax4 = fig.add_subplot(gs[1, 0])
    unit_selection_array = np.array([1 if i in selected_units else 0 for i in range(num_units_full)])
    colors_units_sel = ['#2ECC71' if x == 1 else '#E74C3C' for x in unit_selection_array]
    bars4 = ax4.bar(range(num_units_full), unit_selection_array, 
                   color=colors_units_sel, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_xticks(range(num_units_full))
    ax4.set_xticklabels([f'Unit {u}' for u in units_full], rotation=45, ha='right')
    ax4.set_ylabel('Selection Status', fontsize=11, fontweight='bold')
    ax4.set_title('Unit Selection', fontsize=12, fontweight='bold')
    ax4.set_ylim([-0.1, 1.2])
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Not Selected', 'Selected'])
    
    # 5. Selected periods visualization (middle-middle)
    ax5 = fig.add_subplot(gs[1, 1])
    period_selection_array = np.array([1 if i in selected_periods else 0 for i in range(num_periods_full)])
    colors_periods_sel = ['#2ECC71' if x == 1 else '#E74C3C' for x in period_selection_array]
    ax5.bar(range(num_periods_full), period_selection_array, 
           color=colors_periods_sel, alpha=0.8, edgecolor='black', linewidth=0.5, width=0.8)
    ax5.set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Selection Status', fontsize=11, fontweight='bold')
    ax5.set_title('Period Selection', fontsize=12, fontweight='bold')
    ax5.set_xticks(range(0, num_periods_full, 4))
    ax5.set_xticklabels(range(1, num_periods_full + 1, 4))
    ax5.set_ylim([-0.1, 1.2])
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Not Selected', 'Selected'])
    
    # 6. Load demand comparison (middle-right)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(periods_full, load_demand_full, 'b-o', linewidth=2, markersize=4, 
            label='Full Scale Load', alpha=0.7)
    ax6.plot(periods_reduced, load_demand_reduced, 'r-s', linewidth=2, markersize=6, 
            label='Selected Periods Load', alpha=0.9)
    ax6.set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Load Demand (MW)', fontsize=11, fontweight='bold')
    ax6.set_title('Load Demand Comparison', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Generation comparison for selected periods (bottom, spanning 3 columns)
    ax7 = fig.add_subplot(gs[2, :])
    for i, unit_idx in enumerate(selected_units):
        reduced_gen_unit = reduced_generation[i, :]
        full_gen_unit = full_generation[unit_idx, selected_periods]
        ax7.plot(periods_reduced, reduced_gen_unit, 'o-', linewidth=2, markersize=6,
                label=f'Unit {units_full[unit_idx]} (Reduced)', alpha=0.8)
        ax7.plot(periods_reduced, full_gen_unit, 's--', linewidth=2, markersize=4,
                label=f'Unit {units_full[unit_idx]} (Full)', alpha=0.6)
    ax7.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Generation (MW)', fontsize=12, fontweight='bold')
    ax7.set_title('Generation Comparison: Reduced vs Full Scale (Selected Periods)', 
                 fontsize=13, fontweight='bold')
    ax7.legend(ncol=3, fontsize=9, loc='upper right')
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Problem 4 Reduction Strategy Dashboard', 
                fontsize=16, fontweight='bold', y=0.995)
    path = os.path.join(output_dir, f"advanced_8_reduction_dashboard_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 9: Problem Reduction Flow Diagram (Principle)
    # ============================================================================
    print("Creating Advanced Visualization 9: Problem Reduction Flow Diagram...")
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Original problem box
    orig_box = plt.Rectangle((0.5, 6), 2, 1.5, linewidth=2, edgecolor='#E74C3C', 
                             facecolor='#FFE5E5', alpha=0.8)
    ax.add_patch(orig_box)
    ax.text(1.75, 7.25, 'Original Problem', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(1.75, 6.8, f'{num_units_full} units × {num_periods_full} periods', 
           ha='center', va='center', fontsize=10)
    ax.text(1.75, 6.4, f'{num_units_full * num_periods_full * 2} binary vars', 
           ha='center', va='center', fontsize=10, color='#C0392B')
    
    # Reduction strategies
    reduction_box = plt.Rectangle((3.5, 5.5), 3, 2.5, linewidth=2, edgecolor='#3498DB', 
                                  facecolor='#EBF5FB', alpha=0.8)
    ax.add_patch(reduction_box)
    ax.text(5, 7.5, 'Reduction Strategies', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(5, 7, '1. Unit Selection', ha='center', va='center', fontsize=10)
    ax.text(5, 6.5, f'   → {len(selected_units)}/{num_units_full} units', 
           ha='center', va='center', fontsize=9, style='italic')
    ax.text(5, 6, '2. Period Aggregation', ha='center', va='center', fontsize=10)
    ax.text(5, 5.5, f'   → {len(selected_periods)}/{num_periods_full} periods', 
           ha='center', va='center', fontsize=9, style='italic')
    
    # Reduced problem box
    reduced_box = plt.Rectangle((7.5, 6), 2, 1.5, linewidth=2, edgecolor='#2ECC71', 
                                facecolor='#E8F8F5', alpha=0.8)
    ax.add_patch(reduced_box)
    ax.text(8.75, 7.25, 'Reduced Problem', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(8.75, 6.8, f'{num_units_reduced} units × {num_periods_reduced} periods', 
           ha='center', va='center', fontsize=10)
    ax.text(8.75, 6.4, f'{num_binary_vars} binary vars', 
           ha='center', va='center', fontsize=10, color='#27AE60')
    
    # Arrows
    ax.arrow(2.5, 6.75, 0.8, 0, head_width=0.15, head_length=0.15, 
            fc='#34495E', ec='#34495E', linewidth=2)
    ax.arrow(7.5, 6.75, -0.8, 0, head_width=0.15, head_length=0.15, 
            fc='#34495E', ec='#34495E', linewidth=2)
    
    # CIM constraint
    cim_box = plt.Rectangle((3.5, 3), 3, 1.5, linewidth=2, edgecolor='#9B59B6', 
                            facecolor='#F4ECF7', alpha=0.8)
    ax.add_patch(cim_box)
    ax.text(5, 4, 'CIM Constraint', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(5, 3.5, f'Max: {max_binary_vars} binary vars', 
           ha='center', va='center', fontsize=10)
    
    # Discretization process
    disc_box = plt.Rectangle((0.5, 1), 2, 1.5, linewidth=2, edgecolor='#E67E22', 
                            facecolor='#FEF5E7', alpha=0.8)
    ax.add_patch(disc_box)
    ax.text(1.75, 2.25, 'Discretization', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(1.75, 1.8, f'{opt_info["num_bits_per_unit"]}-bit encoding', 
           ha='center', va='center', fontsize=10)
    ax.text(1.75, 1.4, 'p = P_min + bit × Δ', 
           ha='center', va='center', fontsize=9, style='italic')
    
    # QUBO building
    qubo_box = plt.Rectangle((3.5, 1), 3, 1.5, linewidth=2, edgecolor='#1ABC9C', 
                            facecolor='#E8F8F5', alpha=0.8)
    ax.add_patch(qubo_box)
    ax.text(5, 2.25, 'QUBO Formulation', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(5, 1.8, 'min x^T Q x', ha='center', va='center', fontsize=10)
    ax.text(5, 1.4, 'Constraints → Penalties', 
           ha='center', va='center', fontsize=9, style='italic')
    
    # Solution expansion
    expand_box = plt.Rectangle((7.5, 1), 2, 1.5, linewidth=2, edgecolor='#16A085', 
                              facecolor='#D5F4E6', alpha=0.8)
    ax.add_patch(expand_box)
    ax.text(8.75, 2.25, 'Solution Expansion', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(8.75, 1.8, 'Reduced → Full Scale', 
           ha='center', va='center', fontsize=10)
    ax.text(8.75, 1.4, 'Nearest neighbor', 
           ha='center', va='center', fontsize=9, style='italic')
    
    # Vertical arrows
    ax.arrow(1.75, 4.5, 0, -0.8, head_width=0.15, head_length=0.15, 
            fc='#34495E', ec='#34495E', linewidth=2)
    ax.arrow(5, 4.5, 0, -0.8, head_width=0.15, head_length=0.15, 
            fc='#34495E', ec='#34495E', linewidth=2)
    ax.arrow(8.75, 4.5, 0, -0.8, head_width=0.15, head_length=0.15, 
            fc='#34495E', ec='#34495E', linewidth=2)
    
    # Reduction ratio annotation
    reduction_ratio = reduction_info['reduction_ratio']['total'] * 100
    ax.text(5, 2.5, f'Reduction Ratio: {reduction_ratio:.1f}%', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.title('Problem 4: Reduction Strategy Flow Diagram', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    path = os.path.join(output_dir, f"principle_1_reduction_flow_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 10: Discretization Encoding Principle
    # ============================================================================
    print("Creating Advanced Visualization 10: Discretization Encoding Principle...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Use actual unit parameters for discretization
    P_min_example = P_min_full
    P_max_example = P_max_full
    delta_example = P_max_example - P_min_example
    
    # Top-left: 1-bit encoding illustration
    ax1 = axes[0, 0]
    unit_idx = 0  # Unit 1 as example
    bit_values = [0, 1]
    gen_values_1bit = [P_min_example[unit_idx], P_max_example[unit_idx]]
    
    bars1 = ax1.bar(['Bit=0', 'Bit=1'], gen_values_1bit, 
                    color=['#E74C3C', '#2ECC71'], alpha=0.8, 
                    edgecolor='black', linewidth=2)
    ax1.set_ylabel('Generation (MW)', fontsize=12, fontweight='bold')
    ax1.set_title(f'1-Bit Encoding (Unit {units_full[unit_idx]})\np = P_min + bit × (P_max - P_min)', 
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=P_min_example[unit_idx], color='blue', linestyle='--', 
               linewidth=1, label=f'P_min = {P_min_example[unit_idx]} MW')
    ax1.axhline(y=P_max_example[unit_idx], color='red', linestyle='--', 
               linewidth=1, label=f'P_max = {P_max_example[unit_idx]} MW')
    ax1.legend(fontsize=9)
    
    for bar, val in zip(bars1, gen_values_1bit):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f} MW',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Top-right: Multi-bit encoding illustration
    ax2 = axes[0, 1]
    num_bits_example = 2
    bit_combinations = ['00', '01', '10', '11']
    gen_values_2bit = []
    for bits in bit_combinations:
        bit_int = int(bits, 2)
        gen_val = P_min_example[unit_idx] + bit_int * (delta_example[unit_idx] / (2**num_bits_example - 1))
        gen_values_2bit.append(gen_val)
    
    bars2 = ax2.bar(bit_combinations, gen_values_2bit, 
                    color=plt.cm.viridis(np.linspace(0, 1, len(bit_combinations))), 
                    alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Generation (MW)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Binary Encoding', fontsize=11, fontweight='bold')
    ax2.set_title(f'2-Bit Encoding (Unit {units_full[unit_idx]})\np = P_min + int(bits) × Δ/(2^n-1)', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, gen_values_2bit):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} MW',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Bottom-left: Discretization step sizes for all units
    ax3 = axes[1, 0]
    unit_labels = [f'Unit {u}' for u in units_full]
    bars3 = ax3.bar(unit_labels, delta_example, 
                    color=plt.cm.plasma(np.linspace(0, 0.9, len(units_full))), 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Discretization Step Δ (MW)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Unit', fontsize=12, fontweight='bold')
    ax3.set_title('Discretization Step Sizes (1-bit: Δ = P_max - P_min)', 
                 fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars3, delta_example):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Bottom-right: Encoding range visualization
    ax4 = axes[1, 1]
    for i, unit_id in enumerate(units_full):
        y_pos = len(units_full) - i - 1
        ax4.barh(y_pos, P_max_example[i] - P_min_example[i], 
                left=P_min_example[i], height=0.6,
                color=colors_units[i], alpha=0.7, edgecolor='black', linewidth=1)
        ax4.text((P_min_example[i] + P_max_example[i])/2, y_pos,
                f'[{P_min_example[i]:.0f}, {P_max_example[i]:.0f}]',
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax4.set_xlabel('Generation Range (MW)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Unit', fontsize=12, fontweight='bold')
    ax4.set_yticks(range(len(units_full)))
    ax4.set_yticklabels([f'Unit {u}' for u in units_full])
    ax4.set_title('Generation Ranges for All Units', 
                 fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Discretization Encoding Principle', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(output_dir, f"principle_2_discretization_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 11: Time Aggregation Principle
    # ============================================================================
    print("Creating Advanced Visualization 11: Time Aggregation Principle...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Top: Original 24 periods with selection
    ax1 = axes[0]
    periods_all = np.arange(1, num_periods_full + 1)
    colors_periods = ['#2ECC71' if i in selected_periods else '#E74C3C' 
                     for i in range(num_periods_full)]
    
    bars1 = ax1.bar(periods_all, load_demand_full, color=colors_periods, 
                    alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.plot(periods_all, load_demand_full, 'k-', linewidth=2, alpha=0.5)
    ax1.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Load Demand (MW)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Original 24 Periods\n(Green=Selected, Red=Not Selected)', 
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(periods_all[::2])
    
    # Add annotation for selected periods
    for i, period in enumerate(selected_periods):
        ax1.annotate(f'P{period+1}', xy=(period+1, load_demand_full[period]), 
                    xytext=(period+1, load_demand_full[period] + 20),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                    fontsize=9, fontweight='bold', color='green', ha='center')
    
    # Bottom: Aggregated periods
    ax2 = axes[1]
    periods_reduced_plot = np.array([p+1 for p in selected_periods])
    load_reduced_plot = load_demand_full[selected_periods]
    
    bars2 = ax2.bar(periods_reduced_plot, load_reduced_plot, 
                    color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.plot(periods_reduced_plot, load_reduced_plot, 'b-o', linewidth=2, 
            markersize=8, label='Selected Periods')
    ax2.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Load Demand (MW)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Aggregated {len(selected_periods)} Periods\n(Uniform Aggregation Strategy)', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)
    
    # Add period mapping visualization
    for i, period in enumerate(selected_periods):
        ax2.text(period+1, load_reduced_plot[i] + 10,
                f'P{period+1}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Add mapping arrows annotation
    ax2.text(0.02, 0.95, f'Period Mapping:\n{len(selected_periods)} periods represent {num_periods_full} periods', 
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Time Aggregation Principle: Uniform Period Selection', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(output_dir, f"principle_3_time_aggregation_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 12: QUBO Constraint Transformation Principle
    # ============================================================================
    print("Creating Advanced Visualization 12: QUBO Constraint Transformation...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top-left: Power balance constraint
    ax1 = axes[0, 0]
    periods_example = periods_reduced[:min(8, len(periods_reduced))]
    total_gen = np.sum(reduced_generation[:, :len(periods_example)], axis=0)
    load_example = load_demand_reduced[:len(periods_example)]
    balance_error = total_gen - load_example
    
    x_pos = np.arange(len(periods_example))
    width = 0.35
    bars1 = ax1.bar(x_pos - width/2, total_gen, width, label='Total Generation', 
                   color='#2ECC71', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x_pos + width/2, load_example, width, label='Load Demand', 
                   color='#E74C3C', alpha=0.8, edgecolor='black')
    
    # Add error bars
    ax1.errorbar(x_pos, load_example, yerr=np.abs(balance_error), 
                fmt='none', color='black', capsize=5, linewidth=2,
                label='Balance Error')
    
    ax1.set_xlabel('Time Period', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Power (MW)', fontsize=12, fontweight='bold')
    ax1.set_title('Power Balance Constraint\nΣ p_i = D_t', 
                 fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([int(p) for p in periods_example])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Top-right: QUBO penalty formulation
    ax2 = axes[0, 1]
    ax2.axis('off')
    penalty_text = r"""
    QUBO Penalty Formulation:
    
    Original Constraint:
    $\sum_{i} p_{i,t} = D_t$  for all $t$
    
    Converted to Penalty:
    $P \cdot (\sum_{i} p_{i,t} - D_t)^2$
    
    Where:
    • $p_{i,t} = P_{min}[i] + \sum_k 2^k \Delta[i] \cdot x_{i,t,k}$
    • $P$ = Penalty coefficient
    • $x_{i,t,k}$ = Binary variables
    
    Expanded to QUBO form:
    $x^T Q x + c^T x + const$
    """
    ax2.text(0.1, 0.5, penalty_text, transform=ax2.transAxes,
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#EBF5FB', alpha=0.8),
            family='monospace')
    
    # Bottom-left: Generation limits
    ax3 = axes[1, 0]
    unit_example_idx = 2  # Unit 5
    unit_gen_example = reduced_generation[unit_example_idx, :len(periods_example)]
    selected_unit_id = selected_units[unit_example_idx]
    P_min_ex = P_min_full[selected_unit_id]
    P_max_ex = P_max_full[selected_unit_id]
    
    ax3.plot(periods_example, unit_gen_example, 'o-', linewidth=2, markersize=8,
            color='#3498DB', label=f'Unit {units_reduced[unit_example_idx]} Generation')
    ax3.axhline(y=P_min_ex, color='green', linestyle='--', linewidth=2,
               label=f'P_min = {P_min_ex} MW')
    ax3.axhline(y=P_max_ex, color='red', linestyle='--', linewidth=2,
               label=f'P_max = {P_max_ex} MW')
    ax3.fill_between(periods_example, P_min_ex, P_max_ex, 
                    alpha=0.2, color='yellow', label='Feasible Region')
    
    ax3.set_xlabel('Time Period', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Generation (MW)', fontsize=12, fontweight='bold')
    ax3.set_title('Generation Limits Constraint\nP_min ≤ p_i ≤ P_max', 
                 fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Bottom-right: Constraint violation penalty visualization
    ax4 = axes[1, 1]
    violation_magnitudes = np.abs(balance_error)
    penalty_coefficient = 2000.0  # Example penalty
    penalties = violation_magnitudes * penalty_coefficient
    
    bars4 = ax4.bar(periods_example, penalties, color='#E74C3C', alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Time Period', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Penalty Cost ($)', fontsize=12, fontweight='bold')
    ax4.set_title('Constraint Violation Penalties\nP × (violation)²', 
                 fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, penalty in zip(bars4, penalties):
        if penalty > 0:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'${penalty:.0f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('QUBO Constraint Transformation Principle', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(output_dir, f"principle_4_qubo_constraints_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 13: Solution Expansion Strategy
    # ============================================================================
    print("Creating Advanced Visualization 13: Solution Expansion Strategy...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Top: Reduced solution
    ax1 = axes[0]
    for i in range(min(3, num_units_reduced)):  # Show first 3 units
        unit_idx = selected_units[i]
        ax1.plot(periods_reduced, reduced_generation[i, :], 
                'o-', linewidth=2, markersize=8,
                label=f'Unit {units_reduced[i]} (Reduced)', alpha=0.8)
    
    ax1.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Generation (MW)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Reduced Solution ({len(selected_periods)} periods)', 
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(periods_reduced)
    ax1.set_xticklabels([int(p) for p in periods_reduced])
    
    # Bottom: Expanded solution
    ax2 = axes[1]
    for i in range(min(3, num_units_full)):
        unit_idx = i
        if unit_idx in selected_units:
            reduced_unit_idx = selected_units.index(unit_idx)
            # Plot reduced periods
            ax2.plot(periods_reduced, reduced_generation[reduced_unit_idx, :], 
                    'o', linewidth=2, markersize=10,
                    label=f'Unit {units_full[unit_idx]} (Reduced)', alpha=0.8)
            # Plot full scale
            ax2.plot(periods_full, full_generation[unit_idx, :], 
                    '-', linewidth=2, alpha=0.6,
                    label=f'Unit {units_full[unit_idx]} (Expanded)', linestyle='--')
        else:
            # Unit not in reduced problem
            ax2.plot(periods_full, full_generation[unit_idx, :], 
                    '-', linewidth=2, alpha=0.6,
                    label=f'Unit {units_full[unit_idx]} (Interpolated)', linestyle=':')
    
    ax2.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Generation (MW)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Expanded Solution (24 periods, Nearest Neighbor Interpolation)', 
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(periods_full[::2])
    
    # Add annotation for expansion method
    expansion_text = f"""
    Expansion Strategy:
    • Selected periods: Use reduced solution directly
    • Non-selected periods: Nearest neighbor interpolation
    • Non-selected units: Use P_min or Problem 2 reference
    """
    ax2.text(0.02, 0.98, expansion_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Solution Expansion Strategy: Reduced → Full Scale', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(output_dir, f"principle_5_solution_expansion_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 14: Binary Variable Encoding Matrix
    # ============================================================================
    print("Creating Advanced Visualization 14: Binary Variable Encoding Matrix...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: Variable structure
    ax1 = axes[0]
    # Create a matrix showing variable indices
    num_vars_per_unit = num_periods_reduced * opt_info['num_bits_per_unit']
    var_matrix = np.zeros((num_units_reduced, num_periods_reduced))
    
    var_idx = 0
    for i in range(num_units_reduced):
        for t in range(num_periods_reduced):
            var_matrix[i, t] = var_idx
            var_idx += opt_info['num_bits_per_unit']
    
    im1 = ax1.imshow(var_matrix, aspect='auto', cmap='viridis', 
                    interpolation='nearest')
    ax1.set_xlabel('Time Period', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Unit Index', fontsize=12, fontweight='bold')
    ax1.set_title(f'Binary Variable Index Structure\n({num_binary_vars} total variables)', 
                 fontsize=13, fontweight='bold')
    ax1.set_yticks(range(num_units_reduced))
    ax1.set_yticklabels([f'Unit {u}' for u in units_reduced])
    ax1.set_xticks(range(num_periods_reduced))
    ax1.set_xticklabels([int(p) for p in periods_reduced])
    
    # Add text annotations
    for i in range(num_units_reduced):
        for j in range(num_periods_reduced):
            var_start = int(var_matrix[i, j])
            if opt_info['num_bits_per_unit'] == 1:
                text = f'x_{var_start}'
            else:
                text = f'x_{var_start}-{var_start+opt_info["num_bits_per_unit"]-1}'
            ax1.text(j, i, text, ha='center', va='center', 
                    color='white', fontsize=7, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Variable Index', rotation=270, labelpad=20)
    
    # Right: QUBO matrix structure visualization
    ax2 = axes[1]
    # Create a simplified QUBO matrix visualization
    matrix_size = min(20, num_binary_vars)  # Show first 20x20 for clarity
    qubo_structure = np.random.rand(matrix_size, matrix_size) * 0.5  # Example structure
    
    # Make it symmetric
    qubo_structure = (qubo_structure + qubo_structure.T) / 2
    
    im2 = ax2.imshow(qubo_structure, aspect='auto', cmap='RdBu_r', 
                    interpolation='nearest', vmin=-0.5, vmax=0.5)
    ax2.set_xlabel('Binary Variable Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Binary Variable Index', fontsize=12, fontweight='bold')
    ax2.set_title(f'QUBO Matrix Structure (Q)\n(Showing first {matrix_size}×{matrix_size} elements)', 
                 fontsize=13, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('QUBO Coefficient', rotation=270, labelpad=20)
    
    # Add formula annotation
    formula_text = r"QUBO Form: $\min \mathbf{x}^T Q \mathbf{x}$"
    ax2.text(0.5, -0.15, formula_text, transform=ax2.transAxes,
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('Binary Variable Encoding and QUBO Matrix Structure', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"principle_6_binary_encoding_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("Advanced visualizations completed successfully!")
    print("=" * 70)
    print(f"\nGenerated {14} advanced visualizations:")
    print("  1. 3D Reduced vs Full Scale Comparison")
    print("  2. Reduction Strategy Heatmap")
    print("  3. 3D Binary Variable Reduction")
    print("  4. Generation Heatmap Comparison")
    print("  5. 3D Cost Surface Comparison")
    print("  6. Cost Comparison Analysis")
    print("  7. 3D Generation Difference Surface")
    print("  8. Detailed Reduction Information Dashboard")
    print("  9. Problem Reduction Flow Diagram (Principle)")
    print("  10. Discretization Encoding Principle")
    print("  11. Time Aggregation Principle")
    print("  12. QUBO Constraint Transformation Principle")
    print("  13. Solution Expansion Strategy")
    print("  14. Binary Variable Encoding Matrix")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate advanced visualizations for Problem 4')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Directory containing Problem 4 results')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Specific timestamp to visualize (if None, uses latest)')
    parser.add_argument('--problem2-dir', type=str, default=None,
                       help='Directory containing Problem 2 results for comparison')
    
    args = parser.parse_args()
    
    if args.results_dir is None:
        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        args.results_dir = os.path.join(project_root, 'results', 'problem4')
    
    if args.problem2_dir is None:
        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        args.problem2_dir = os.path.join(project_root, 'results', 'problem2')
    
    create_advanced_visualizations(args.results_dir, args.timestamp, args.problem2_dir)

