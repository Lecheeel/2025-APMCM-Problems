"""
Advanced Visualizations for Problem 3
======================================
Generates high-quality 3D and advanced 2D visualizations for QUBO solutions.
Can work with existing result files or live data.
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
    Load Problem 3 results from files.
    
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
    
    # Load CSV generation schedule
    csv_path = os.path.join(results_dir, f"generation_schedule_{timestamp}.csv")
    if os.path.exists(csv_path):
        schedule_df = pd.read_csv(csv_path)
    else:
        # Generate from JSON if CSV doesn't exist
        schedule_df = None
    
    return {
        'summary': summary_data,
        'schedule_df': schedule_df,
        'timestamp': timestamp
    }


def create_advanced_visualizations(results_dir: str, timestamp: Optional[str] = None,
                                   problem2_results_dir: Optional[str] = None):
    """
    Generate advanced visualizations from Problem 3 results.
    
    Args:
        results_dir: Directory containing Problem 3 results
        timestamp: Specific timestamp to visualize (if None, uses latest)
        problem2_results_dir: Directory containing Problem 2 results for comparison
    """
    # Load Problem 3 results
    print("Loading Problem 3 results...")
    results = load_results_from_files(results_dir, timestamp)
    summary = results['summary']
    opt_info = summary['optimization_info']
    generation = np.array(summary['generation'])
    
    num_units = opt_info['num_units']
    num_periods = opt_info['num_periods']
    units = [1, 2, 5, 8, 11, 13][:num_units]
    periods = np.arange(1, num_periods + 1)
    
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
                    if 'unit_statistics' in p2_data:
                        # Extract generation from Problem 2
                        problem2_generation = np.zeros((num_units, num_periods))
                        # Try to load from CSV if available
                        p2_csv_files = [f for f in os.listdir(problem2_results_dir) 
                                      if f.startswith('uc_schedule_') and f.endswith('.csv')]
                        if p2_csv_files:
                            latest_p2_csv = max(p2_csv_files, key=lambda f: 
                                              os.path.getmtime(os.path.join(problem2_results_dir, f)))
                            p2_df = pd.read_csv(os.path.join(problem2_results_dir, latest_p2_csv))
                            for i, unit_id in enumerate(units):
                                col_name = f'Unit_{unit_id}_Generation_MW'
                                if col_name in p2_df.columns:
                                    problem2_generation[i, :] = p2_df[col_name].values
                        problem2_cost = p2_data['optimization_info']['total_cost']
        except Exception as e:
            print(f"Warning: Could not load Problem 2 results: {e}")
    
    # Extract data
    load_demand = np.array([166, 196, 229, 257, 263, 252, 246, 213, 192, 161, 147, 160,
                           170, 185, 208, 232, 246, 241, 236, 225, 204, 182, 161, 131])
    
    total_cost = opt_info['total_cost']
    num_bits = opt_info.get('num_bits_per_unit', 1)
    num_binary_vars = opt_info.get('num_binary_vars', num_units * num_periods)
    verification = opt_info.get('verification', {})
    comparison = opt_info.get('comparison', {})
    
    # Professional color schemes
    colors_units = plt.cm.viridis(np.linspace(0, 0.9, num_units))
    colors_qubo = ['#E74C3C', '#3498DB', '#2ECC71']  # Red, Blue, Green
    
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
    # Visualization 1: 3D QUBO Generation Surface
    # ============================================================================
    print("Creating Advanced Visualization 1: 3D QUBO Generation Surface...")
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(periods, range(num_units))
    Z = generation
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85, 
                          linewidth=0.5, antialiased=True, edgecolor='black', linewidths=0.3)
    
    # Add discrete value markers (QUBO characteristic)
    for i in range(num_units):
        for t in range(num_periods):
            ax.scatter(periods[t], i, generation[i, t], 
                      c='red', s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('Time Period (Hour)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Unit Index', labelpad=12, fontweight='bold')
    ax.set_zlabel('Power Generation (MW)', labelpad=12, fontweight='bold')
    ax.set_title(f'3D QUBO Generation Landscape ({num_bits}-bit Discretization)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(num_units))
    ax.set_yticklabels([f'Unit {u}' for u in units])
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Generation (MW)', rotation=270, labelpad=20, fontweight='bold')
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"advanced_1_3d_qubo_generation_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 2: QUBO vs Continuous Comparison (3D)
    # ============================================================================
    if problem2_generation is not None:
        print("Creating Advanced Visualization 2: QUBO vs Continuous 3D Comparison...")
        fig = plt.figure(figsize=(20, 12))
        
        # Left: QUBO solution
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        X, Y = np.meshgrid(periods, range(num_units))
        Z_qubo = generation
        
        surf1 = ax1.plot_surface(X, Y, Z_qubo, cmap='Reds', alpha=0.7, 
                                 linewidth=0.3, antialiased=True, edgecolor='black', linewidths=0.2)
        ax1.set_xlabel('Time Period (Hour)', labelpad=10, fontsize=11)
        ax1.set_ylabel('Unit Index', labelpad=10, fontsize=11)
        ax1.set_zlabel('Generation (MW)', labelpad=10, fontsize=11)
        ax1.set_title(f'QUBO Solution\n({num_bits}-bit, Cost: ${total_cost:.0f})', 
                      fontsize=13, fontweight='bold', pad=15)
        ax1.set_yticks(range(num_units))
        ax1.set_yticklabels([f'U{u}' for u in units], fontsize=9)
        ax1.view_init(elev=25, azim=45)
        
        # Right: Continuous solution (Problem 2)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        Z_cont = problem2_generation
        
        surf2 = ax2.plot_surface(X, Y, Z_cont, cmap='Blues', alpha=0.7, 
                                 linewidth=0.3, antialiased=True, edgecolor='black', linewidths=0.2)
        ax2.set_xlabel('Time Period (Hour)', labelpad=10, fontsize=11)
        ax2.set_ylabel('Unit Index', labelpad=10, fontsize=11)
        ax2.set_zlabel('Generation (MW)', labelpad=10, fontsize=11)
        ax2.set_title(f'Continuous Solution (Problem 2)\n(Cost: ${problem2_cost:.0f})', 
                      fontsize=13, fontweight='bold', pad=15)
        ax2.set_yticks(range(num_units))
        ax2.set_yticklabels([f'U{u}' for u in units], fontsize=9)
        ax2.view_init(elev=25, azim=45)
        
        plt.suptitle('QUBO vs Continuous Solution Comparison', 
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(output_dir, f"advanced_2_3d_comparison_{timestamp_str}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
        plt.close()
    
    # ============================================================================
    # Visualization 3: Discretization Error Surface
    # ============================================================================
    if problem2_generation is not None:
        print("Creating Advanced Visualization 3: Discretization Error Surface...")
        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(periods, range(num_units))
        Z_error = np.abs(generation - problem2_generation)
        
        surf = ax.plot_surface(X, Y, Z_error, cmap='YlOrRd', alpha=0.85, 
                              linewidth=0.5, antialiased=True, edgecolor='black', linewidths=0.3)
        
        # Highlight large errors
        large_errors = Z_error > Z_error.mean() + Z_error.std()
        for i in range(num_units):
            for t in range(num_periods):
                if large_errors[i, t]:
                    ax.scatter(periods[t], i, Z_error[i, t], 
                              c='red', s=100, alpha=0.9, edgecolors='black', linewidths=1)
        
        ax.set_xlabel('Time Period (Hour)', labelpad=12, fontweight='bold')
        ax.set_ylabel('Unit Index', labelpad=12, fontweight='bold')
        ax.set_zlabel('Absolute Error (MW)', labelpad=12, fontweight='bold')
        ax.set_title(f'Discretization Error Surface\n(Mean Error: {Z_error.mean():.2f} MW)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_yticks(range(num_units))
        ax.set_yticklabels([f'Unit {u}' for u in units])
        
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('Error (MW)', rotation=270, labelpad=20, fontweight='bold')
        
        ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        path = os.path.join(output_dir, f"advanced_3_3d_error_surface_{timestamp_str}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
        plt.close()
    
    # ============================================================================
    # Visualization 4: Constraint Violation Analysis (3D)
    # ============================================================================
    print("Creating Advanced Visualization 4: Constraint Violation Analysis...")
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Power balance violations
    pb_violations = np.zeros(num_periods)
    if 'power_balance' in verification and 'violations' in verification['power_balance']:
        for viol in verification['power_balance']['violations']:
            period_idx = viol['period'] - 1
            pb_violations[period_idx] = abs(viol['violation'])
    
    # Create violation surface
    X_viol = np.tile(periods, (num_units, 1))
    Y_viol = np.tile(np.arange(num_units).reshape(-1, 1), (1, num_periods))
    Z_viol = np.tile(pb_violations, (num_units, 1))
    
    surf = ax.plot_surface(X_viol, Y_viol, Z_viol, cmap='Reds', alpha=0.7, 
                          linewidth=0.3, antialiased=True, edgecolor='black', linewidths=0.2)
    
    # Add violation markers
    for t in range(num_periods):
        if pb_violations[t] > 0:
            ax.scatter(periods[t], num_units//2, pb_violations[t], 
                      c='darkred', s=200, alpha=0.9, edgecolors='black', linewidths=1,
                      marker='X')
    
    ax.set_xlabel('Time Period (Hour)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Unit Index', labelpad=12, fontweight='bold')
    ax.set_zlabel('Power Balance Violation (MW)', labelpad=12, fontweight='bold')
    ax.set_title('Constraint Violation Analysis (Power Balance)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(num_units))
    ax.set_yticklabels([f'Unit {u}' for u in units])
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Violation (MW)', rotation=270, labelpad=20, fontweight='bold')
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"advanced_4_3d_violations_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 5: Discrete Value Distribution (3D Histogram)
    # ============================================================================
    print("Creating Advanced Visualization 5: Discrete Value Distribution...")
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique discrete values for each unit
    unique_values = {}
    for i in range(num_units):
        unique_values[i] = np.unique(generation[i, :])
    
    # Create 3D bar chart showing discrete value usage
    xpos_list = []
    ypos_list = []
    zpos_list = []
    dx_list = []
    dy_list = []
    dz_list = []
    colors_list = []
    
    for i in range(num_units):
        for val in unique_values[i]:
            count = np.sum(generation[i, :] == val)
            xpos_list.append(val)
            ypos_list.append(i)
            zpos_list.append(0)
            dx_list.append(2.0)  # Width
            dy_list.append(0.6)   # Depth
            dz_list.append(count) # Height
            colors_list.append(colors_units[i])
    
    if xpos_list:
        ax.bar3d(xpos_list, ypos_list, zpos_list, 
                dx_list, dy_list, dz_list,
                color=colors_list, alpha=0.8, edgecolor='black', linewidth=0.5, shade=True)
    
    ax.set_xlabel('Generation Value (MW)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Unit Index', labelpad=12, fontweight='bold')
    ax.set_zlabel('Frequency (Periods)', labelpad=12, fontweight='bold')
    ax.set_title(f'Discrete Value Distribution\n({num_bits}-bit Discretization)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(num_units))
    ax.set_yticklabels([f'Unit {u}' for u in units])
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"advanced_5_3d_discrete_dist_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 6: Multi-Panel QUBO Analysis Dashboard
    # ============================================================================
    print("Creating Advanced Visualization 6: Multi-Panel QUBO Dashboard...")
    fig = plt.figure(figsize=(20, 14))
    
    # Panel 1: Generation vs Load
    ax1 = plt.subplot(2, 2, 1)
    total_generation = generation.sum(axis=0)
    ax1.plot(periods, load_demand, 'b-', linewidth=2.5, marker='o', markersize=5, 
            label='Load Demand', alpha=0.8)
    ax1.plot(periods, total_generation, 'r--', linewidth=2.5, marker='s', markersize=5, 
            label='QUBO Generation', alpha=0.8)
    if problem2_generation is not None:
        p2_total = problem2_generation.sum(axis=0)
        ax1.plot(periods, p2_total, 'g:', linewidth=2, marker='^', markersize=4, 
                label='Continuous (P2)', alpha=0.7)
    ax1.fill_between(periods, load_demand, total_generation, 
                     where=(total_generation >= load_demand), alpha=0.2, color='green')
    ax1.fill_between(periods, load_demand, total_generation, 
                     where=(total_generation < load_demand), alpha=0.2, color='red')
    ax1.set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Power (MW)', fontsize=11, fontweight='bold')
    ax1.set_title('Generation vs Load', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Power Balance Violations
    ax2 = plt.subplot(2, 2, 2)
    pb_violations_array = np.zeros(num_periods)
    if 'power_balance' in verification and 'violations' in verification['power_balance']:
        for viol in verification['power_balance']['violations']:
            period_idx = viol['period'] - 1
            pb_violations_array[period_idx] = viol['violation']
    
    colors_viol = ['red' if v > 0 else 'green' if v < 0 else 'gray' for v in pb_violations_array]
    bars = ax2.bar(periods, pb_violations_array, color=colors_viol, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Violation (MW)', fontsize=11, fontweight='bold')
    ax2.set_title('Power Balance Violations', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Unit Generation Heatmap
    ax3 = plt.subplot(2, 2, 3)
    im = ax3.imshow(generation, aspect='auto', cmap='viridis', interpolation='bilinear')
    ax3.set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Unit', fontsize=11, fontweight='bold')
    ax3.set_title('QUBO Generation Heatmap', fontsize=13, fontweight='bold')
    ax3.set_yticks(range(num_units))
    ax3.set_yticklabels([f'Unit {u}' for u in units])
    ax3.set_xticks(range(0, num_periods, 4))
    ax3.set_xticklabels(range(1, num_periods + 1, 4))
    plt.colorbar(im, ax=ax3, label='Generation (MW)')
    
    # Panel 4: Cost Comparison
    ax4 = plt.subplot(2, 2, 4)
    methods = ['QUBO\nSolution']
    costs = [total_cost]
    colors_cost_bar = ['#E74C3C']
    
    if problem2_cost:
        methods.append('Continuous\n(Problem 2)')
        costs.append(problem2_cost)
        colors_cost_bar.append('#3498DB')
    
    if comparison and 'cost_difference' in comparison:
        methods.append('Difference')
        costs.append(comparison['cost_difference'])
        colors_cost_bar.append('#2ECC71')
    
    bars = ax4.bar(methods, costs, color=colors_cost_bar[:len(methods)], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Cost ($)', fontsize=11, fontweight='bold')
    ax4.set_title('Cost Comparison', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.0f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    plt.suptitle(f'QUBO Solution Analysis Dashboard ({num_bits}-bit, {num_binary_vars} vars)', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"advanced_6_dashboard_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 7: Contour Comparison
    # ============================================================================
    print("Creating Advanced Visualization 7: Contour Comparison...")
    fig, axes = plt.subplots(1, 2 if problem2_generation is not None else 1, 
                            figsize=(20 if problem2_generation is not None else 14, 8))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # Left: QUBO contour
    X, Y = np.meshgrid(periods, range(num_units))
    Z = generation
    
    contour1 = axes[0].contourf(X, Y, Z, levels=15, cmap='viridis', alpha=0.85)
    contour_lines1 = axes[0].contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    axes[0].clabel(contour_lines1, inline=True, fontsize=7, fmt='%d')
    
    axes[0].set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Unit Index', fontsize=12, fontweight='bold')
    axes[0].set_title(f'QUBO Generation Contour ({num_bits}-bit)', 
                     fontsize=14, fontweight='bold')
    axes[0].set_yticks(range(num_units))
    axes[0].set_yticklabels([f'Unit {u}' for u in units])
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    cbar1 = plt.colorbar(contour1, ax=axes[0])
    cbar1.set_label('Generation (MW)', rotation=270, labelpad=20, fontweight='bold')
    
    # Right: Continuous contour (if available)
    if problem2_generation is not None and len(axes) > 1:
        Z_cont = problem2_generation
        
        contour2 = axes[1].contourf(X, Y, Z_cont, levels=15, cmap='plasma', alpha=0.85)
        contour_lines2 = axes[1].contour(X, Y, Z_cont, levels=15, colors='black', alpha=0.3, linewidths=0.5)
        
        axes[1].set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Unit Index', fontsize=12, fontweight='bold')
        axes[1].set_title('Continuous Generation Contour (Problem 2)', 
                         fontsize=14, fontweight='bold')
        axes[1].set_yticks(range(num_units))
        axes[1].set_yticklabels([f'Unit {u}' for u in units])
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        cbar2 = plt.colorbar(contour2, ax=axes[1])
        cbar2.set_label('Generation (MW)', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"advanced_7_contour_comparison_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 8: Multi-Angle 3D View
    # ============================================================================
    print("Creating Advanced Visualization 8: Multi-Angle 3D View...")
    fig = plt.figure(figsize=(20, 12))
    
    X, Y = np.meshgrid(periods, range(num_units))
    Z = generation
    
    # Four different viewing angles
    angles = [
        (25, 45, 'Isometric View'),
        (90, 0, 'Top View'),
        (0, 0, 'Front View'),
        (0, 90, 'Side View')
    ]
    
    for idx, (elev, azim, title) in enumerate(angles):
        ax = fig.add_subplot(2, 2, idx+1, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85, 
                              linewidth=0.3, antialiased=True, edgecolor='black', linewidths=0.2)
        
        # Add discrete markers
        for i in range(0, num_units, 2):  # Every other unit
            for t in range(0, num_periods, 4):  # Every 4th period
                ax.scatter(periods[t], i, generation[i, t], 
                          c='red', s=30, alpha=0.6, edgecolors='black', linewidths=0.3)
        
        ax.set_xlabel('Time Period', labelpad=8, fontsize=9)
        ax.set_ylabel('Unit Index', labelpad=8, fontsize=9)
        ax.set_zlabel('Generation (MW)', labelpad=8, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_yticks(range(num_units))
        ax.set_yticklabels([f'U{u}' for u in units], fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        
        if idx == 0:  # Only add colorbar to first subplot
            cbar = fig.colorbar(surf, ax=ax, shrink=0.8, aspect=15, pad=0.05)
            cbar.set_label('Generation (MW)', rotation=270, labelpad=15, fontsize=9)
    
    plt.suptitle(f'Multi-Angle QUBO Generation Landscape ({num_bits}-bit)', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"advanced_8_multi_angle_3d_{timestamp_str}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Additional Comparison Visualizations
    # ============================================================================
    
    if problem2_generation is not None:
        # ========================================================================
        # Visualization 9: Side-by-Side 3D Comparison (Detailed)
        # ========================================================================
        print("Creating Advanced Visualization 9: Detailed Side-by-Side 3D Comparison...")
        fig = plt.figure(figsize=(22, 12))
        
        X, Y = np.meshgrid(periods, range(num_units))
        
        # Left: QUBO with discrete markers
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        Z_qubo = generation
        surf1 = ax1.plot_surface(X, Y, Z_qubo, cmap='Reds', alpha=0.6, 
                                 linewidth=0.3, antialiased=True, edgecolor='black', linewidths=0.2)
        # Add all discrete points
        for i in range(num_units):
            for t in range(num_periods):
                ax1.scatter(periods[t], i, generation[i, t], 
                           c='darkred', s=80, alpha=0.9, edgecolors='black', linewidths=0.5)
        ax1.set_xlabel('Time Period (Hour)', labelpad=12, fontsize=12, fontweight='bold')
        ax1.set_ylabel('Unit Index', labelpad=12, fontsize=12, fontweight='bold')
        ax1.set_zlabel('Generation (MW)', labelpad=12, fontsize=12, fontweight='bold')
        ax1.set_title(f'QUBO Solution\n({num_bits}-bit Discrete)', 
                      fontsize=14, fontweight='bold', pad=15)
        ax1.set_yticks(range(num_units))
        ax1.set_yticklabels([f'Unit {u}' for u in units], fontsize=10)
        ax1.view_init(elev=25, azim=45)
        cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.1)
        cbar1.set_label('Generation (MW)', rotation=270, labelpad=20, fontweight='bold')
        
        # Right: Continuous solution
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        Z_cont = problem2_generation
        surf2 = ax2.plot_surface(X, Y, Z_cont, cmap='Blues', alpha=0.6, 
                                 linewidth=0.3, antialiased=True, edgecolor='black', linewidths=0.2)
        # Add sample points
        for i in range(0, num_units, 1):
            for t in range(0, num_periods, 2):
                ax2.scatter(periods[t], i, problem2_generation[i, t], 
                           c='darkblue', s=60, alpha=0.7, edgecolors='black', linewidths=0.3)
        ax2.set_xlabel('Time Period (Hour)', labelpad=12, fontsize=12, fontweight='bold')
        ax2.set_ylabel('Unit Index', labelpad=12, fontsize=12, fontweight='bold')
        ax2.set_zlabel('Generation (MW)', labelpad=12, fontsize=12, fontweight='bold')
        ax2.set_title('Continuous Solution\n(Problem 2)', 
                      fontsize=14, fontweight='bold', pad=15)
        ax2.set_yticks(range(num_units))
        ax2.set_yticklabels([f'Unit {u}' for u in units], fontsize=10)
        ax2.view_init(elev=25, azim=45)
        cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.1)
        cbar2.set_label('Generation (MW)', rotation=270, labelpad=20, fontweight='bold')
        
        plt.suptitle('Detailed 3D Comparison: QUBO vs Continuous', 
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(output_dir, f"advanced_9_detailed_3d_comparison_{timestamp_str}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
        plt.close()
        
        # ========================================================================
        # Visualization 10: Difference Heatmap (Absolute)
        # ========================================================================
        print("Creating Advanced Visualization 10: Absolute Difference Heatmap...")
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        
        diff_abs = np.abs(generation - problem2_generation)
        diff_rel = np.zeros_like(diff_abs)
        for i in range(num_units):
            for t in range(num_periods):
                if problem2_generation[i, t] > 1e-6:
                    diff_rel[i, t] = (diff_abs[i, t] / problem2_generation[i, t]) * 100
        
        # Top-left: Absolute difference heatmap
        im1 = axes[0, 0].imshow(diff_abs, aspect='auto', cmap='YlOrRd', interpolation='bilinear')
        axes[0, 0].set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Unit', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Absolute Difference Heatmap\n(QUBO - Continuous)', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].set_yticks(range(num_units))
        axes[0, 0].set_yticklabels([f'Unit {u}' for u in units])
        axes[0, 0].set_xticks(range(0, num_periods, 4))
        axes[0, 0].set_xticklabels(range(1, num_periods + 1, 4))
        # Add text annotations
        for i in range(num_units):
            for t in range(0, num_periods, 4):
                if diff_abs[i, t] > diff_abs.mean():
                    axes[0, 0].text(t, i, f'{diff_abs[i, t]:.1f}', 
                                   ha='center', va='center', fontsize=7, 
                                   color='white' if diff_abs[i, t] > diff_abs.max()/2 else 'black',
                                   fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=axes[0, 0])
        cbar1.set_label('Absolute Difference (MW)', rotation=270, labelpad=20, fontweight='bold')
        
        # Top-right: Relative difference heatmap (%)
        im2 = axes[0, 1].imshow(diff_rel, aspect='auto', cmap='RdYlGn_r', interpolation='bilinear',
                               vmin=0, vmax=min(100, diff_rel.max()))
        axes[0, 1].set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Unit', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Relative Difference Heatmap\n(Percentage Error)', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_yticks(range(num_units))
        axes[0, 1].set_yticklabels([f'Unit {u}' for u in units])
        axes[0, 1].set_xticks(range(0, num_periods, 4))
        axes[0, 1].set_xticklabels(range(1, num_periods + 1, 4))
        # Add text annotations
        for i in range(num_units):
            for t in range(0, num_periods, 4):
                if diff_rel[i, t] > diff_rel.mean():
                    axes[0, 1].text(t, i, f'{diff_rel[i, t]:.1f}%', 
                                   ha='center', va='center', fontsize=7, 
                                   color='white' if diff_rel[i, t] > diff_rel.max()/2 else 'black',
                                   fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=axes[0, 1])
        cbar2.set_label('Relative Difference (%)', rotation=270, labelpad=20, fontweight='bold')
        
        # Bottom-left: QUBO generation heatmap
        im3 = axes[1, 0].imshow(generation, aspect='auto', cmap='viridis', interpolation='bilinear')
        axes[1, 0].set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Unit', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('QUBO Generation Heatmap', fontsize=14, fontweight='bold')
        axes[1, 0].set_yticks(range(num_units))
        axes[1, 0].set_yticklabels([f'Unit {u}' for u in units])
        axes[1, 0].set_xticks(range(0, num_periods, 4))
        axes[1, 0].set_xticklabels(range(1, num_periods + 1, 4))
        cbar3 = plt.colorbar(im3, ax=axes[1, 0])
        cbar3.set_label('Generation (MW)', rotation=270, labelpad=20, fontweight='bold')
        
        # Bottom-right: Continuous generation heatmap
        im4 = axes[1, 1].imshow(problem2_generation, aspect='auto', cmap='plasma', 
                               interpolation='bilinear')
        axes[1, 1].set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Unit', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Continuous Generation Heatmap (Problem 2)', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_yticks(range(num_units))
        axes[1, 1].set_yticklabels([f'Unit {u}' for u in units])
        axes[1, 1].set_xticks(range(0, num_periods, 4))
        axes[1, 1].set_xticklabels(range(1, num_periods + 1, 4))
        cbar4 = plt.colorbar(im4, ax=axes[1, 1])
        cbar4.set_label('Generation (MW)', rotation=270, labelpad=20, fontweight='bold')
        
        plt.suptitle('Comprehensive Heatmap Comparison: QUBO vs Continuous', 
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(output_dir, f"advanced_10_heatmap_comparison_{timestamp_str}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
        plt.close()
        
        # ========================================================================
        # Visualization 11: 3D Difference Surface
        # ========================================================================
        print("Creating Advanced Visualization 11: 3D Difference Surface...")
        fig = plt.figure(figsize=(20, 14))
        
        # Create two subplots: absolute and relative difference
        X, Y = np.meshgrid(periods, range(num_units))
        Z_diff_abs = np.abs(generation - problem2_generation)
        
        # Left: Absolute difference
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_diff_abs, cmap='YlOrRd', alpha=0.85, 
                                linewidth=0.5, antialiased=True, edgecolor='black', linewidths=0.3)
        # Highlight large differences
        large_diff_mask = Z_diff_abs > (Z_diff_abs.mean() + Z_diff_abs.std())
        for i in range(num_units):
            for t in range(num_periods):
                if large_diff_mask[i, t]:
                    ax1.scatter(periods[t], i, Z_diff_abs[i, t], 
                               c='darkred', s=150, alpha=0.9, edgecolors='black', linewidths=1,
                               marker='X')
        ax1.set_xlabel('Time Period (Hour)', labelpad=12, fontsize=12, fontweight='bold')
        ax1.set_ylabel('Unit Index', labelpad=12, fontsize=12, fontweight='bold')
        ax1.set_zlabel('Absolute Difference (MW)', labelpad=12, fontsize=12, fontweight='bold')
        ax1.set_title(f'Absolute Difference Surface\n(Mean: {Z_diff_abs.mean():.2f} MW)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.set_yticks(range(num_units))
        ax1.set_yticklabels([f'Unit {u}' for u in units], fontsize=10)
        ax1.view_init(elev=25, azim=45)
        cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.1)
        cbar1.set_label('Difference (MW)', rotation=270, labelpad=20, fontweight='bold')
        
        # Right: Relative difference
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        Z_diff_rel = np.zeros_like(Z_diff_abs)
        for i in range(num_units):
            for t in range(num_periods):
                if problem2_generation[i, t] > 1e-6:
                    Z_diff_rel[i, t] = (Z_diff_abs[i, t] / problem2_generation[i, t]) * 100
        
        surf2 = ax2.plot_surface(X, Y, Z_diff_rel, cmap='RdYlGn_r', alpha=0.85, 
                                linewidth=0.5, antialiased=True, edgecolor='black', linewidths=0.3)
        # Highlight large relative differences
        large_rel_mask = Z_diff_rel > (Z_diff_rel.mean() + Z_diff_rel.std())
        for i in range(num_units):
            for t in range(num_periods):
                if large_rel_mask[i, t] and problem2_generation[i, t] > 1e-6:
                    ax2.scatter(periods[t], i, Z_diff_rel[i, t], 
                               c='darkred', s=150, alpha=0.9, edgecolors='black', linewidths=1,
                               marker='X')
        ax2.set_xlabel('Time Period (Hour)', labelpad=12, fontsize=12, fontweight='bold')
        ax2.set_ylabel('Unit Index', labelpad=12, fontsize=12, fontweight='bold')
        ax2.set_zlabel('Relative Difference (%)', labelpad=12, fontsize=12, fontweight='bold')
        ax2.set_title(f'Relative Difference Surface\n(Mean: {Z_diff_rel.mean():.2f}%)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.set_yticks(range(num_units))
        ax2.set_yticklabels([f'Unit {u}' for u in units], fontsize=10)
        ax2.view_init(elev=25, azim=45)
        cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.1)
        cbar2.set_label('Difference (%)', rotation=270, labelpad=20, fontweight='bold')
        
        plt.suptitle('3D Difference Surfaces: Absolute and Relative', 
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(output_dir, f"advanced_11_3d_difference_surface_{timestamp_str}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
        plt.close()
        
        # ========================================================================
        # Visualization 12: Unit-wise Comparison Heatmap
        # ========================================================================
        print("Creating Advanced Visualization 12: Unit-wise Comparison Heatmap...")
        fig, axes = plt.subplots(num_units, 3, figsize=(22, 3*num_units))
        
        if num_units == 1:
            axes = axes.reshape(1, -1)
        
        for i, unit_id in enumerate(units):
            # Column 1: QUBO generation
            axes[i, 0].plot(periods, generation[i, :], 'r-o', linewidth=2.5, 
                           markersize=6, label='QUBO', alpha=0.8)
            axes[i, 0].set_ylabel(f'Unit {unit_id}\nGeneration (MW)', 
                                 fontsize=11, fontweight='bold')
            if i == 0:
                axes[i, 0].set_title('QUBO Solution', fontsize=13, fontweight='bold')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].set_ylim([0, max(generation[i, :].max(), 
                                       problem2_generation[i, :].max()) * 1.1])
            if i == num_units - 1:
                axes[i, 0].set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
            
            # Column 2: Continuous generation
            axes[i, 1].plot(periods, problem2_generation[i, :], 'b-s', linewidth=2.5, 
                           markersize=6, label='Continuous', alpha=0.8)
            if i == 0:
                axes[i, 1].set_title('Continuous Solution', fontsize=13, fontweight='bold')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].set_ylim([0, max(generation[i, :].max(), 
                                       problem2_generation[i, :].max()) * 1.1])
            if i == num_units - 1:
                axes[i, 1].set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
            
            # Column 3: Difference
            diff_unit = generation[i, :] - problem2_generation[i, :]
            colors_diff = ['red' if d > 0 else 'blue' if d < 0 else 'gray' for d in diff_unit]
            bars = axes[i, 2].bar(periods, diff_unit, color=colors_diff, alpha=0.7, edgecolor='black')
            axes[i, 2].axhline(y=0, color='black', linestyle='-', linewidth=1)
            if i == 0:
                axes[i, 2].set_title('Difference (QUBO - Continuous)', 
                                    fontsize=13, fontweight='bold')
            axes[i, 2].grid(True, alpha=0.3, axis='y')
            if i == num_units - 1:
                axes[i, 2].set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
            axes[i, 2].set_ylabel('Difference (MW)', fontsize=10, fontweight='bold')
        
        plt.suptitle('Unit-wise Detailed Comparison: QUBO vs Continuous', 
                     fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        path = os.path.join(output_dir, f"advanced_12_unitwise_comparison_{timestamp_str}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
        plt.close()
        
        # ========================================================================
        # Visualization 13: Multi-Metric 3D Comparison
        # ========================================================================
        print("Creating Advanced Visualization 13: Multi-Metric 3D Comparison...")
        fig = plt.figure(figsize=(22, 14))
        
        metrics = [
            (generation, problem2_generation, 'viridis', 'Reds', 'Blues', 
             'Generation (MW)', 'QUBO Generation', 'Continuous Generation'),
            (np.abs(generation - problem2_generation), None, 'YlOrRd', None, None,
             'Absolute Error (MW)', 'Absolute Difference', None),
        ]
        
        for idx, (data1, data2, cmap1, cmap2, cmap3, zlabel, title1, title2) in enumerate(metrics):
            if idx == 0:  # Generation comparison
                # Left: QUBO
                ax1 = fig.add_subplot(2, 2, 1, projection='3d')
                X, Y = np.meshgrid(periods, range(num_units))
                Z1 = data1
                surf1 = ax1.plot_surface(X, Y, Z1, cmap=cmap2, alpha=0.7, 
                                        linewidth=0.3, antialiased=True)
                ax1.set_title(title1, fontsize=12, fontweight='bold', pad=10)
                ax1.set_xlabel('Time Period', fontsize=10)
                ax1.set_ylabel('Unit', fontsize=10)
                ax1.set_zlabel(zlabel, fontsize=10)
                ax1.view_init(elev=25, azim=45)
                
                # Right: Continuous
                ax2 = fig.add_subplot(2, 2, 2, projection='3d')
                Z2 = data2
                surf2 = ax2.plot_surface(X, Y, Z2, cmap=cmap3, alpha=0.7, 
                                        linewidth=0.3, antialiased=True)
                ax2.set_title(title2, fontsize=12, fontweight='bold', pad=10)
                ax2.set_xlabel('Time Period', fontsize=10)
                ax2.set_ylabel('Unit', fontsize=10)
                ax2.set_zlabel(zlabel, fontsize=10)
                ax2.view_init(elev=25, azim=45)
            else:  # Difference
                # Bottom: Absolute difference
                ax3 = fig.add_subplot(2, 2, 3, projection='3d')
                Z_diff = data1
                surf3 = ax3.plot_surface(X, Y, Z_diff, cmap=cmap1, alpha=0.85, 
                                        linewidth=0.3, antialiased=True)
                ax3.set_title(title1, fontsize=12, fontweight='bold', pad=10)
                ax3.set_xlabel('Time Period', fontsize=10)
                ax3.set_ylabel('Unit', fontsize=10)
                ax3.set_zlabel(zlabel, fontsize=10)
                ax3.view_init(elev=25, azim=45)
                
                # Bottom-right: Relative difference
                ax4 = fig.add_subplot(2, 2, 4, projection='3d')
                Z_rel = np.zeros_like(Z_diff)
                for i in range(num_units):
                    for t in range(num_periods):
                        if problem2_generation[i, t] > 1e-6:
                            Z_rel[i, t] = (Z_diff[i, t] / problem2_generation[i, t]) * 100
                surf4 = ax4.plot_surface(X, Y, Z_rel, cmap='RdYlGn_r', alpha=0.85, 
                                         linewidth=0.3, antialiased=True)
                ax4.set_title('Relative Difference (%)', fontsize=12, fontweight='bold', pad=10)
                ax4.set_xlabel('Time Period', fontsize=10)
                ax4.set_ylabel('Unit', fontsize=10)
                ax4.set_zlabel('Relative Error (%)', fontsize=10)
                ax4.view_init(elev=25, azim=45)
        
        plt.suptitle('Multi-Metric 3D Comparison Dashboard', 
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(output_dir, f"advanced_13_multimetric_3d_{timestamp_str}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
        plt.close()
    
    print("\n" + "=" * 70)
    print("All advanced visualizations generated successfully!")
    print(f"Results directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate advanced visualizations for Problem 3 results'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory containing Problem 3 results (default: ../results/problem3)'
    )
    parser.add_argument(
        '--timestamp',
        type=str,
        default=None,
        help='Specific timestamp to visualize (default: latest)'
    )
    parser.add_argument(
        '--problem2-dir',
        type=str,
        default=None,
        help='Directory containing Problem 2 results for comparison'
    )
    
    args = parser.parse_args()
    
    # Set default directories
    if args.results_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        args.results_dir = os.path.join(project_root, 'results', 'problem3')
    
    if args.problem2_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        args.problem2_dir = os.path.join(project_root, 'results', 'problem2')
    
    create_advanced_visualizations(
        results_dir=args.results_dir,
        timestamp=args.timestamp,
        problem2_results_dir=args.problem2_dir
    )

