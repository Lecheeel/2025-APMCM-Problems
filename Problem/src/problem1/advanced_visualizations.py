"""
Advanced Visualizations for Problem 1
=======================================
Generates high-quality 3D and advanced 2D visualizations suitable for academic papers.
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    # For newer matplotlib versions, Axes3D is automatically available
    pass
from matplotlib.colors import LinearSegmentedColormap
import os


def create_advanced_visualizations(unit_generation, unit_status_matrix, load_demand, 
                                   fuel_cost_total, startup_cost_total, shutdown_cost_total,
                                   a_coeff, b_coeff, c_coeff, units, periods, 
                                   results_dir, timestamp):
    """
    Generate advanced visualizations including 3D plots.
    
    Args:
        unit_generation: Array of shape (num_units, num_periods) with generation values
        unit_status_matrix: Array of shape (num_units, num_periods) with unit status
        load_demand: Array of shape (num_periods,) with load values
        fuel_cost_total, startup_cost_total, shutdown_cost_total: Cost values
        a_coeff, b_coeff, c_coeff: Cost coefficients for units
        units: List of unit IDs
        periods: Array of period numbers
        results_dir: Directory to save visualizations
        timestamp: Timestamp for file naming
    """
    num_units = len(units)
    num_periods = len(periods)
    
    # Professional color schemes
    colors_units = plt.cm.viridis(np.linspace(0, 0.9, num_units))
    colors_cost = ['#2E86AB', '#A23B72', '#F18F01']  # Professional blue, purple, orange
    
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
    
    # ============================================================================
    # Visualization 1: 3D Surface Plot - Generation Landscape
    # ============================================================================
    print("Creating Advanced Visualization 1: 3D Generation Surface...")
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    X, Y = np.meshgrid(periods, range(num_units))
    Z = unit_generation
    
    # Create surface plot with gradient colormap
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85, 
                          linewidth=0.5, antialiased=True, edgecolor='black', linewidths=0.3)
    
    # Add contour lines on the base
    ax.contour(X, Y, Z, zdir='z', offset=Z.min()-20, cmap='viridis', alpha=0.5, linewidths=1)
    
    ax.set_xlabel('Time Period (Hour)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Unit Index', labelpad=12, fontweight='bold')
    ax.set_zlabel('Power Generation (MW)', labelpad=12, fontweight='bold')
    ax.set_title('3D Generation Landscape: Power Output Across Units and Time', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set custom tick labels for units
    ax.set_yticks(range(num_units))
    ax.set_yticklabels([f'Unit {u}' for u in units])
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Generation (MW)', rotation=270, labelpad=20, fontweight='bold')
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    path = os.path.join(results_dir, f"advanced_1_3d_generation_surface_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 2: 3D Bar Chart - Unit Commitment Status
    # ============================================================================
    print("Creating Advanced Visualization 2: 3D Unit Commitment Bar Chart...")
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create 3D bar chart
    xpos, ypos = np.meshgrid(periods, range(num_units))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    
    dx = dy = 0.8
    dz = unit_status_matrix.flatten()
    
    # Color bars based on status
    colors_bar = ['#2ECC71' if d > 0.5 else '#E74C3C' for d in dz]  # Green for ON, Red for OFF
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors_bar, alpha=0.8, 
             edgecolor='black', linewidth=0.5, shade=True)
    
    ax.set_xlabel('Time Period (Hour)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Unit Index', labelpad=12, fontweight='bold')
    ax.set_zlabel('Status (1=ON, 0=OFF)', labelpad=12, fontweight='bold')
    ax.set_title('3D Unit Commitment Status Visualization', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(num_units))
    ax.set_yticklabels([f'Unit {u}' for u in units])
    ax.set_zlim(0, 1.2)
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    path = os.path.join(results_dir, f"advanced_2_3d_unit_commitment_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 3: 3D Scatter Plot - Cost vs Generation vs Time
    # ============================================================================
    print("Creating Advanced Visualization 3: 3D Cost-Generation-Time Scatter...")
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate hourly costs for each unit
    for i, unit_id in enumerate(units):
        hourly_costs = []
        hourly_generation = []
        hourly_periods = []
        
        for t in range(num_periods):
            gen = unit_generation[i, t]
            if gen > 0.1:  # Only plot when unit is generating
                cost = a_coeff[i] * gen * gen + b_coeff[i] * gen + c_coeff[i]
                hourly_costs.append(cost)
                hourly_generation.append(gen)
                hourly_periods.append(periods[t])
        
        if hourly_costs:
            ax.scatter(hourly_periods, hourly_generation, hourly_costs, 
                      c=colors_units[i], s=100, alpha=0.7, 
                      label=f'Unit {unit_id}', edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('Time Period (Hour)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Power Generation (MW)', labelpad=12, fontweight='bold')
    ax.set_zlabel('Hourly Fuel Cost ($)', labelpad=12, fontweight='bold')
    ax.set_title('3D Cost-Generation-Time Relationship', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    path = os.path.join(results_dir, f"advanced_3_3d_cost_scatter_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 4: Contour Plot - Generation Contours
    # ============================================================================
    print("Creating Advanced Visualization 4: Generation Contour Plot...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    X, Y = np.meshgrid(periods, range(num_units))
    Z = unit_generation
    
    # Create contour plot
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.85)
    contour_lines = ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%d')
    
    # Add load demand line
    ax2 = ax.twinx()
    ax2.plot(periods, load_demand, 'r-', linewidth=3, marker='o', markersize=6, 
             label='Load Demand', alpha=0.9)
    ax2.set_ylabel('Load Demand (MW)', fontsize=12, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right', fontsize=10)
    
    ax.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unit Index', fontsize=12, fontweight='bold')
    ax.set_title('Generation Contour Map with Load Demand Overlay', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_yticks(range(num_units))
    ax.set_yticklabels([f'Unit {u}' for u in units])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    cbar = plt.colorbar(contour, ax=ax, pad=0.02)
    cbar.set_label('Power Generation (MW)', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(results_dir, f"advanced_4_contour_generation_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 5: 3D Cost Surface - Total Cost Landscape
    # ============================================================================
    print("Creating Advanced Visualization 5: 3D Cost Surface...")
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate total hourly cost (fuel + startup + shutdown)
    hourly_total_cost = np.zeros(num_periods)
    for t in range(num_periods):
        fuel_cost_t = sum(a_coeff[i] * unit_generation[i, t] * unit_generation[i, t] + 
                          b_coeff[i] * unit_generation[i, t] + c_coeff[i]
                          for i in range(num_units))
        # Note: startup/shutdown costs are event-based, not hourly
        # For visualization, we'll use fuel cost as proxy
        hourly_total_cost[t] = fuel_cost_t
    
    # Create surface showing cost vs generation vs time
    X = np.tile(periods, (num_units, 1))
    Y = np.tile(np.arange(num_units).reshape(-1, 1), (1, num_periods))
    Z_cost = np.zeros_like(X)
    
    for i in range(num_units):
        for t in range(num_periods):
            gen = unit_generation[i, t]
            Z_cost[i, t] = a_coeff[i] * gen * gen + b_coeff[i] * gen + c_coeff[i]
    
    surf = ax.plot_surface(X, Y, Z_cost, cmap='plasma', alpha=0.9, 
                          linewidth=0.3, antialiased=True, edgecolor='black', linewidths=0.2)
    
    ax.set_xlabel('Time Period (Hour)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Unit Index', labelpad=12, fontweight='bold')
    ax.set_zlabel('Hourly Fuel Cost ($)', labelpad=12, fontweight='bold')
    ax.set_title('3D Cost Surface: Fuel Cost Distribution Across Units and Time', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(num_units))
    ax.set_yticklabels([f'Unit {u}' for u in units])
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Fuel Cost ($)', rotation=270, labelpad=20, fontweight='bold')
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    path = os.path.join(results_dir, f"advanced_5_3d_cost_surface_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 6: Multi-Panel 3D View - Different Angles
    # ============================================================================
    print("Creating Advanced Visualization 6: Multi-Angle 3D View...")
    fig = plt.figure(figsize=(20, 12))
    
    X, Y = np.meshgrid(periods, range(num_units))
    Z = unit_generation
    
    # Four different viewing angles
    angles = [
        (25, 45, 'Isometric View'),
        (90, 0, 'Top View (Time vs Unit)'),
        (0, 0, 'Front View (Time vs Generation)'),
        (0, 90, 'Side View (Unit vs Generation)')
    ]
    
    for idx, (elev, azim, title) in enumerate(angles):
        ax = fig.add_subplot(2, 2, idx+1, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85, 
                              linewidth=0.3, antialiased=True, edgecolor='black', linewidths=0.2)
        
        ax.set_xlabel('Time Period (Hour)', labelpad=8, fontsize=10)
        ax.set_ylabel('Unit Index', labelpad=8, fontsize=10)
        ax.set_zlabel('Generation (MW)', labelpad=8, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_yticks(range(num_units))
        ax.set_yticklabels([f'U{u}' for u in units], fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        
        if idx == 0:  # Only add colorbar to first subplot
            cbar = fig.colorbar(surf, ax=ax, shrink=0.8, aspect=15, pad=0.05)
            cbar.set_label('Generation (MW)', rotation=270, labelpad=15, fontsize=9)
    
    plt.suptitle('Multi-Angle 3D Generation Landscape', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(results_dir, f"advanced_6_multi_angle_3d_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 7: Heatmap with Load Overlay - Enhanced
    # ============================================================================
    print("Creating Advanced Visualization 7: Enhanced Heatmap with Load...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1])
    
    # Top: Generation heatmap
    im1 = ax1.imshow(unit_generation, aspect='auto', cmap='YlOrRd', 
                     interpolation='bilinear', origin='lower')
    ax1.set_ylabel('Unit', fontsize=12, fontweight='bold')
    ax1.set_title('Enhanced Generation Heatmap with Load Profile', 
                   fontsize=16, fontweight='bold', pad=15)
    ax1.set_yticks(range(num_units))
    ax1.set_yticklabels([f'Unit {u}' for u in units])
    ax1.set_xticks(range(0, num_periods, 2))
    ax1.set_xticklabels(range(1, num_periods + 1, 2))
    
    # Add generation values as text (for key periods)
    for i in range(num_units):
        for t in range(0, num_periods, 4):  # Every 4th period
            if unit_generation[i, t] > 5:  # Only show if significant
                ax1.text(t, i, f'{int(unit_generation[i, t])}', 
                        ha='center', va='center', fontsize=7, 
                        color='white' if unit_generation[i, t] > unit_generation.max()/2 else 'black',
                        fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
    cbar1.set_label('Power Generation (MW)', rotation=270, labelpad=20, fontweight='bold')
    
    # Bottom: Load profile with generation overlay
    ax2.fill_between(periods, 0, load_demand, alpha=0.3, color='blue', label='Load Demand')
    ax2.plot(periods, load_demand, 'b-', linewidth=2.5, marker='o', markersize=5)
    
    total_generation = unit_generation.sum(axis=0)
    ax2.plot(periods, total_generation, 'r--', linewidth=2, marker='s', 
             markersize=4, label='Total Generation', alpha=0.8)
    
    ax2.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Power (MW)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0.5, num_periods + 0.5)
    
    plt.tight_layout()
    path = os.path.join(results_dir, f"advanced_7_enhanced_heatmap_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 8: Cost Breakdown - 3D Pie Chart Alternative
    # ============================================================================
    print("Creating Advanced Visualization 8: 3D Cost Breakdown...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data
    cost_labels = ['Fuel Cost', 'Startup Cost', 'Shutdown Cost']
    cost_values = [fuel_cost_total, startup_cost_total, shutdown_cost_total]
    
    # Filter out zero costs
    non_zero = [(label, val) for label, val in zip(cost_labels, cost_values) if val > 1e-6]
    if not non_zero:
        print("  ⚠ Skipping cost breakdown (no costs to display)")
        return
    
    labels, values = zip(*non_zero)
    colors_3d = colors_cost[:len(labels)]
    
    # Create 3D bar chart for cost breakdown
    xpos = np.arange(len(labels))
    ypos = np.zeros(len(labels))
    zpos = np.zeros(len(labels))
    
    dx = dy = 0.5
    dz = np.array(values)
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors_3d, alpha=0.9, 
             edgecolor='black', linewidth=1.5, shade=True)
    
    # Add value labels
    for i, (label, val) in enumerate(zip(labels, values)):
        ax.text(i, 0, val + max(dz)*0.05, f'{label}\n${val:.0f}', 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Cost Category', labelpad=12, fontweight='bold')
    ax.set_ylabel('', labelpad=12)
    ax.set_zlabel('Cost ($)', labelpad=12, fontweight='bold')
    ax.set_title('3D Cost Breakdown Visualization', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(xpos)
    ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=10)
    ax.set_yticks([])
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    path = os.path.join(results_dir, f"advanced_8_3d_cost_breakdown_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("All advanced visualizations generated successfully!")
    print("=" * 70)

