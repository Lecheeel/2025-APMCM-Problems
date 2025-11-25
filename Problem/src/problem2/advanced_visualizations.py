"""
Advanced Visualizations for Problem 2
======================================
Generates high-quality 3D and advanced 2D visualizations suitable for academic papers.
Includes network-specific visualizations.
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
                                   spinning_reserve_available, spinning_reserve_required,
                                   line_flows_matrix, branches, bus_angles_matrix, all_buses,
                                   bus_to_idx, results_dir, timestamp):
    """
    Generate advanced visualizations including 3D plots for Problem 2.
    
    Args:
        unit_generation: Array of shape (num_units, num_periods) with generation values
        unit_status_matrix: Array of shape (num_units, num_periods) with unit status
        load_demand: Array of shape (num_periods,) with load values
        fuel_cost_total, startup_cost_total, shutdown_cost_total: Cost values
        a_coeff, b_coeff, c_coeff: Cost coefficients for units
        units: List of unit IDs
        periods: Array of period numbers
        spinning_reserve_available: Array of available reserve
        spinning_reserve_required: Array of required reserve
        line_flows_matrix: Array of shape (num_branches, num_periods) with line flows
        branches: List of branch dictionaries
        bus_angles_matrix: Array of shape (num_buses, num_periods) with bus angles
        all_buses: List of all bus numbers
        bus_to_idx: Dictionary mapping bus numbers to indices
        results_dir: Directory to save visualizations
        timestamp: Timestamp for file naming
    """
    num_units = len(units)
    num_periods = len(periods)
    num_branches = len(branches)
    num_buses = len(all_buses)
    
    # Professional color schemes
    colors_units = plt.cm.viridis(np.linspace(0, 0.9, num_units))
    colors_cost = ['#2E86AB', '#A23B72', '#F18F01']  # Professional blue, purple, orange
    colors_network = plt.cm.plasma(np.linspace(0, 0.9, num_branches))
    
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
    # Visualization 1: 3D Generation Surface with Reserve Overlay
    # ============================================================================
    print("Creating Advanced Visualization 1: 3D Generation Surface with Reserve...")
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(periods, range(num_units))
    Z = unit_generation
    
    # Main surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.75, 
                          linewidth=0.5, antialiased=True, edgecolor='black', linewidths=0.3)
    
    # Add reserve capacity as transparent overlay
    reserve_capacity = np.zeros_like(Z)
    for i in range(num_units):
        for t in range(num_periods):
            if unit_status_matrix[i, t] > 0.5:
                # Reserve = P_max - current generation (simplified)
                reserve_capacity[i, t] = spinning_reserve_available[t] / num_units  # Approximate
    
    # Reserve surface (semi-transparent)
    surf2 = ax.plot_surface(X, Y, Z + reserve_capacity, cmap='Reds', alpha=0.3, 
                            linewidth=0.3, antialiased=True)
    
    ax.set_xlabel('Time Period (Hour)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Unit Index', labelpad=12, fontweight='bold')
    ax.set_zlabel('Power (MW)', labelpad=12, fontweight='bold')
    ax.set_title('3D Generation Landscape with Spinning Reserve Overlay', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(num_units))
    ax.set_yticklabels([f'Unit {u}' for u in units])
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Generation (MW)', rotation=270, labelpad=20, fontweight='bold')
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    path = os.path.join(results_dir, f"advanced_1_3d_generation_reserve_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 2: 3D Line Flow Surface
    # ============================================================================
    print("Creating Advanced Visualization 2: 3D Line Flow Surface...")
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Select top 20 lines by average flow
    avg_flows = np.abs(line_flows_matrix).mean(axis=1)
    top_lines_idx = np.argsort(avg_flows)[-20:][::-1]
    top_flows = line_flows_matrix[top_lines_idx, :]
    
    X_line, Y_line = np.meshgrid(periods, range(len(top_lines_idx)))
    Z_line = top_flows
    
    # Normalize by capacity for visualization
    Z_line_pct = np.zeros_like(Z_line)
    for idx, br_idx in enumerate(top_lines_idx):
        if branches[br_idx]['P_max'] > 0:
            Z_line_pct[idx, :] = Z_line[idx, :] / branches[br_idx]['P_max'] * 100
    
    surf = ax.plot_surface(X_line, Y_line, Z_line_pct, cmap='RdYlGn_r', alpha=0.85, 
                          linewidth=0.5, antialiased=True, edgecolor='black', linewidths=0.3)
    
    # Add zero plane
    ax.plot_surface(X_line, Y_line, np.zeros_like(Z_line_pct), 
                    color='gray', alpha=0.2, linewidth=0.1)
    
    line_labels = [f"Br{top_lines_idx[i]+1}" for i in range(len(top_lines_idx))]
    ax.set_xlabel('Time Period (Hour)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Transmission Line', labelpad=12, fontweight='bold')
    ax.set_zlabel('Flow Utilization (%)', labelpad=12, fontweight='bold')
    ax.set_title('3D Line Flow Utilization Surface (Top 20 Lines)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(len(top_lines_idx)))
    ax.set_yticklabels(line_labels, fontsize=8)
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Flow Utilization (%)', rotation=270, labelpad=20, fontweight='bold')
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    path = os.path.join(results_dir, f"advanced_2_3d_line_flow_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 3: 3D Bus Voltage Angle Surface
    # ============================================================================
    print("Creating Advanced Visualization 3: 3D Bus Voltage Angle Surface...")
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Select key buses (generator buses + some load buses)
    key_buses = [1, 2, 5, 8, 11, 13, 3, 4, 7, 9, 10, 12, 15, 20]
    key_bus_indices = [bus_to_idx[bus] for bus in key_buses if bus in bus_to_idx]
    key_bus_labels = [f'Bus {bus}' for bus in key_buses if bus in bus_to_idx]
    
    X_bus, Y_bus = np.meshgrid(periods, range(len(key_bus_indices)))
    Z_bus = bus_angles_matrix[key_bus_indices, :] * 180 / np.pi  # Convert to degrees
    
    surf = ax.plot_surface(X_bus, Y_bus, Z_bus, cmap='coolwarm', alpha=0.85, 
                          linewidth=0.5, antialiased=True, edgecolor='black', linewidths=0.3)
    
    # Add reference plane at zero
    ax.plot_surface(X_bus, Y_bus, np.zeros_like(Z_bus), 
                    color='black', alpha=0.1, linewidth=0.1)
    
    ax.set_xlabel('Time Period (Hour)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Bus Index', labelpad=12, fontweight='bold')
    ax.set_zlabel('Voltage Angle (degrees)', labelpad=12, fontweight='bold')
    ax.set_title('3D Bus Voltage Angle Surface (DC Power Flow)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(len(key_bus_indices)))
    ax.set_yticklabels(key_bus_labels, fontsize=8)
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Voltage Angle (degrees)', rotation=270, labelpad=20, fontweight='bold')
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    path = os.path.join(results_dir, f"advanced_3_3d_bus_angles_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 4: 3D Spinning Reserve Landscape
    # ============================================================================
    print("Creating Advanced Visualization 4: 3D Spinning Reserve Landscape...")
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create reserve matrix per unit
    reserve_per_unit = np.zeros((num_units, num_periods))
    for i in range(num_units):
        for t in range(num_periods):
            if unit_status_matrix[i, t] > 0.5:
                # Approximate reserve per unit
                reserve_per_unit[i, t] = (spinning_reserve_available[t] / 
                                         max(1, unit_status_matrix[:, t].sum()))
    
    X_res, Y_res = np.meshgrid(periods, range(num_units))
    Z_res = reserve_per_unit
    
    # Surface plot
    surf = ax.plot_surface(X_res, Y_res, Z_res, cmap='YlOrRd', alpha=0.85, 
                          linewidth=0.5, antialiased=True, edgecolor='black', linewidths=0.3)
    
    # Add required reserve as wireframe
    required_reserve_per_unit = np.tile(spinning_reserve_required / num_units, 
                                       (num_units, 1))
    ax.plot_wireframe(X_res, Y_res, required_reserve_per_unit, 
                     color='red', alpha=0.5, linewidth=1.5, label='Required Reserve')
    
    ax.set_xlabel('Time Period (Hour)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Unit Index', labelpad=12, fontweight='bold')
    ax.set_zlabel('Spinning Reserve (MW)', labelpad=12, fontweight='bold')
    ax.set_title('3D Spinning Reserve Landscape: Available vs Required', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(num_units))
    ax.set_yticklabels([f'Unit {u}' for u in units])
    ax.legend(loc='upper left', fontsize=9)
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Available Reserve (MW)', rotation=270, labelpad=20, fontweight='bold')
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    path = os.path.join(results_dir, f"advanced_4_3d_reserve_landscape_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 5: Multi-Panel Network Analysis
    # ============================================================================
    print("Creating Advanced Visualization 5: Multi-Panel Network Analysis...")
    fig = plt.figure(figsize=(20, 14))
    
    # Panel 1: Line flow heatmap (top 15 lines)
    ax1 = plt.subplot(2, 2, 1)
    avg_flows = np.abs(line_flows_matrix).mean(axis=1)
    top_lines_idx = np.argsort(avg_flows)[-15:][::-1]
    top_flows_pct = np.zeros((len(top_lines_idx), num_periods))
    
    for idx, br_idx in enumerate(top_lines_idx):
        if branches[br_idx]['P_max'] > 0:
            top_flows_pct[idx, :] = (line_flows_matrix[br_idx, :] / 
                                    branches[br_idx]['P_max'] * 100)
    
    im1 = ax1.imshow(top_flows_pct, aspect='auto', cmap='RdYlGn_r', 
                     interpolation='bilinear', vmin=-100, vmax=100)
    ax1.set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Transmission Line', fontsize=11, fontweight='bold')
    ax1.set_title('Line Flow Utilization Heatmap', fontsize=13, fontweight='bold')
    ax1.set_yticks(range(len(top_lines_idx)))
    ax1.set_yticklabels([f'Br{top_lines_idx[i]+1}' for i in range(len(top_lines_idx))], 
                        fontsize=8)
    plt.colorbar(im1, ax=ax1, label='Utilization (%)')
    
    # Panel 2: Bus angle heatmap
    ax2 = plt.subplot(2, 2, 2)
    key_buses = [1, 2, 5, 8, 11, 13, 3, 4, 7, 9, 10, 12]
    key_bus_indices = [bus_to_idx[bus] for bus in key_buses if bus in bus_to_idx]
    bus_angles_deg = bus_angles_matrix[key_bus_indices, :] * 180 / np.pi
    
    im2 = ax2.imshow(bus_angles_deg, aspect='auto', cmap='coolwarm', 
                     interpolation='bilinear')
    ax2.set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Bus', fontsize=11, fontweight='bold')
    ax2.set_title('Bus Voltage Angle Heatmap', fontsize=13, fontweight='bold')
    ax2.set_yticks(range(len(key_bus_indices)))
    ax2.set_yticklabels([f'Bus {key_buses[i]}' for i in range(len(key_bus_indices))], 
                        fontsize=8)
    plt.colorbar(im2, ax=ax2, label='Angle (degrees)')
    
    # Panel 3: Spinning reserve comparison
    ax3 = plt.subplot(2, 2, 3)
    ax3.fill_between(periods, 0, spinning_reserve_required, alpha=0.3, 
                     color='red', label='Required')
    ax3.fill_between(periods, spinning_reserve_required, spinning_reserve_available, 
                     alpha=0.3, color='green', label='Margin')
    ax3.plot(periods, spinning_reserve_available, 'b-', linewidth=2.5, 
            marker='o', markersize=4, label='Available')
    ax3.plot(periods, spinning_reserve_required, 'r--', linewidth=2, 
            marker='s', markersize=4, label='Required')
    ax3.set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Reserve (MW)', fontsize=11, fontweight='bold')
    ax3.set_title('Spinning Reserve Analysis', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Generation vs Load with Reserve
    ax4 = plt.subplot(2, 2, 4)
    total_generation = unit_generation.sum(axis=0)
    ax4.fill_between(periods, 0, load_demand, alpha=0.3, color='blue', label='Load')
    ax4.fill_between(periods, load_demand, total_generation, alpha=0.3, 
                     color='green', label='Generation Margin')
    ax4.plot(periods, load_demand, 'b-', linewidth=2.5, marker='o', markersize=4)
    ax4.plot(periods, total_generation, 'g-', linewidth=2.5, marker='s', 
            markersize=4, label='Total Generation')
    ax4.set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Power (MW)', fontsize=11, fontweight='bold')
    ax4.set_title('Generation vs Load with Margin', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Network Analysis Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(results_dir, f"advanced_5_network_dashboard_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 6: 3D Cost-Generation-Time with Network Constraints
    # ============================================================================
    print("Creating Advanced Visualization 6: 3D Cost Analysis with Network...")
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate hourly costs
    for i, unit_id in enumerate(units):
        hourly_costs = []
        hourly_generation = []
        hourly_periods = []
        
        for t in range(num_periods):
            gen = unit_generation[i, t]
            if gen > 0.1:
                cost = a_coeff[i] * gen * gen + b_coeff[i] * gen + c_coeff[i]
                hourly_costs.append(cost)
                hourly_generation.append(gen)
                hourly_periods.append(periods[t])
        
        if hourly_costs:
            # Color by reserve availability
            reserve_ratios = []
            for t_idx in hourly_periods:
                period_idx = int(t_idx) - 1  # Convert period number to index
                if 0 <= period_idx < len(spinning_reserve_available):
                    ratio = spinning_reserve_available[period_idx] / \
                           (spinning_reserve_required[period_idx] + 1e-6)
                    reserve_ratios.append(ratio)
                else:
                    reserve_ratios.append(1.0)
            
            reserve_ratios = np.array(reserve_ratios)
            if reserve_ratios.max() > 0:
                colors_scatter = plt.cm.RdYlGn(reserve_ratios / reserve_ratios.max())
            else:
                colors_scatter = plt.cm.RdYlGn([0.5] * len(hourly_periods))
            
            ax.scatter(hourly_periods, hourly_generation, hourly_costs, 
                      c=colors_scatter, s=120, alpha=0.7, 
                      label=f'Unit {unit_id}', edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('Time Period (Hour)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Power Generation (MW)', labelpad=12, fontweight='bold')
    ax.set_zlabel('Hourly Fuel Cost ($)', labelpad=12, fontweight='bold')
    ax.set_title('3D Cost-Generation-Time with Reserve Indicator', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    path = os.path.join(results_dir, f"advanced_6_3d_cost_network_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 7: Contour Plot - Generation with Network Overlay
    # ============================================================================
    print("Creating Advanced Visualization 7: Contour with Network Overlay...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left: Generation contour
    X, Y = np.meshgrid(periods, range(num_units))
    Z = unit_generation
    
    contour1 = ax1.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.85)
    contour_lines1 = ax1.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax1.clabel(contour_lines1, inline=True, fontsize=7, fmt='%d')
    
    ax1.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Unit Index', fontsize=12, fontweight='bold')
    ax1.set_title('Generation Contour Map', fontsize=14, fontweight='bold')
    ax1.set_yticks(range(num_units))
    ax1.set_yticklabels([f'Unit {u}' for u in units])
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    cbar1 = plt.colorbar(contour1, ax=ax1)
    cbar1.set_label('Generation (MW)', rotation=270, labelpad=20, fontweight='bold')
    
    # Right: Reserve contour
    reserve_matrix = np.zeros_like(Z)
    for i in range(num_units):
        for t in range(num_periods):
            if unit_status_matrix[i, t] > 0.5:
                reserve_matrix[i, t] = spinning_reserve_available[t] / max(1, unit_status_matrix[:, t].sum())
    
    contour2 = ax2.contourf(X, Y, reserve_matrix, levels=15, cmap='YlOrRd', alpha=0.85)
    contour_lines2 = ax2.contour(X, Y, reserve_matrix, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    
    ax2.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Unit Index', fontsize=12, fontweight='bold')
    ax2.set_title('Spinning Reserve Contour Map', fontsize=14, fontweight='bold')
    ax2.set_yticks(range(num_units))
    ax2.set_yticklabels([f'Unit {u}' for u in units])
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    cbar2 = plt.colorbar(contour2, ax=ax2)
    cbar2.set_label('Reserve (MW)', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(results_dir, f"advanced_7_contour_network_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 8: 3D Multi-Metric Dashboard
    # ============================================================================
    print("Creating Advanced Visualization 8: 3D Multi-Metric Dashboard...")
    fig = plt.figure(figsize=(20, 14))
    
    # Create 4 subplots showing different 3D views
    metrics = [
        (unit_generation, 'viridis', 'Generation (MW)', 'Generation Landscape'),
        (reserve_matrix, 'YlOrRd', 'Reserve (MW)', 'Reserve Landscape'),
        (line_flows_matrix[:min(10, num_branches), :], 'coolwarm', 'Flow (MW)', 'Line Flow Landscape'),
        (bus_angles_matrix[:min(10, num_buses), :] * 180 / np.pi, 'plasma', 'Angle (deg)', 'Bus Angle Landscape')
    ]
    
    for idx, (data, cmap, zlabel, title) in enumerate(metrics):
        ax = fig.add_subplot(2, 2, idx+1, projection='3d')
        
        if len(data.shape) == 2:
            rows, cols = data.shape
            X_dash, Y_dash = np.meshgrid(range(cols), range(rows))
            Z_dash = data
            
            surf = ax.plot_surface(X_dash, Y_dash, Z_dash, cmap=cmap, alpha=0.85, 
                                  linewidth=0.3, antialiased=True, edgecolor='black', linewidths=0.2)
            
            ax.set_xlabel('Time Period', labelpad=8, fontsize=9)
            ax.set_ylabel('Index', labelpad=8, fontsize=9)
            ax.set_zlabel(zlabel, labelpad=8, fontsize=9)
            ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
            ax.view_init(elev=25, azim=45)
            
            if idx == 0:  # Only add colorbar to first
                cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.05)
                cbar.set_label(zlabel, rotation=270, labelpad=15, fontsize=8)
    
    plt.suptitle('3D Multi-Metric Network Analysis Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(results_dir, f"advanced_8_3d_dashboard_{timestamp}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {path}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("All advanced visualizations generated successfully!")
    print("=" * 70)

