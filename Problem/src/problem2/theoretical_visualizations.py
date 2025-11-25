"""
Theoretical/Principle Visualizations for Problem 2
==================================================
Generates high-quality 2D principle diagrams explaining network constraints,
security mechanisms, and system operation principles.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def create_theoretical_visualizations(a_coeff, b_coeff, c_coeff, P_min, P_max, 
                                     units, branches, all_buses, bus_to_idx,
                                     spinning_reserve_available=None, spinning_reserve_required=None,
                                     line_flows_matrix=None, bus_angles_matrix=None,
                                     unit_generation=None, unit_status_matrix=None,
                                     load_demand=None, results_dir=None, timestamp=None):
    """
    Generate theoretical/principle visualizations for Problem 2.
    
    Args:
        a_coeff, b_coeff, c_coeff: Cost coefficients
        P_min, P_max: Generation limits
        units: List of unit IDs
        branches: List of branch dictionaries
        all_buses: List of all bus numbers
        bus_to_idx: Dictionary mapping bus numbers to indices
        spinning_reserve_available: Optional reserve data
        spinning_reserve_required: Optional required reserve data
        line_flows_matrix: Optional line flow data
        bus_angles_matrix: Optional bus angle data
        unit_generation: Optional actual generation data
        unit_status_matrix: Optional actual status data
        load_demand: Optional load demand data
        results_dir: Directory to save visualizations
        timestamp: Timestamp for file naming
    """
    num_units = len(units)
    num_branches = len(branches)
    num_buses = len(all_buses)
    
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
    # Visualization 1: Spinning Reserve Principle
    # ============================================================================
    print("Creating Theoretical Visualization 1: Spinning Reserve Principle...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    periods = np.arange(1, 25)
    
    if load_demand is None:
        load_demand = np.array([166, 196, 229, 257, 263, 252, 246, 213, 192, 161, 147, 160,
                               170, 185, 208, 232, 246, 241, 236, 225, 204, 182, 161, 131])
    
    # Calculate total generation capacity
    if unit_generation is not None and unit_status_matrix is not None:
        total_generation = unit_generation.sum(axis=0)
        total_capacity_online = np.array([sum(P_max[i] * unit_status_matrix[i, t] 
                                             for i in range(num_units)) 
                                         for t in range(len(periods))])
    else:
        # Conceptual values
        total_generation = load_demand * 1.1  # 10% margin
        total_capacity_online = load_demand * 1.2
    
    if spinning_reserve_available is None:
        spinning_reserve_available = total_capacity_online - total_generation
    
    if spinning_reserve_required is None:
        spinning_reserve_required = np.maximum(load_demand * 0.1, 
                                              np.full(len(periods), max(P_max)))
    
    # Top plot: Reserve visualization
    ax1.fill_between(periods, 0, load_demand, alpha=0.3, color='blue', label='Load Demand')
    ax1.fill_between(periods, load_demand, total_generation, alpha=0.3, 
                    color='green', label='Generation')
    ax1.fill_between(periods, total_generation, total_capacity_online, alpha=0.5, 
                    color='orange', label='Spinning Reserve')
    ax1.plot(periods, load_demand, 'b-', linewidth=2.5, marker='o', markersize=5)
    ax1.plot(periods, total_generation, 'g-', linewidth=2.5, marker='s', markersize=5)
    ax1.plot(periods, total_capacity_online, 'r--', linewidth=2.5, marker='^', markersize=5,
            label='Total Online Capacity')
    
    # Add reserve requirement line
    ax1.plot(periods, total_generation + spinning_reserve_required, 'm--', 
            linewidth=2, label='Required Reserve Level')
    
    ax1.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Power (MW)', fontsize=12, fontweight='bold')
    ax1.set_title('Spinning Reserve Principle: Available vs Required Reserve',
                 fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0.5, 24.5)
    
    # Bottom plot: Reserve margin
    reserve_margin = spinning_reserve_available - spinning_reserve_required
    ax2.fill_between(periods, 0, spinning_reserve_required, alpha=0.3, 
                    color='red', label='Required Reserve')
    ax2.fill_between(periods, spinning_reserve_required, spinning_reserve_available, 
                    alpha=0.5, color='green', label='Reserve Margin')
    ax2.plot(periods, spinning_reserve_available, 'b-', linewidth=2.5, 
            marker='o', markersize=5, label='Available Reserve')
    ax2.plot(periods, spinning_reserve_required, 'r--', linewidth=2.5, 
            marker='s', markersize=5, label='Required Reserve')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Add formula
    formula_text = (r'Reserve Available = $\sum_{i} (P_{max,i} \cdot u_{i,t} - P_{i,t})$' + '\n' +
                   r'Reserve Required $\geq$ max(10% Load, Largest Unit Capacity)')
    ax2.text(0.02, 0.98, formula_text, transform=ax2.transAxes, 
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.9))
    
    ax2.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Reserve (MW)', fontsize=12, fontweight='bold')
    ax2.set_title('Reserve Margin Analysis: Available - Required',
                 fontsize=14, fontweight='bold', pad=10)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0.5, 24.5)
    
    plt.tight_layout()
    
    if results_dir and timestamp:
        path = os.path.join(results_dir, f"theoretical_1_spinning_reserve_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 2: N-1 Security Constraint Principle
    # ============================================================================
    print("Creating Theoretical Visualization 2: N-1 Security Constraint...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    periods = np.arange(1, 25)
    
    if load_demand is None:
        load_demand = np.array([166, 196, 229, 257, 263, 252, 246, 213, 192, 161, 147, 160,
                               170, 185, 208, 232, 246, 241, 236, 225, 204, 182, 161, 131])
    
    # Calculate capacity scenarios
    total_capacity = sum(P_max)
    
    # Scenario 1: Normal operation
    ax = axes[0, 0]
    if unit_generation is not None and unit_status_matrix is not None:
        total_gen_normal = unit_generation.sum(axis=0)
        total_cap_normal = np.array([sum(P_max[i] * unit_status_matrix[i, t] 
                                        for i in range(num_units)) 
                                    for t in range(len(periods))])
    else:
        total_gen_normal = load_demand * 1.05
        total_cap_normal = load_demand * 1.15
    
    ax.fill_between(periods, 0, load_demand, alpha=0.3, color='blue', label='Load')
    ax.fill_between(periods, load_demand, total_gen_normal, alpha=0.3, 
                   color='green', label='Generation')
    ax.fill_between(periods, total_gen_normal, total_cap_normal, alpha=0.5, 
                   color='orange', label='Reserve')
    ax.plot(periods, load_demand, 'b-', linewidth=2, marker='o', markersize=4)
    ax.plot(periods, total_cap_normal, 'r--', linewidth=2, label='Available Capacity')
    ax.set_title('Normal Operation: All Units Available', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power (MW)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 24.5)
    
    # Scenario 2: Largest unit outage
    ax = axes[0, 1]
    largest_unit_idx = np.argmax(P_max)
    largest_unit_capacity = P_max[largest_unit_idx]
    
    if unit_generation is not None and unit_status_matrix is not None:
        total_cap_outage = total_cap_normal - np.array([P_max[largest_unit_idx] * 
                                                        unit_status_matrix[largest_unit_idx, t]
                                                        for t in range(len(periods))])
    else:
        total_cap_outage = total_cap_normal - largest_unit_capacity
    
    ax.fill_between(periods, 0, load_demand, alpha=0.3, color='blue', label='Load')
    ax.fill_between(periods, load_demand, total_gen_normal, alpha=0.3, 
                   color='green', label='Generation')
    ax.fill_between(periods, total_gen_normal, total_cap_outage, alpha=0.5, 
                   color='orange', label='Remaining Reserve')
    ax.plot(periods, load_demand, 'b-', linewidth=2, marker='o', markersize=4)
    ax.plot(periods, total_cap_outage, 'r--', linewidth=2, label='Capacity After Outage')
    ax.axhline(y=load_demand.max(), color='red', linestyle=':', linewidth=2, 
              label='Peak Load', alpha=0.7)
    ax.set_title(f'N-1 Generator Outage: Unit {units[largest_unit_idx]} Out\n'
                f'Remaining Capacity Must ≥ Load + Reserve',
                fontsize=12, fontweight='bold')
    ax.set_ylabel('Power (MW)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 24.5)
    
    # Scenario 3: Line flow limits
    ax = axes[1, 0]
    if line_flows_matrix is not None:
        # Show top 5 lines
        avg_flows = np.abs(line_flows_matrix).mean(axis=1)
        top_lines_idx = np.argsort(avg_flows)[-5:][::-1]
        
        for idx, br_idx in enumerate(top_lines_idx):
            branch = branches[br_idx]
            flows = line_flows_matrix[br_idx, :]
            flow_pct = flows / branch['P_max'] * 100
            
            ax.plot(periods, flow_pct, linewidth=2, marker='o', markersize=3,
                   label=f'Line {br_idx+1}: {branch["from"]}-{branch["to"]}')
        
        ax.axhline(y=80, color='red', linestyle='--', linewidth=2, 
                  label='Safety Limit (80%)', alpha=0.7)
        ax.axhline(y=100, color='darkred', linestyle='--', linewidth=2, 
                  label='Maximum (100%)', alpha=0.7)
    else:
        # Conceptual
        for i in range(5):
            flows = np.random.uniform(40, 75, len(periods))
            ax.plot(periods, flows, linewidth=2, marker='o', markersize=3,
                   label=f'Line {i+1}')
        ax.axhline(y=80, color='red', linestyle='--', linewidth=2, 
                  label='Safety Limit (80%)', alpha=0.7)
    
    ax.set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Flow Utilization (%)', fontsize=11, fontweight='bold')
    ax.set_title('N-1 Line Outage: Flow Redistribution\n'
                'No Single Line Should Exceed Safety Limit',
                fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 24.5)
    ax.set_ylim(0, 105)
    
    # Scenario 4: Security margin
    ax = axes[1, 1]
    if unit_generation is not None and unit_status_matrix is not None:
        security_margin = []
        for t in range(len(periods)):
            # Calculate minimum capacity after losing largest unit
            online_units = [i for i in range(num_units) if unit_status_matrix[i, t] > 0.5]
            if online_units:
                remaining_capacity = sum(P_max[i] for i in online_units) - max(P_max[i] for i in online_units)
                margin = remaining_capacity - load_demand[t]
                security_margin.append(margin)
            else:
                security_margin.append(0)
        security_margin = np.array(security_margin)
    else:
        security_margin = total_cap_outage - load_demand
    
    ax.fill_between(periods, 0, security_margin, alpha=0.5, color='green', 
                   label='Security Margin')
    ax.plot(periods, security_margin, 'g-', linewidth=2.5, marker='o', markersize=5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, 
              label='Minimum Required', alpha=0.7)
    
    formula_text = (r'Security Margin = Remaining Capacity - Load' + '\n' +
                   r'After Largest Unit Outage')
    ax.text(0.02, 0.98, formula_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.9))
    
    ax.set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Security Margin (MW)', fontsize=11, fontweight='bold')
    ax.set_title('N-1 Security Margin: Capacity After Outage',
                fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 24.5)
    
    plt.suptitle('N-1 Security Constraint Principle: System Must Operate\n'
                'Safely Under Any Single Component Outage',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if results_dir and timestamp:
        path = os.path.join(results_dir, f"theoretical_2_n1_security_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 3: DC Power Flow Principle
    # ============================================================================
    print("Creating Theoretical Visualization 3: DC Power Flow Principle...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    periods = np.arange(1, 25)
    
    # Top left: Bus voltage angles
    ax = axes[0, 0]
    if bus_angles_matrix is not None:
        # Show key buses (generator buses)
        gen_buses = [bus for bus in units if bus in bus_to_idx]
        gen_bus_indices = [bus_to_idx[bus] for bus in gen_buses]
        
        for bus_idx in gen_bus_indices[:4]:  # Show first 4 generator buses
            bus = all_buses[bus_idx]
            angles_deg = bus_angles_matrix[bus_idx, :] * 180 / np.pi
            ax.plot(periods, angles_deg, linewidth=2, marker='o', markersize=4,
                   label=f'Bus {bus}')
    else:
        # Conceptual
        for bus in [1, 2, 5, 8]:
            angles = np.random.uniform(-5, 5, len(periods))
            ax.plot(periods, angles, linewidth=2, marker='o', markersize=4,
                   label=f'Bus {bus}')
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, 
              label='Reference Bus (Bus 1)', alpha=0.7)
    ax.set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Voltage Angle (degrees)', fontsize=11, fontweight='bold')
    ax.set_title('DC Power Flow: Bus Voltage Angles\n'
                'θ₁ = 0 (Reference), P_ij = B_ij(θ_i - θ_j)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 24.5)
    
    # Top right: Line flow vs angle difference
    ax = axes[0, 1]
    if line_flows_matrix is not None and bus_angles_matrix is not None:
        # Select a key line (e.g., line 1: bus 1-2)
        br_idx = 0  # First branch
        branch = branches[br_idx]
        from_bus_idx = bus_to_idx[branch['from']]
        to_bus_idx = bus_to_idx[branch['to']]
        
        angle_diff = (bus_angles_matrix[from_bus_idx, :] - 
                     bus_angles_matrix[to_bus_idx, :]) * 180 / np.pi
        flows = line_flows_matrix[br_idx, :]
        
        ax.scatter(angle_diff, flows, s=100, alpha=0.6, c=periods, 
                  cmap='viridis', edgecolors='black', linewidths=1)
        ax.set_xlabel('Angle Difference (degrees)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Line Flow (MW)', fontsize=11, fontweight='bold')
        ax.set_title(f'Line {br_idx+1}: Flow vs Angle Difference\n'
                    f'P = B × Δθ (DC Power Flow Equation)',
                    fontsize=12, fontweight='bold')
        
        # Add theoretical line
        B = branch['B'] if 'B' in branch else 1.0 / branch['X']
        angle_range = np.linspace(angle_diff.min(), angle_diff.max(), 100)
        flow_theoretical = B * angle_range * 180 / np.pi  # Approximate
        ax.plot(angle_range, flow_theoretical, 'r--', linewidth=2, 
               label='Theoretical: P = B×Δθ', alpha=0.7)
        
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Time Period', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        # Conceptual
        angle_diff = np.linspace(-10, 10, len(periods))
        flows = angle_diff * 50  # Simplified relationship
        ax.plot(angle_diff, flows, 'b-o', linewidth=2, markersize=5)
        ax.set_xlabel('Angle Difference (degrees)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Line Flow (MW)', fontsize=11, fontweight='bold')
        ax.set_title('DC Power Flow: P = B × Δθ\n'
                    'Flow Proportional to Angle Difference',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Bottom left: Power balance at buses
    ax = axes[1, 0]
    if unit_generation is not None and load_demand is not None:
        # Show power balance for generator buses
        gen_buses = [bus for bus in units if bus in bus_to_idx]
        
        for i, bus in enumerate(gen_buses[:3]):  # Show first 3
            bus_idx = bus_to_idx[bus]
            unit_idx = units.index(bus)
            
            # Generation at this bus
            gen = unit_generation[unit_idx, :]
            
            # Simplified: assume load is distributed
            load_at_bus = load_demand / len(gen_buses)
            
            # Net power (generation - load)
            net_power = gen - load_at_bus
            
            ax.plot(periods, gen, linewidth=2, marker='o', markersize=4,
                   label=f'Bus {bus}: Generation', linestyle='-')
            ax.plot(periods, net_power, linewidth=2, marker='s', markersize=4,
                   label=f'Bus {bus}: Net Power', linestyle='--', alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Time Period (Hour)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Power (MW)', fontsize=11, fontweight='bold')
        ax.set_title('Bus Power Balance: Generation - Load = Net Flow\n'
                    'Σ P_gen - Σ P_load = Σ P_flow_out - Σ P_flow_in',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 24.5)
    else:
        ax.text(0.5, 0.5, 'Power Balance:\nGeneration - Load = Net Flow',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        ax.set_title('Bus Power Balance Principle', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Bottom right: Line flow limits
    ax = axes[1, 1]
    if line_flows_matrix is not None:
        # Show flow utilization distribution
        utilizations = []
        for br_idx in range(min(20, num_branches)):  # Top 20 lines
            branch = branches[br_idx]
            flows = np.abs(line_flows_matrix[br_idx, :])
            util = flows / branch['P_max'] * 100
            utilizations.extend(util)
        
        ax.hist(utilizations, bins=30, alpha=0.7, color='steelblue', 
               edgecolor='black', linewidth=1)
        ax.axvline(x=80, color='red', linestyle='--', linewidth=2, 
                  label='Safety Limit (80%)', alpha=0.7)
        ax.axvline(x=100, color='darkred', linestyle='--', linewidth=2, 
                  label='Maximum (100%)', alpha=0.7)
        ax.set_xlabel('Flow Utilization (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Line Flow Utilization Distribution\n'
                    'Most Lines Should Operate Below Safety Limit',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        # Conceptual histogram
        utilizations = np.random.uniform(20, 75, 200)
        ax.hist(utilizations, bins=30, alpha=0.7, color='steelblue', 
               edgecolor='black', linewidth=1)
        ax.axvline(x=80, color='red', linestyle='--', linewidth=2, 
                  label='Safety Limit (80%)', alpha=0.7)
        ax.set_xlabel('Flow Utilization (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Line Flow Utilization Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('DC Power Flow Principle: Network Constraints and Power Balance',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if results_dir and timestamp:
        path = os.path.join(results_dir, f"theoretical_3_dc_power_flow_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 4: Constraint Interaction and Feasibility Region
    # ============================================================================
    print("Creating Theoretical Visualization 4: Constraint Interaction...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Top left: Generation limits vs ramp constraints
    ax = axes[0, 0]
    # Show for one unit (Unit 1)
    unit_idx = 0
    P_range = np.linspace(P_min[unit_idx], P_max[unit_idx], 100)
    
    # Ramp constraints create a feasible band
    periods_2 = np.array([0, 1])
    P_initial = P_min[unit_idx]
    
    # Feasible region from ramp constraints
    P_upper_ramp = P_initial + Ramp_Up[unit_idx]
    P_lower_ramp = P_initial - Ramp_Down[unit_idx]
    
    ax.fill_between([0, 1], P_lower_ramp, P_upper_ramp, alpha=0.3, 
                   color='green', label='Ramp Feasible Region')
    ax.axhline(y=P_max[unit_idx], color='red', linestyle='--', linewidth=2, 
              label=f'P_max = {P_max[unit_idx]} MW')
    ax.axhline(y=P_min[unit_idx], color='blue', linestyle='--', linewidth=2, 
              label=f'P_min = {P_min[unit_idx]} MW')
    
    # Mark intersection
    feasible_max = min(P_upper_ramp, P_max[unit_idx])
    feasible_min = max(P_lower_ramp, P_min[unit_idx])
    ax.fill_between([0, 1], feasible_min, feasible_max, alpha=0.5, 
                   color='yellow', label='Final Feasible Region')
    
    ax.set_xlabel('Time Period', fontsize=11, fontweight='bold')
    ax.set_ylabel('Power Generation (MW)', fontsize=11, fontweight='bold')
    ax.set_title(f'Unit {units[unit_idx]}: Constraint Interaction\n'
                f'Generation Limits ∩ Ramp Constraints',
                fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 1.1)
    
    # Top right: Cost vs constraints
    ax = axes[0, 1]
    P_range_cost = np.linspace(P_min[unit_idx], P_max[unit_idx], 100)
    cost = (a_coeff[unit_idx] * P_range_cost**2 + 
           b_coeff[unit_idx] * P_range_cost + c_coeff[unit_idx])
    
    ax.plot(P_range_cost, cost, 'b-', linewidth=2.5, label='Cost Function')
    
    # Mark feasible region
    ax.axvspan(P_min[unit_idx], P_max[unit_idx], alpha=0.2, color='green', 
             label='Feasible Region')
    
    # Mark optimal point (minimum cost in feasible region)
    optimal_P = max(P_min[unit_idx], -b_coeff[unit_idx] / (2 * a_coeff[unit_idx]))
    optimal_P = min(optimal_P, P_max[unit_idx])
    optimal_cost = (a_coeff[unit_idx] * optimal_P**2 + 
                   b_coeff[unit_idx] * optimal_P + c_coeff[unit_idx])
    
    ax.plot(optimal_P, optimal_cost, 'ro', markersize=12, label='Optimal Point')
    ax.axvline(x=P_min[unit_idx], color='blue', linestyle='--', linewidth=2, 
              label=f'P_min = {P_min[unit_idx]} MW')
    ax.axvline(x=P_max[unit_idx], color='red', linestyle='--', linewidth=2, 
              label=f'P_max = {P_max[unit_idx]} MW')
    
    ax.set_xlabel('Power Generation (MW)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cost ($)', fontsize=11, fontweight='bold')
    ax.set_title(f'Unit {units[unit_idx]}: Cost Optimization\n'
                f'Minimize Cost Subject to Constraints',
                fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Bottom left: System-wide constraint summary
    ax = axes[1, 0]
    constraint_types = ['Power\nBalance', 'Generation\nLimits', 'Ramp\nRates', 
                       'Min Up/Down\nTime', 'Startup/\nShutdown', 'Network\nFlow']
    constraint_counts = [24, 144, 144, 144, 144, 984]  # Approximate counts
    
    colors_bar = plt.cm.viridis(np.linspace(0, 1, len(constraint_types)))
    bars = ax.bar(constraint_types, constraint_counts, color=colors_bar, 
                 alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Number of Constraints', fontsize=11, fontweight='bold')
    ax.set_title('Problem 2: Constraint Complexity\n'
                'Total Constraints: ~1,680',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Bottom right: Feasibility region visualization
    ax = axes[1, 1]
    # Create 2D feasibility region (simplified for two units)
    if num_units >= 2:
        P1_range = np.linspace(P_min[0], P_max[0], 50)
        P2_range = np.linspace(P_min[1], P_max[1], 50)
        P1_grid, P2_grid = np.meshgrid(P1_range, P2_range)
        
        # Power balance constraint: P1 + P2 = Load (for one period)
        load_sample = load_demand[10] if load_demand is not None else 200
        balance_line = load_sample - P1_range
        
        # Plot feasible region
        ax.fill_between(P1_range, P_min[1], np.minimum(P_max[1], balance_line), 
                       where=(balance_line >= P_min[1]), alpha=0.3, color='green',
                       label='Feasible Region')
        ax.plot(P1_range, balance_line, 'b-', linewidth=2.5, 
               label=f'Power Balance: P1 + P2 = {load_sample} MW')
        
        # Generation limits
        ax.axhline(y=P_max[1], color='red', linestyle='--', linewidth=2, 
                 label=f'P2_max = {P_max[1]} MW')
        ax.axhline(y=P_min[1], color='blue', linestyle='--', linewidth=2, 
                 label=f'P2_min = {P_min[1]} MW')
        ax.axvline(x=P_max[0], color='red', linestyle='--', linewidth=2, 
                 label=f'P1_max = {P_max[0]} MW')
        ax.axvline(x=P_min[0], color='blue', linestyle='--', linewidth=2, 
                 label=f'P1_min = {P_min[0]} MW')
        
        ax.set_xlabel(f'Unit {units[0]} Generation (MW)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Unit {units[1]} Generation (MW)', fontsize=11, fontweight='bold')
        ax.set_title('Feasibility Region: Two Units\n'
                    'Intersection of All Constraints',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(P_min[0] - 10, P_max[0] + 10)
        ax.set_ylim(P_min[1] - 10, P_max[1] + 10)
    else:
        ax.text(0.5, 0.5, 'Feasibility Region:\nIntersection of All Constraints',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        ax.set_title('Feasibility Region Concept', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Constraint Interaction and Feasibility Analysis',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if results_dir and timestamp:
        path = os.path.join(results_dir, f"theoretical_4_constraint_interaction_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("All theoretical visualizations generated successfully!")
    print("=" * 70)

