"""
Theoretical/Principle Visualizations for Problem 1
==================================================
Generates high-quality 2D principle diagrams explaining UC optimization concepts,
constraint mechanisms, and system operation principles.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def create_theoretical_visualizations(a_coeff, b_coeff, c_coeff, P_min, P_max, 
                                     Ramp_Up, Ramp_Down, Min_Up_Time, Min_Down_Time,
                                     Startup_Cost, Shutdown_Cost, units, 
                                     unit_generation=None, unit_status_matrix=None,
                                     load_demand=None, results_dir=None, timestamp=None):
    """
    Generate theoretical/principle visualizations for Problem 1.
    
    Args:
        a_coeff, b_coeff, c_coeff: Cost coefficients
        P_min, P_max: Generation limits
        Ramp_Up, Ramp_Down: Ramp rate limits
        Min_Up_Time, Min_Down_Time: Minimum up/down times
        Startup_Cost, Shutdown_Cost: Startup/shutdown costs
        units: List of unit IDs
        unit_generation: Optional actual generation data
        unit_status_matrix: Optional actual status data
        load_demand: Optional load demand data
        results_dir: Directory to save visualizations
        timestamp: Timestamp for file naming
    """
    num_units = len(units)
    
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
    # Visualization 1: Cost Function Characteristics
    # ============================================================================
    print("Creating Theoretical Visualization 1: Cost Function Characteristics...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, unit_id in enumerate(units):
        ax = axes[i]
        P_range = np.linspace(0, P_max[i], 200)
        cost = a_coeff[i] * P_range**2 + b_coeff[i] * P_range + c_coeff[i]
        
        # Plot cost function
        ax.plot(P_range, cost, 'b-', linewidth=2.5, label='Fuel Cost')
        
        # Mark operating range
        P_op_range = np.linspace(P_min[i], P_max[i], 100)
        cost_op = a_coeff[i] * P_op_range**2 + b_coeff[i] * P_op_range + c_coeff[i]
        ax.fill_between(P_op_range, cost_op, alpha=0.3, color='green', 
                        label=f'Operating Range [{P_min[i]}, {P_max[i]}] MW')
        
        # Mark minimum and maximum points
        cost_min = a_coeff[i] * P_min[i]**2 + b_coeff[i] * P_min[i] + c_coeff[i]
        cost_max = a_coeff[i] * P_max[i]**2 + b_coeff[i] * P_max[i] + c_coeff[i]
        ax.plot(P_min[i], cost_min, 'go', markersize=10, label='P_min')
        ax.plot(P_max[i], cost_max, 'ro', markersize=10, label='P_max')
        
        # Add marginal cost line
        marginal_cost = 2 * a_coeff[i] * P_range + b_coeff[i]
        ax2 = ax.twinx()
        ax2.plot(P_range, marginal_cost, 'r--', linewidth=1.5, alpha=0.7, label='Marginal Cost')
        ax2.set_ylabel('Marginal Cost ($/MWh)', fontsize=10, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_xlabel('Power Generation (MW)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Fuel Cost ($)', fontsize=11, fontweight='bold')
        ax.set_title(f'Unit {unit_id}: Cost Function\n'
                    f'C(P) = {a_coeff[i]:.4f}P² + {b_coeff[i]:.2f}P + {c_coeff[i]:.0f}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(0, P_max[i] * 1.1)
    
    plt.suptitle('Cost Function Characteristics: Quadratic Fuel Cost Curves',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if results_dir and timestamp:
        path = os.path.join(results_dir, f"theoretical_1_cost_functions_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 2: Ramp Rate Constraint Mechanism
    # ============================================================================
    print("Creating Theoretical Visualization 2: Ramp Rate Constraint Mechanism...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, unit_id in enumerate(units):
        ax = axes[i]
        
        # Simulate a scenario: unit starts at P_min, ramps up, then down
        periods = np.array([0, 1, 2, 3, 4, 5])
        P_initial = P_min[i]
        
        # Calculate feasible trajectories
        # Ramp up trajectory
        P_ramp_up = [P_initial]
        for t in range(1, len(periods)):
            P_next = min(P_ramp_up[-1] + Ramp_Up[i], P_max[i])
            P_ramp_up.append(P_next)
        
        # Ramp down trajectory
        P_ramp_down = [P_max[i]]
        for t in range(1, len(periods)):
            P_next = max(P_ramp_down[-1] - Ramp_Down[i], P_min[i])
            P_ramp_down.append(P_next)
        
        # Plot feasible region
        for t in range(len(periods)-1):
            # Upper bound: P[t] + Ramp_Up
            P_upper = P_ramp_up[t] + Ramp_Up[i]
            P_upper = min(P_upper, P_max[i])
            # Lower bound: P[t] - Ramp_Down
            P_lower = max(P_ramp_up[t] - Ramp_Down[i], P_min[i])
            
            # Draw feasible region
            ax.fill_between([periods[t], periods[t+1]], P_lower, P_upper, 
                           alpha=0.2, color='green', label='Feasible Region' if t == 0 else '')
        
        # Plot ramp trajectories
        ax.plot(periods, P_ramp_up, 'b-o', linewidth=2.5, markersize=8, 
               label='Ramp Up Trajectory')
        ax.plot(periods, P_ramp_down, 'r-s', linewidth=2.5, markersize=8, 
               label='Ramp Down Trajectory')
        
        # Add ramp rate annotations
        for t in range(len(periods)-1):
            # Up arrow
            ax.annotate('', xy=(periods[t+1], P_ramp_up[t+1]), 
                       xytext=(periods[t], P_ramp_up[t]),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2))
            # Down arrow
            ax.annotate('', xy=(periods[t+1], P_ramp_down[t+1]), 
                       xytext=(periods[t], P_ramp_down[t]),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # Mark limits
        ax.axhline(y=P_max[i], color='red', linestyle='--', linewidth=2, 
                  alpha=0.7, label='P_max')
        ax.axhline(y=P_min[i], color='blue', linestyle='--', linewidth=2, 
                  alpha=0.7, label='P_min')
        
        # Add text annotations
        ax.text(0.5, P_max[i] - 10, f'Ramp Up: {Ramp_Up[i]} MW/h', 
               fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(0.5, P_min[i] + 10, f'Ramp Down: {Ramp_Down[i]} MW/h', 
               fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        ax.set_xlabel('Time Period', fontsize=11, fontweight='bold')
        ax.set_ylabel('Power Generation (MW)', fontsize=11, fontweight='bold')
        ax.set_title(f'Unit {unit_id}: Ramp Rate Constraint Mechanism',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(-0.2, periods[-1] + 0.2)
        ax.set_ylim(0, P_max[i] * 1.15)
    
    plt.suptitle('Ramp Rate Constraint Mechanism: Feasible Generation Trajectories',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if results_dir and timestamp:
        path = os.path.join(results_dir, f"theoretical_2_ramp_constraints_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 3: Minimum Up/Down Time Constraint Mechanism
    # ============================================================================
    print("Creating Theoretical Visualization 3: Min Up/Down Time Mechanism...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, unit_id in enumerate(units):
        ax = axes[i]
        
        # Create timeline showing min up/down time constraints
        periods = np.arange(0, 15)
        
        # Scenario: Unit starts OFF, then starts up at period 3
        status = np.zeros(len(periods))
        status[3:] = 1  # Starts at period 3
        
        # Apply min up time constraint: must stay ON for Min_Up_Time periods
        # If unit starts at period 3, must stay ON until period 3 + Min_Up_Time
        min_up_end = 3 + Min_Up_Time[i]
        status[3:min_up_end+1] = 1
        
        # Then unit shuts down at period min_up_end + 1
        shutdown_period = min_up_end + 1
        if shutdown_period < len(periods):
            status[shutdown_period:] = 0
            # Apply min down time: must stay OFF for Min_Down_Time periods
            min_down_end = shutdown_period + Min_Down_Time[i]
            if min_down_end < len(periods):
                status[shutdown_period:min_down_end+1] = 0
        
        # Plot status
        ax.fill_between(periods, 0, status, alpha=0.5, color='green', 
                       label='Unit ON', step='pre')
        ax.fill_between(periods, status, 1, alpha=0.3, color='red', 
                       label='Unit OFF', step='pre')
        
        # Mark startup and shutdown events
        startup_period = 3
        ax.axvline(x=startup_period, color='blue', linestyle='--', linewidth=2, 
                  label='Startup Event')
        ax.plot(startup_period, 0.5, 'b^', markersize=15, label='Startup')
        
        if shutdown_period < len(periods):
            ax.axvline(x=shutdown_period, color='orange', linestyle='--', linewidth=2, 
                      label='Shutdown Event')
            ax.plot(shutdown_period, 0.5, 'rv', markersize=15, label='Shutdown')
        
        # Mark min up time period
        ax.axvspan(startup_period, min_up_end, alpha=0.2, color='blue', 
                  label=f'Min Up Time = {Min_Up_Time[i]} periods')
        
        # Mark min down time period
        if shutdown_period < len(periods):
            ax.axvspan(shutdown_period, min(min_down_end, len(periods)-1), 
                      alpha=0.2, color='red', 
                      label=f'Min Down Time = {Min_Down_Time[i]} periods')
        
        # Add annotations
        ax.text(startup_period + Min_Up_Time[i]/2, 0.8, 
               f'Must stay ON\nfor {Min_Up_Time[i]} periods', 
               fontsize=9, ha='center', bbox=dict(boxstyle='round', 
               facecolor='lightblue', alpha=0.8))
        
        if shutdown_period < len(periods):
            ax.text(shutdown_period + Min_Down_Time[i]/2, 0.2, 
                   f'Must stay OFF\nfor {Min_Down_Time[i]} periods', 
                   fontsize=9, ha='center', bbox=dict(boxstyle='round', 
                   facecolor='lightcoral', alpha=0.8))
        
        ax.set_xlabel('Time Period', fontsize=11, fontweight='bold')
        ax.set_ylabel('Unit Status (1=ON, 0=OFF)', fontsize=11, fontweight='bold')
        ax.set_title(f'Unit {unit_id}: Minimum Up/Down Time Constraints\n'
                    f'Min Up: {Min_Up_Time[i]}h, Min Down: {Min_Down_Time[i]}h',
                    fontsize=12, fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['OFF', 'ON'])
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax.legend(loc='best', fontsize=8, ncol=2)
    
    plt.suptitle('Minimum Up/Down Time Constraint Mechanism',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if results_dir and timestamp:
        path = os.path.join(results_dir, f"theoretical_3_min_updown_time_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 4: Power Balance and Economic Dispatch Principle
    # ============================================================================
    print("Creating Theoretical Visualization 4: Power Balance Principle...")
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create a conceptual diagram showing power balance
    periods = np.arange(1, 25)
    
    if load_demand is not None:
        load = load_demand
    else:
        # Use default load
        load = np.array([166, 196, 229, 257, 263, 252, 246, 213, 192, 161, 147, 160,
                        170, 185, 208, 232, 246, 241, 236, 225, 204, 182, 161, 131])
    
    # Calculate total capacity
    total_capacity = sum(P_max)
    
    # Plot load demand
    ax.fill_between(periods, 0, load, alpha=0.3, color='blue', label='Load Demand')
    ax.plot(periods, load, 'b-', linewidth=3, marker='o', markersize=6, label='Load Profile')
    
    # Plot total capacity
    ax.axhline(y=total_capacity, color='red', linestyle='--', linewidth=2.5, 
              label=f'Total Capacity ({total_capacity} MW)')
    
    # Plot individual unit capacities (stacked)
    bottom = np.zeros(len(periods))
    colors = plt.cm.tab10(np.linspace(0, 1, num_units))
    
    for i, unit_id in enumerate(units):
        # Show unit capacity as available (not actual generation)
        unit_capacity = np.full(len(periods), P_max[i])
        ax.bar(periods, unit_capacity, bottom=bottom, alpha=0.4, 
              color=colors[i], label=f'Unit {unit_id} Capacity ({P_max[i]} MW)',
              edgecolor='black', linewidth=0.5)
        bottom += unit_capacity
    
    # Add power balance constraint visualization
    for t in range(len(periods)):
        # Draw arrow showing balance requirement
        if t % 4 == 0:  # Every 4th period
            ax.annotate('', xy=(periods[t], load[t]), 
                       xytext=(periods[t], load[t] + 20),
                       arrowprops=dict(arrowstyle='<->', color='green', lw=2))
            ax.text(periods[t], load[t] + 30, 'Balance\nRequired', 
                   fontsize=8, ha='center', bbox=dict(boxstyle='round', 
                   facecolor='lightgreen', alpha=0.8))
    
    ax.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power (MW)', fontsize=12, fontweight='bold')
    ax.set_title('Power Balance Constraint: Total Generation = Load Demand\n'
                'Economic Dispatch Principle',
                fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, 24.5)
    
    plt.tight_layout()
    
    if results_dir and timestamp:
        path = os.path.join(results_dir, f"theoretical_4_power_balance_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 5: Startup/Shutdown Cost Impact
    # ============================================================================
    print("Creating Theoretical Visualization 5: Startup/Shutdown Cost Impact...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, unit_id in enumerate(units):
        ax = axes[i]
        
        # Create scenario: unit cycles ON-OFF-ON
        periods = np.arange(0, 12)
        status = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        
        # Calculate cumulative costs
        cumulative_cost = np.zeros(len(periods))
        events = []
        
        for t in range(len(periods)):
            if t == 0:
                cumulative_cost[t] = 0
            else:
                cumulative_cost[t] = cumulative_cost[t-1]
                
                # Startup cost
                if status[t-1] == 0 and status[t] == 1:
                    cumulative_cost[t] += Startup_Cost[i]
                    events.append((t, 'Startup', Startup_Cost[i]))
                
                # Shutdown cost
                if status[t-1] == 1 and status[t] == 0:
                    cumulative_cost[t] += Shutdown_Cost[i]
                    events.append((t, 'Shutdown', Shutdown_Cost[i]))
        
        # Plot status
        ax2 = ax.twinx()
        ax2.fill_between(periods, 0, status, alpha=0.3, color='green', 
                        label='Unit Status', step='pre')
        ax2.set_ylabel('Unit Status', fontsize=10, color='green')
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['OFF', 'ON'])
        
        # Plot cumulative cost
        ax.plot(periods, cumulative_cost, 'r-o', linewidth=2.5, markersize=8, 
               label='Cumulative Cost')
        
        # Mark cost jumps
        for event_t, event_type, cost in events:
            ax.plot(event_t, cumulative_cost[event_t], 'ro', markersize=12)
            ax.annotate(f'{event_type}\n${cost}', 
                       xy=(event_t, cumulative_cost[event_t]),
                       xytext=(event_t, cumulative_cost[event_t] + max(cumulative_cost)*0.15),
                       fontsize=9, ha='center',
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        ax.set_xlabel('Time Period', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cumulative Cost ($)', fontsize=11, fontweight='bold', color='red')
        ax.tick_params(axis='y', labelcolor='red')
        ax.set_title(f'Unit {unit_id}: Startup/Shutdown Cost Impact\n'
                    f'Startup: ${Startup_Cost[i]}, Shutdown: ${Shutdown_Cost[i]}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('Startup/Shutdown Cost Impact on Unit Commitment Decisions',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if results_dir and timestamp:
        path = os.path.join(results_dir, f"theoretical_5_startup_shutdown_cost_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
    plt.close()
    
    # ============================================================================
    # Visualization 6: Optimization Objective Function Structure
    # ============================================================================
    print("Creating Theoretical Visualization 6: Objective Function Structure...")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create conceptual diagram of objective function components
    periods = np.arange(1, 25)
    
    # Simulate cost components (conceptual)
    if unit_generation is not None and unit_status_matrix is not None:
        # Calculate actual costs
        fuel_costs = []
        startup_costs = []
        shutdown_costs = []
        
        for t in range(len(periods)):
            fuel_t = sum(a_coeff[i] * unit_generation[i, t]**2 + 
                        b_coeff[i] * unit_generation[i, t] + c_coeff[i]
                        for i in range(num_units))
            fuel_costs.append(fuel_t)
            
            startup_t = sum(Startup_Cost[i] * (1 if t > 0 and unit_status_matrix[i, t-1] < 0.5 
                                             and unit_status_matrix[i, t] > 0.5 else 0)
                           for i in range(num_units))
            startup_costs.append(startup_t)
            
            shutdown_t = sum(Shutdown_Cost[i] * (1 if t > 0 and unit_status_matrix[i, t-1] > 0.5 
                                                and unit_status_matrix[i, t] < 0.5 else 0)
                            for i in range(num_units))
            shutdown_costs.append(shutdown_t)
    else:
        # Use conceptual values
        fuel_costs = np.random.uniform(800, 1200, len(periods))
        startup_costs = np.zeros(len(periods))
        startup_costs[[3, 8, 15]] = [180, 40, 60]  # Some startup events
        shutdown_costs = np.zeros(len(periods))
        shutdown_costs[[6, 12]] = [180, 40]  # Some shutdown events
    
    # Plot stacked cost components
    ax.fill_between(periods, 0, fuel_costs, alpha=0.6, color='blue', 
                   label='Fuel Cost (Continuous)')
    ax.fill_between(periods, fuel_costs, 
                   np.array(fuel_costs) + np.array(startup_costs), 
                   alpha=0.6, color='green', label='Startup Cost (Discrete)')
    ax.fill_between(periods, 
                   np.array(fuel_costs) + np.array(startup_costs),
                   np.array(fuel_costs) + np.array(startup_costs) + np.array(shutdown_costs),
                   alpha=0.6, color='red', label='Shutdown Cost (Discrete)')
    
    # Add total cost line
    total_costs = np.array(fuel_costs) + np.array(startup_costs) + np.array(shutdown_costs)
    ax.plot(periods, total_costs, 'k-', linewidth=3, marker='o', markersize=6, 
           label='Total Cost', zorder=10)
    
    # Add formula annotation
    formula_text = (r'$\min \sum_{t=1}^{24} \sum_{i=1}^{6} \left[ a_i P_{i,t}^2 + b_i P_{i,t} + c_i \right]$' + '\n' +
                   r'$+ \sum_{t=1}^{24} \sum_{i=1}^{6} \left[ C_{startup,i} \cdot v_{i,t} + C_{shutdown,i} \cdot w_{i,t} \right]$')
    ax.text(0.02, 0.98, formula_text, transform=ax.transAxes, 
           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.9), family='monospace')
    
    ax.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cost ($)', fontsize=12, fontweight='bold')
    ax.set_title('Objective Function Structure: Cost Components Over Time\n'
                'Minimize Total Operating Cost',
                fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, 24.5)
    
    plt.tight_layout()
    
    if results_dir and timestamp:
        path = os.path.join(results_dir, f"theoretical_6_objective_function_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {path}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("All theoretical visualizations generated successfully!")
    print("=" * 70)

