"""
Problem 2: Comparison Script for Problem 1 vs Problem 2
========================================================

This script compares the results from Problem 1 (classical UC) and 
Problem 2 (UC with network and security constraints).

Usage:
    python compare_results.py [--problem1-dir DIR] [--problem2-dir DIR]
    
    Options:
        --problem1-dir: Directory containing Problem 1 results (default: latest)
        --problem2-dir: Directory containing Problem 2 results (default: latest)
"""

import argparse
import os
import json
import glob
from datetime import datetime

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    VISUALIZATION_AVAILABLE = True
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Visualization libraries not available")

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(
    description='Compare Problem 1 and Problem 2 Results'
)
parser.add_argument(
    '--problem1-dir',
    type=str,
    default=None,
    help='Directory containing Problem 1 results (default: latest)'
)
parser.add_argument(
    '--problem2-dir',
    type=str,
    default=None,
    help='Directory containing Problem 2 results (default: latest)'
)
args = parser.parse_args()

# ============================================================================
# Helper Functions
# ============================================================================

def find_latest_results_dir(base_dir):
    """Find the latest results directory based on timestamp in filenames."""
    if not os.path.exists(base_dir):
        return None
    
    # Look for summary JSON files
    summary_files = glob.glob(os.path.join(base_dir, "summary_*.json"))
    if not summary_files:
        return None
    
    # Extract timestamps and find latest
    timestamps = []
    for f in summary_files:
        try:
            # Extract timestamp from filename: summary_YYYYMMDD_HHMMSS.json
            basename = os.path.basename(f)
            timestamp_str = basename.replace("summary_", "").replace(".json", "")
            timestamps.append((datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S"), f))
        except:
            continue
    
    if not timestamps:
        return None
    
    timestamps.sort(reverse=True)
    latest_file = timestamps[0][1]
    return os.path.dirname(latest_file)

def load_summary_json(results_dir):
    """Load summary JSON file from results directory."""
    summary_files = glob.glob(os.path.join(results_dir, "summary_*.json"))
    if not summary_files:
        return None
    
    # Get the latest one
    summary_files.sort(reverse=True)
    latest_file = summary_files[0]
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_schedule_csv(results_dir):
    """Load UC schedule CSV file from results directory."""
    csv_files = glob.glob(os.path.join(results_dir, "uc_schedule_*.csv"))
    if not csv_files:
        return None
    
    csv_files.sort(reverse=True)
    latest_file = csv_files[0]
    
    if VISUALIZATION_AVAILABLE:
        return pd.read_csv(latest_file)
    else:
        return None

# ============================================================================
# Main Comparison
# ============================================================================

print("=" * 70)
print("Problem 1 vs Problem 2 Results Comparison")
print("=" * 70)

# Get project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
problem1_base = os.path.join(project_root, "results", "problem1")
problem2_base = os.path.join(project_root, "results", "problem2")

# Find results directories
if args.problem1_dir:
    problem1_dir = args.problem1_dir
else:
    problem1_dir = find_latest_results_dir(problem1_base)

if args.problem2_dir:
    problem2_dir = args.problem2_dir
else:
    problem2_dir = find_latest_results_dir(problem2_base)

if not problem1_dir or not os.path.exists(problem1_dir):
    print(f"ERROR: Problem 1 results directory not found: {problem1_dir}")
    exit(1)

if not problem2_dir or not os.path.exists(problem2_dir):
    print(f"ERROR: Problem 2 results directory not found: {problem2_dir}")
    exit(1)

print(f"\nProblem 1 results: {problem1_dir}")
print(f"Problem 2 results: {problem2_dir}")

# Load summary data
p1_summary = load_summary_json(problem1_dir)
p2_summary = load_summary_json(problem2_dir)

if not p1_summary or not p2_summary:
    print("ERROR: Could not load summary files")
    exit(1)

# Load schedule data
p1_schedule = load_schedule_csv(problem1_dir)
p2_schedule = load_schedule_csv(problem2_dir)

# ============================================================================
# Cost Comparison
# ============================================================================

print("\n" + "=" * 70)
print("COST COMPARISON")
print("=" * 70)

p1_cost = p1_summary['optimization_info']['total_cost']
p2_cost = p2_summary['optimization_info']['total_cost']
cost_increase = p2_cost - p1_cost
cost_increase_pct = (cost_increase / p1_cost) * 100 if p1_cost > 0 else 0

p1_fuel = p1_summary['optimization_info']['fuel_cost']
p2_fuel = p2_summary['optimization_info']['fuel_cost']
p1_startup = p1_summary['optimization_info']['startup_cost']
p2_startup = p2_summary['optimization_info']['startup_cost']
p1_shutdown = p1_summary['optimization_info']['shutdown_cost']
p2_shutdown = p2_summary['optimization_info']['shutdown_cost']

print(f"\nTotal Cost:")
print(f"  Problem 1: ${p1_cost:.2f}")
print(f"  Problem 2: ${p2_cost:.2f}")
print(f"  Increase:  ${cost_increase:.2f} ({cost_increase_pct:+.2f}%)")

print(f"\nCost Breakdown:")
print(f"  {'Component':<20} {'Problem 1':>15} {'Problem 2':>15} {'Difference':>15}")
print(f"  {'-'*65}")
print(f"  {'Fuel Cost':<20} ${p1_fuel:>14.2f} ${p2_fuel:>14.2f} ${p2_fuel-p1_fuel:>14.2f}")
print(f"  {'Startup Cost':<20} ${p1_startup:>14.2f} ${p2_startup:>14.2f} ${p2_startup-p1_startup:>14.2f}")
print(f"  {'Shutdown Cost':<20} ${p1_shutdown:>14.2f} ${p2_shutdown:>14.2f} ${p2_shutdown-p1_shutdown:>14.2f}")
print(f"  {'Total Cost':<20} ${p1_cost:>14.2f} ${p2_cost:>14.2f} ${cost_increase:>14.2f}")

# ============================================================================
# Unit Commitment Comparison
# ============================================================================

print("\n" + "=" * 70)
print("UNIT COMMITMENT COMPARISON")
print("=" * 70)

if p1_schedule is not None and p2_schedule is not None:
    # Compare unit utilization
    p1_units = p1_summary['optimization_info']['units']
    p2_units = p2_summary['optimization_info']['units']
    
    print(f"\nUnit Utilization Rates:")
    print(f"  {'Unit':<10} {'Problem 1':>15} {'Problem 2':>15} {'Difference':>15}")
    print(f"  {'-'*55}")
    
    p1_unit_stats = {stat['unit_id']: stat for stat in p1_summary['unit_statistics']}
    p2_unit_stats = {stat['unit_id']: stat for stat in p2_summary['unit_statistics']}
    
    for unit_id in p1_units:
        if unit_id in p1_unit_stats and unit_id in p2_unit_stats:
            p1_util = p1_unit_stats[unit_id]['utilization_rate'] * 100
            p2_util = p2_unit_stats[unit_id]['utilization_rate'] * 100
            diff = p2_util - p1_util
            print(f"  Unit {unit_id:<5} {p1_util:>14.1f}% {p2_util:>14.1f}% {diff:>+14.1f}%")
    
    # Compare total generation
    print(f"\nTotal Generation:")
    p1_total_gen = sum(stat['total_generation_MWh'] for stat in p1_summary['unit_statistics'])
    p2_total_gen = sum(stat['total_generation_MWh'] for stat in p2_summary['unit_statistics'])
    print(f"  Problem 1: {p1_total_gen:.2f} MWh")
    print(f"  Problem 2: {p2_total_gen:.2f} MWh")
    print(f"  Difference: {p2_total_gen - p1_total_gen:+.2f} MWh")

# ============================================================================
# Additional Metrics
# ============================================================================

print("\n" + "=" * 70)
print("ADDITIONAL METRICS")
print("=" * 70)

print(f"\nProblem 2 Features:")
print(f"  Spinning Reserve: {'Enabled' if p2_summary['optimization_info'].get('spinning_reserve_enabled', False) else 'Disabled'}")
print(f"  N-1 Security: {'Enabled' if p2_summary['optimization_info'].get('n1_security_enabled', False) else 'Disabled'}")
print(f"  Inertia Constraint: {'Enabled' if p2_summary['optimization_info'].get('inertia_constraint_enabled', False) else 'Disabled'}")

if p2_schedule is not None and 'Spinning_Reserve_MW' in p2_schedule.columns:
    avg_reserve = p2_schedule['Spinning_Reserve_MW'].mean()
    min_reserve = p2_schedule['Spinning_Reserve_MW'].min()
    max_reserve = p2_schedule['Spinning_Reserve_MW'].max()
    print(f"\nSpinning Reserve Statistics:")
    print(f"  Average: {avg_reserve:.2f} MW")
    print(f"  Minimum: {min_reserve:.2f} MW")
    print(f"  Maximum: {max_reserve:.2f} MW")

# ============================================================================
# Visualization
# ============================================================================

if VISUALIZATION_AVAILABLE and p1_schedule is not None and p2_schedule is not None:
    print("\n" + "=" * 70)
    print("Generating Comparison Visualizations...")
    print("=" * 70)
    
    # Create output directory
    comparison_dir = os.path.join(project_root, "results", "problem2", "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    periods = np.arange(1, len(p1_schedule) + 1)
    
    # Visualization 1: Cost Comparison Bar Chart
    print("Creating Visualization 1: Cost Comparison...")
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 6))
    
    categories = ['Fuel Cost', 'Startup Cost', 'Shutdown Cost', 'Total Cost']
    p1_values = [p1_fuel, p1_startup, p1_shutdown, p1_cost]
    p2_values = [p2_fuel, p2_startup, p2_shutdown, p2_cost]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1a.bar(x - width/2, p1_values, width, label='Problem 1', alpha=0.8)
    bars2 = ax1a.bar(x + width/2, p2_values, width, label='Problem 2', alpha=0.8)
    
    ax1a.set_xlabel('Cost Component', fontsize=12, fontweight='bold')
    ax1a.set_ylabel('Cost ($)', fontsize=12, fontweight='bold')
    ax1a.set_title('Cost Comparison: Problem 1 vs Problem 2', fontsize=13, fontweight='bold')
    ax1a.set_xticks(x)
    ax1a.set_xticklabels(categories, rotation=45, ha='right')
    ax1a.legend()
    ax1a.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1a.text(bar.get_x() + bar.get_width()/2., height,
                     f'${height:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Cost increase percentage
    cost_diff = [p2_values[i] - p1_values[i] for i in range(len(categories))]
    colors = ['green' if diff <= 0 else 'red' for diff in cost_diff]
    bars_diff = ax1b.bar(categories, cost_diff, color=colors, alpha=0.7, edgecolor='black')
    ax1b.set_xlabel('Cost Component', fontsize=12, fontweight='bold')
    ax1b.set_ylabel('Cost Increase ($)', fontsize=12, fontweight='bold')
    ax1b.set_title('Cost Increase: Problem 2 - Problem 1', fontsize=13, fontweight='bold')
    ax1b.set_xticklabels(categories, rotation=45, ha='right')
    ax1b.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1b.grid(True, alpha=0.3, axis='y')
    
    for bar in bars_diff:
        height = bar.get_height()
        ax1b.text(bar.get_x() + bar.get_width()/2., height,
                 f'${height:+.0f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    fig1_path = os.path.join(comparison_dir, f"1_cost_comparison_{timestamp}.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {fig1_path}")
    plt.close()
    
    # Visualization 2: Generation Schedule Comparison
    print("Creating Visualization 2: Generation Schedule Comparison...")
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    
    p1_total_gen = p1_schedule[[col for col in p1_schedule.columns if 'Generation_MW' in col and 'Total' in col]].iloc[:, 0]
    p2_total_gen = p2_schedule[[col for col in p2_schedule.columns if 'Generation_MW' in col and 'Total' in col]].iloc[:, 0]
    load = p1_schedule['Load_MW']
    
    ax2.plot(periods, p1_total_gen, 'b-', linewidth=2, marker='o', markersize=4, label='Problem 1 Generation', alpha=0.7)
    ax2.plot(periods, p2_total_gen, 'r-', linewidth=2, marker='s', markersize=4, label='Problem 2 Generation', alpha=0.7)
    ax2.plot(periods, load, 'k--', linewidth=2.5, label='Load Demand', alpha=0.7)
    
    ax2.set_xlabel('Time Period (Hour)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Power Generation (MW)', fontsize=12, fontweight='bold')
    ax2.set_title('Generation Schedule Comparison: Problem 1 vs Problem 2', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, len(periods) + 0.5)
    
    plt.tight_layout()
    fig2_path = os.path.join(comparison_dir, f"2_generation_comparison_{timestamp}.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {fig2_path}")
    plt.close()
    
    # Visualization 3: Unit Utilization Comparison
    print("Creating Visualization 3: Unit Utilization Comparison...")
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    
    unit_ids = []
    p1_utils = []
    p2_utils = []
    
    for unit_id in p1_units:
        if unit_id in p1_unit_stats and unit_id in p2_unit_stats:
            unit_ids.append(f'Unit {unit_id}')
            p1_utils.append(p1_unit_stats[unit_id]['utilization_rate'] * 100)
            p2_utils.append(p2_unit_stats[unit_id]['utilization_rate'] * 100)
    
    x = np.arange(len(unit_ids))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, p1_utils, width, label='Problem 1', alpha=0.8)
    bars2 = ax3.bar(x + width/2, p2_utils, width, label='Problem 2', alpha=0.8)
    
    ax3.set_xlabel('Unit', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Utilization Rate (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Unit Utilization Rate Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(unit_ids)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 105)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    fig3_path = os.path.join(comparison_dir, f"3_utilization_comparison_{timestamp}.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {fig3_path}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("All comparison visualizations saved successfully!")
    print(f"Comparison directory: {comparison_dir}")
    print("=" * 70)

# ============================================================================
# Save Comparison Summary
# ============================================================================

comparison_summary = {
    'comparison_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
    'problem1': {
        'results_dir': problem1_dir,
        'total_cost': p1_cost,
        'fuel_cost': p1_fuel,
        'startup_cost': p1_startup,
        'shutdown_cost': p1_shutdown
    },
    'problem2': {
        'results_dir': problem2_dir,
        'total_cost': p2_cost,
        'fuel_cost': p2_fuel,
        'startup_cost': p2_startup,
        'shutdown_cost': p2_shutdown,
        'spinning_reserve_enabled': p2_summary['optimization_info'].get('spinning_reserve_enabled', False),
        'n1_security_enabled': p2_summary['optimization_info'].get('n1_security_enabled', False),
        'inertia_constraint_enabled': p2_summary['optimization_info'].get('inertia_constraint_enabled', False)
    },
    'differences': {
        'cost_increase': cost_increase,
        'cost_increase_percent': cost_increase_pct,
        'fuel_cost_increase': p2_fuel - p1_fuel,
        'startup_cost_increase': p2_startup - p1_startup,
        'shutdown_cost_increase': p2_shutdown - p1_shutdown
    }
}

comparison_dir = os.path.join(project_root, "results", "problem2", "comparison")
os.makedirs(comparison_dir, exist_ok=True)
comparison_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
comparison_json_path = os.path.join(comparison_dir, f"comparison_summary_{comparison_timestamp}.json")
with open(comparison_json_path, 'w', encoding='utf-8') as f:
    json.dump(comparison_summary, f, indent=2, ensure_ascii=False)
print(f"\n✓ Comparison summary saved to: {comparison_json_path}")

print("\n" + "=" * 70)
print("Comparison Complete!")
print("=" * 70)

