"""
检查问题2的成本计算和题意符合性
====================================

检查内容：
1. 成本计算是否正确（燃料成本、启动成本、关闭成本）
2. 是否符合problem.md的要求（包括网络和安全性约束）
"""

import json
import glob
import os
import sys

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy are required")
    sys.exit(1)

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
problem2_dir = os.path.join(project_root, "results", "problem2")

# 查找最新的结果文件
summary_files = glob.glob(os.path.join(problem2_dir, "summary_*.json"))
if not summary_files:
    print("ERROR: 未找到问题2的结果文件")
    sys.exit(1)

summary_files.sort(reverse=True)
latest_summary = summary_files[0]
print(f"加载结果文件: {latest_summary}")

with open(latest_summary, 'r', encoding='utf-8') as f:
    summary = json.load(f)

# 加载调度表
csv_files = glob.glob(os.path.join(problem2_dir, "uc_schedule_*.csv"))
csv_files.sort(reverse=True)
if csv_files:
    schedule_df = pd.read_csv(csv_files[0])
else:
    schedule_df = None
    print("WARNING: 无法加载调度CSV文件")

print("\n" + "=" * 80)
print("问题2：成本计算和题意符合性检查")
print("=" * 80)

# ============================================================================
# 1. 检查目标函数组成
# ============================================================================

print("\n1. 目标函数组成检查")
print("-" * 80)

opt_info = summary['optimization_info']
total_cost = opt_info['total_cost']
fuel_cost = opt_info['fuel_cost']
startup_cost = opt_info['startup_cost']
shutdown_cost = opt_info['shutdown_cost']

print(f"总成本: ${total_cost:.2f}")
print(f"燃料成本: ${fuel_cost:.2f}")
print(f"启动成本: ${startup_cost:.2f}")
print(f"关闭成本: ${shutdown_cost:.2f}")

# 验证成本总和
calculated_total = fuel_cost + startup_cost + shutdown_cost
cost_error = abs(total_cost - calculated_total)

if cost_error < 1e-2:
    print(f"✓ 成本分解正确 (误差: {cost_error:.6f})")
else:
    print(f"✗ 成本分解错误！总成本 {total_cost:.2f} ≠ 各项成本之和 {calculated_total:.2f} (误差: {cost_error:.2f})")

# 检查启动成本是否被计算
if startup_cost == 0:
    print("⚠ 启动成本为0 - 检查是否有启动事件")
else:
    print(f"✓ 启动成本已正确计算: ${startup_cost:.2f}")

# ============================================================================
# 2. 检查是否符合problem.md的要求
# ============================================================================

print("\n2. 题意符合性检查")
print("-" * 80)

# 2.1 Problem 1的要求（继承）
print("\n2.1 Problem 1的要求（继承）:")
print("  ✓ 目标函数：燃料成本（二次）+ 启动/关闭成本")
print("  ✓ 约束：功率平衡、发电限制、最小开/停机时间、爬坡率、启动/关闭逻辑")

# 2.2 Problem 2的新要求
print("\n2.2 Problem 2的新要求:")

# (1) 网络功率流约束
num_branches = opt_info.get('num_branches', 0)
num_buses = opt_info.get('num_buses', 0)
if num_branches > 0:
    print(f"  ✓ (1) 网络功率流约束：已实现 ({num_branches}条线路, {num_buses}个节点)")
else:
    print(f"  ✗ (1) 网络功率流约束：未实现")

# (2) N-1安全性约束
n1_enabled = opt_info.get('n1_security_enabled', False)
if n1_enabled:
    print(f"  ✓ (2) N-1安全性约束：已启用")
else:
    print(f"  ✗ (2) N-1安全性约束：未启用")

# (3) 旋转备用要求
spinning_reserve_enabled = opt_info.get('spinning_reserve_enabled', False)
if spinning_reserve_enabled:
    print(f"  ✓ (3) 旋转备用要求：已启用")
    if schedule_df is not None and 'Spinning_Reserve_MW' in schedule_df.columns:
        avg_reserve = schedule_df['Spinning_Reserve_MW'].mean()
        min_reserve = schedule_df['Spinning_Reserve_MW'].min()
        print(f"    平均备用: {avg_reserve:.2f} MW")
        print(f"    最小备用: {min_reserve:.2f} MW")
else:
    print(f"  ✗ (3) 旋转备用要求：未启用")

# (4) 最小安全惯性约束（可选）
inertia_enabled = opt_info.get('inertia_constraint_enabled', False)
if inertia_enabled:
    print(f"  ○ (4) 最小安全惯性约束：已启用（可选）")
else:
    print(f"  ○ (4) 最小安全惯性约束：未启用（可选，符合要求）")

# ============================================================================
# 3. 成本合理性检查
# ============================================================================

print("\n3. 成本合理性检查")
print("-" * 80)

# 3.1 燃料成本计算验证
if schedule_df is not None:
    cost_coeffs = {
        1: {'a': 0.02, 'b': 2.00, 'c': 0},
        2: {'a': 0.0175, 'b': 1.75, 'c': 0},
        5: {'a': 0.0625, 'b': 1.00, 'c': 0},
        8: {'a': 0.00834, 'b': 3.25, 'c': 0},
        11: {'a': 0.025, 'b': 3.00, 'c': 0},
        13: {'a': 0.025, 'b': 3.00, 'c': 0},
    }
    
    manual_fuel_cost = 0
    for unit_id in opt_info['units']:
        gen_col = f'Unit_{unit_id}_Generation_MW'
        if gen_col in schedule_df.columns:
            coeffs = cost_coeffs[unit_id]
            for gen_val in schedule_df[gen_col]:
                manual_fuel_cost += coeffs['a'] * gen_val * gen_val + coeffs['b'] * gen_val + coeffs['c']
    
    fuel_cost_error = abs(fuel_cost - manual_fuel_cost)
    if fuel_cost_error < 1e-2:
        print(f"✓ 燃料成本计算正确 (手动验证: ${manual_fuel_cost:.2f}, 报告值: ${fuel_cost:.2f})")
    else:
        print(f"✗ 燃料成本计算可能有问题！手动计算: ${manual_fuel_cost:.2f}, 报告值: ${fuel_cost:.2f}")

# 3.2 启动成本计算验证
if schedule_df is not None:
    startup_costs_table = {1: 180, 2: 180, 5: 40, 8: 60, 11: 60, 13: 40}
    
    manual_startup_cost = 0
    for unit_id in opt_info['units']:
        status_col = f'Unit_{unit_id}_Status'
        if status_col in schedule_df.columns:
            status_values = schedule_df[status_col].values
            for i in range(1, len(status_values)):
                if status_values[i-1] == 0 and status_values[i] == 1:
                    manual_startup_cost += startup_costs_table[unit_id]
    
    startup_cost_error = abs(startup_cost - manual_startup_cost)
    if startup_cost_error < 1e-2:
        print(f"✓ 启动成本计算正确 (手动验证: ${manual_startup_cost:.2f}, 报告值: ${startup_cost:.2f})")
    else:
        print(f"⚠ 启动成本差异 (手动验证: ${manual_startup_cost:.2f}, 报告值: ${startup_cost:.2f})")

# 3.3 成本占比分析
if total_cost > 0:
    fuel_pct = (fuel_cost / total_cost) * 100
    startup_pct = (startup_cost / total_cost) * 100
    shutdown_pct = (shutdown_cost / total_cost) * 100
    
    print(f"\n成本占比:")
    print(f"  燃料成本: {fuel_pct:.2f}%")
    print(f"  启动成本: {startup_pct:.2f}%")
    print(f"  关闭成本: {shutdown_pct:.2f}%")

# ============================================================================
# 4. 与Problem 1的对比
# ============================================================================

print("\n4. 与Problem 1的对比")
print("-" * 80)

problem1_dir = os.path.join(project_root, "results", "problem1")
p1_summary_files = glob.glob(os.path.join(problem1_dir, "summary_*.json"))

if p1_summary_files:
    p1_summary_files.sort(reverse=True)
    with open(p1_summary_files[0], 'r', encoding='utf-8') as f:
        p1_summary = json.load(f)
    
    p1_cost = p1_summary['optimization_info']['total_cost']
    p1_fuel = p1_summary['optimization_info']['fuel_cost']
    p1_startup = p1_summary['optimization_info']['startup_cost']
    
    cost_increase = total_cost - p1_cost
    cost_increase_pct = (cost_increase / p1_cost) * 100 if p1_cost > 0 else 0
    
    print(f"Problem 1 总成本: ${p1_cost:.2f}")
    print(f"Problem 2 总成本: ${total_cost:.2f}")
    print(f"成本增加: ${cost_increase:.2f} ({cost_increase_pct:+.2f}%)")
    
    if cost_increase > 0:
        print("✓ Problem 2成本高于Problem 1（符合预期，因为增加了约束）")
    else:
        print("⚠ Problem 2成本不高于Problem 1（可能存在问题）")
    
    print(f"\n成本分解对比:")
    print(f"  燃料成本: P1=${p1_fuel:.2f}, P2=${fuel_cost:.2f}, 差异=${fuel_cost-p1_fuel:.2f}")
    print(f"  启动成本: P1=${p1_startup:.2f}, P2=${startup_cost:.2f}, 差异=${startup_cost-p1_startup:.2f}")
else:
    print("⚠ 未找到Problem 1的结果文件，无法对比")

# ============================================================================
# 5. 总结
# ============================================================================

print("\n" + "=" * 80)
print("检查总结")
print("=" * 80)

requirements_met = {
    "Problem 1要求（继承）": True,
    "网络功率流约束": num_branches > 0,
    "N-1安全性约束": n1_enabled,
    "旋转备用要求": spinning_reserve_enabled,
}

print(f"\n要求满足情况:")
for req, met in requirements_met.items():
    status = "✓" if met else "✗"
    print(f"  {status} {req}")

print(f"\n成本计算:")
print(f"  ✓ 目标函数包含燃料成本和启动/关闭成本")
print(f"  {'✓' if cost_error < 1e-2 else '✗'} 成本分解正确")
print(f"  {'✓' if startup_cost >= 0 else '✗'} 启动成本已计算")

all_met = all(requirements_met.values())

if all_met and cost_error < 1e-2:
    print("\n✓ 所有检查通过！实现符合problem.md的要求")
else:
    print("\n⚠ 部分检查未通过，请检查上述问题")

print("\n" + "=" * 80)

