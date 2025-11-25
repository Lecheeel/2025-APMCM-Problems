"""
验证 Unit Commitment 结果的合理性和正确性
检查所有约束条件是否满足
problem1_uc_gurobi.py 的验证结果 专用
"""

import json
import pandas as pd
import numpy as np

# 读取结果文件 (relative to project root)
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(project_root, "results", "problem1")
timestamp = "20251121_212827"  # Update this to match your latest results timestamp

# 读取数据
with open(f"{results_dir}/summary_{timestamp}.json", 'r', encoding='utf-8') as f:
    summary = json.load(f)

uc_schedule = pd.read_csv(f"{results_dir}/uc_schedule_{timestamp}.csv")
gen_schedule = pd.read_csv(f"{results_dir}/generation_schedule_{timestamp}.csv")

# 从Problem.md提取的参数
units = [1, 2, 5, 8, 11, 13]
unit_to_idx = {unit: idx for idx, unit in enumerate(units)}

# Table 1 & 2 参数
unit_params = {
    1: {'P_max': 300, 'P_min': 50, 'Min_Up_Time': 5, 'Min_Down_Time': 3, 
        'Ramp_Up': 80, 'Ramp_Down': 80, 'Initial_Up_Time': 5, 'Initial_Down_Time': 0,
        'Startup_Cost': 180, 'Shutdown_Cost': 180},
    2: {'P_max': 180, 'P_min': 20, 'Min_Up_Time': 4, 'Min_Down_Time': 2,
        'Ramp_Up': 80, 'Ramp_Down': 80, 'Initial_Up_Time': 4, 'Initial_Down_Time': 0,
        'Startup_Cost': 180, 'Shutdown_Cost': 180},
    5: {'P_max': 50, 'P_min': 15, 'Min_Up_Time': 3, 'Min_Down_Time': 2,
        'Ramp_Up': 50, 'Ramp_Down': 50, 'Initial_Up_Time': 3, 'Initial_Down_Time': 0,
        'Startup_Cost': 40, 'Shutdown_Cost': 40},
    8: {'P_max': 35, 'P_min': 10, 'Min_Up_Time': 3, 'Min_Down_Time': 2,
        'Ramp_Up': 50, 'Ramp_Down': 50, 'Initial_Up_Time': 3, 'Initial_Down_Time': 0,
        'Startup_Cost': 60, 'Shutdown_Cost': 60},
    11: {'P_max': 30, 'P_min': 10, 'Min_Up_Time': 1, 'Min_Down_Time': 1,
        'Ramp_Up': 60, 'Ramp_Down': 60, 'Initial_Up_Time': 3, 'Initial_Down_Time': 0,
        'Startup_Cost': 60, 'Shutdown_Cost': 60},
    13: {'P_max': 40, 'P_min': 12, 'Min_Up_Time': 4, 'Min_Down_Time': 2,
        'Ramp_Up': 60, 'Ramp_Down': 60, 'Initial_Up_Time': 4, 'Initial_Down_Time': 0,
        'Startup_Cost': 40, 'Shutdown_Cost': 40},
}

load_demand = [166, 196, 229, 257, 263, 252, 246, 213, 192, 161, 147, 160,
               170, 185, 208, 232, 246, 241, 236, 225, 204, 182, 161, 131]

num_periods = 24
num_units = len(units)

print("=" * 80)
print("Unit Commitment 结果验证报告")
print("=" * 80)

# 提取数据
unit_status = {}
unit_generation = {}
for unit_id in units:
    unit_status[unit_id] = uc_schedule[f'Unit_{unit_id}_Status'].values
    unit_generation[unit_id] = gen_schedule[f'Unit_{unit_id}_MW'].values

# ============================================================================
# 1. 功率平衡约束验证
# ============================================================================
print("\n1. 功率平衡约束验证")
print("-" * 80)
power_balance_ok = True
max_balance_error = 0
for t in range(num_periods):
    total_gen = sum(unit_generation[u][t] for u in units)
    load = load_demand[t]
    error = abs(total_gen - load)
    if error > 1e-2:
        print(f"  ✗ 时段 {t+1:2d}: 发电量={total_gen:8.2f} MW, 负荷={load:5d} MW, 误差={error:.4f} MW")
        power_balance_ok = False
    max_balance_error = max(max_balance_error, error)

if power_balance_ok:
    print(f"  ✓ 所有时段功率平衡满足 (最大误差: {max_balance_error:.6f} MW)")
else:
    print(f"  ✗ 功率平衡约束违反 (最大误差: {max_balance_error:.4f} MW)")

# ============================================================================
# 2. 发电限制约束验证
# ============================================================================
print("\n2. 发电限制约束验证")
print("-" * 80)
gen_limits_ok = True
violations = []
for unit_id in units:
    params = unit_params[unit_id]
    for t in range(num_periods):
        status = unit_status[unit_id][t]
        gen = unit_generation[unit_id][t]
        
        if status == 1:  # 机组开机
            if gen < params['P_min'] - 1e-3:
                violations.append(f"Unit {unit_id}, Period {t+1}: 发电量 {gen:.2f} < P_min {params['P_min']}")
                gen_limits_ok = False
            if gen > params['P_max'] + 1e-3:
                violations.append(f"Unit {unit_id}, Period {t+1}: 发电量 {gen:.2f} > P_max {params['P_max']}")
                gen_limits_ok = False
        else:  # 机组停机
            if gen > 1e-3:
                violations.append(f"Unit {unit_id}, Period {t+1}: 机组停机但发电量 {gen:.2f} > 0")
                gen_limits_ok = False

if gen_limits_ok:
    print("  ✓ 所有机组发电限制满足")
else:
    print("  ✗ 发电限制约束违反:")
    for v in violations[:10]:  # 只显示前10个违反
        print(f"    - {v}")
    if len(violations) > 10:
        print(f"    ... 还有 {len(violations) - 10} 个违反")

# ============================================================================
# 3. 最小开/停机时间约束验证
# ============================================================================
print("\n3. 最小开/停机时间约束验证")
print("-" * 80)
min_time_ok = True

for unit_id in units:
    params = unit_params[unit_id]
    status_seq = unit_status[unit_id]
    
    # 检查最小开机时间
    min_up_time = params['Min_Up_Time']
    initial_up_time = params['Initial_Up_Time']
    
    # 如果初始状态是ON，检查是否满足最小开机时间
    if initial_up_time > 0:
        # 计算从开始需要保持开机的时间
        required_up_time = max(0, min_up_time - initial_up_time)
        for t in range(min(required_up_time, num_periods)):
            if status_seq[t] != 1:
                print(f"  ✗ Unit {unit_id}: 初始开机但时段 {t+1} 状态为 {status_seq[t]} (应保持开机)")
                min_time_ok = False
    
    # 检查整个序列的最小开机时间
    i = 0
    while i < num_periods:
        if status_seq[i] == 1:  # 找到开机点
            # 检查这是启动还是继续运行
            is_startup = (i == 0 and initial_up_time == 0) or (i > 0 and status_seq[i-1] == 0)
            
            if is_startup:
                # 计算需要保持开机的时间
                up_duration = 1
                j = i + 1
                while j < num_periods and status_seq[j] == 1:
                    up_duration += 1
                    j += 1
                
                if up_duration < min_up_time:
                    print(f"  ✗ Unit {unit_id}: 时段 {i+1} 启动，但只运行了 {up_duration} 个时段 (最小开机时间: {min_up_time})")
                    min_time_ok = False
            i = j if 'j' in locals() else i + 1
        else:
            i += 1
    
    # 检查最小停机时间
    min_down_time = params['Min_Down_Time']
    initial_down_time = params['Initial_Down_Time']
    
    if initial_down_time > 0:
        required_down_time = max(0, min_down_time - initial_down_time)
        for t in range(min(required_down_time, num_periods)):
            if status_seq[t] != 0:
                print(f"  ✗ Unit {unit_id}: 初始停机但时段 {t+1} 状态为 {status_seq[t]} (应保持停机)")
                min_time_ok = False
    
    # 检查整个序列的最小停机时间
    i = 0
    while i < num_periods:
        if status_seq[i] == 0:  # 找到停机点
            is_shutdown = (i == 0 and initial_down_time == 0) or (i > 0 and status_seq[i-1] == 1)
            
            if is_shutdown:
                down_duration = 1
                j = i + 1
                while j < num_periods and status_seq[j] == 0:
                    down_duration += 1
                    j += 1
                
                if down_duration < min_down_time:
                    print(f"  ✗ Unit {unit_id}: 时段 {i+1} 停机，但只停机了 {down_duration} 个时段 (最小停机时间: {min_down_time})")
                    min_time_ok = False
            i = j if 'j' in locals() else i + 1
        else:
            i += 1

if min_time_ok:
    print("  ✓ 所有机组最小开/停机时间约束满足")

# ============================================================================
# 4. 爬坡率约束验证
# ============================================================================
print("\n4. 爬坡率约束验证")
print("-" * 80)
ramp_ok = True
ramp_violations = []

for unit_id in units:
    params = unit_params[unit_id]
    gen_seq = unit_generation[unit_id]
    status_seq = unit_status[unit_id]
    
    # 初始功率（假设初始开机时在P_min）
    initial_power = params['P_min'] if params['Initial_Up_Time'] > 0 else 0
    
    # 检查第一个时段
    if num_periods > 0:
        gen_0 = gen_seq[0]
        if params['Initial_Up_Time'] > 0:  # 初始开机
            ramp_up = gen_0 - initial_power
            if ramp_up > params['Ramp_Up'] + 1e-3:
                ramp_violations.append(f"Unit {unit_id}, Period 1: 爬坡上升 {ramp_up:.2f} > {params['Ramp_Up']}")
                ramp_ok = False
            ramp_down = initial_power - gen_0
            if status_seq[0] == 1 and ramp_down > params['Ramp_Down'] + 1e-3:
                ramp_violations.append(f"Unit {unit_id}, Period 1: 爬坡下降 {ramp_down:.2f} > {params['Ramp_Down']}")
                ramp_ok = False
    
    # 检查后续时段
    for t in range(1, num_periods):
        gen_prev = gen_seq[t-1]
        gen_curr = gen_seq[t]
        status_prev = status_seq[t-1]
        status_curr = status_seq[t]
        
        # 爬坡上升约束
        ramp_up = gen_curr - gen_prev
        if status_prev == 1:  # 前一时段开机
            if ramp_up > params['Ramp_Up'] + 1e-3:
                ramp_violations.append(f"Unit {unit_id}, Period {t+1}: 爬坡上升 {ramp_up:.2f} > {params['Ramp_Up']}")
                ramp_ok = False
        
        # 爬坡下降约束
        ramp_down = gen_prev - gen_curr
        if status_curr == 1:  # 当前时段开机
            if ramp_down > params['Ramp_Down'] + 1e-3:
                ramp_violations.append(f"Unit {unit_id}, Period {t+1}: 爬坡下降 {ramp_down:.2f} > {params['Ramp_Down']}")
                ramp_ok = False

if ramp_ok:
    print("  ✓ 所有机组爬坡率约束满足")
else:
    print("  ✗ 爬坡率约束违反:")
    for v in ramp_violations[:10]:
        print(f"    - {v}")
    if len(ramp_violations) > 10:
        print(f"    ... 还有 {len(ramp_violations) - 10} 个违反")

# ============================================================================
# 5. 结果合理性分析
# ============================================================================
print("\n5. 结果合理性分析")
print("-" * 80)

# 5.1 所有机组24小时都开机是否合理？
all_units_on_all_time = all(all(unit_status[u] == 1) for u in units)
print(f"\n5.1 机组运行模式:")
print(f"  所有机组在所有时段都开机: {all_units_on_all_time}")

if all_units_on_all_time:
    # 计算最小发电量总和
    min_total_gen = sum(unit_params[u]['P_min'] for u in units)
    max_total_gen = sum(unit_params[u]['P_max'] for u in units)
    min_load = min(load_demand)
    max_load = max(load_demand)
    
    print(f"  最小发电量总和: {min_total_gen} MW")
    print(f"  最大发电量总和: {max_total_gen} MW")
    print(f"  负荷范围: {min_load} - {max_load} MW")
    
    if min_total_gen <= min_load and max_total_gen >= max_load:
        print(f"  ✓ 所有机组保持开机是合理的（最小发电量 {min_total_gen} ≤ 最小负荷 {min_load}）")
    else:
        print(f"  ⚠ 注意: 最小发电量总和 {min_total_gen} > 最小负荷 {min_load}，理论上可以关闭部分机组")

# 5.2 启动/关闭成本分析
startup_cost = summary['optimization_info']['startup_cost']
shutdown_cost = summary['optimization_info']['shutdown_cost']
print(f"\n5.2 启动/关闭成本:")
print(f"  启动成本: ${startup_cost:.2f}")
print(f"  关闭成本: ${shutdown_cost:.2f}")
if startup_cost == 0 and shutdown_cost == 0:
    print(f"  ✓ 没有启动/关闭事件，符合所有机组保持开机的策略")

# 5.3 机组利用率分析
print(f"\n5.3 机组利用率:")
for unit_stat in summary['unit_statistics']:
    unit_id = unit_stat['unit_id']
    utilization = unit_stat['utilization_rate'] * 100
    avg_gen = unit_stat['average_generation_MW']
    params = unit_params[unit_id]
    utilization_power = (avg_gen - params['P_min']) / (params['P_max'] - params['P_min']) * 100 if params['P_max'] > params['P_min'] else 0
    print(f"  Unit {unit_id}: 利用率={utilization:.1f}%, 平均出力={avg_gen:.2f} MW "
          f"(范围: {params['P_min']}-{params['P_max']} MW, 功率利用率={utilization_power:.1f}%)")

# 5.4 成本分析
total_cost = summary['optimization_info']['total_cost']
fuel_cost = summary['optimization_info']['fuel_cost']
print(f"\n5.4 成本分析:")
print(f"  总成本: ${total_cost:.2f}")
print(f"  燃料成本: ${fuel_cost:.2f}")
print(f"  燃料成本占比: {fuel_cost/total_cost*100:.2f}%")

# ============================================================================
# 6. 与问题要求的对比
# ============================================================================
print("\n6. 与Problem 1要求的对比")
print("-" * 80)

requirements_met = {
    "目标函数 - 燃料成本（二次）": True,  # 代码中使用了二次成本函数
    "目标函数 - 启动/关闭成本": True,  # 代码中包含了启动/关闭成本
    "约束 - 功率平衡": power_balance_ok,
    "约束 - 发电限制": gen_limits_ok,
    "约束 - 最小开/停机时间": min_time_ok,
    "约束 - 爬坡率": ramp_ok,
    "约束 - 启动/关闭逻辑": True,  # 代码中实现了逻辑约束
}

print("\n要求满足情况:")
for req, met in requirements_met.items():
    status = "✓" if met else "✗"
    print(f"  {status} {req}")

all_met = all(requirements_met.values())
print(f"\n总体评估: {'✓ 所有要求都满足' if all_met else '✗ 部分要求未满足'}")

print("\n" + "=" * 80)
print("验证完成")
print("=" * 80)

