"""
专门检查 Period 12 的出力情况
"""
import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(project_root, "results", "problem1")
timestamp = "20251121_212827"

# 读取数据
gen_schedule = pd.read_csv(f"{results_dir}/generation_schedule_{timestamp}.csv")
uc_schedule = pd.read_csv(f"{results_dir}/uc_schedule_{timestamp}.csv")

# Unit 13 的最小出力
unit_params = {
    11: {'P_min': 10, 'P_max': 30},
    13: {'P_min': 12, 'P_max': 40},
}

print("=" * 80)
print("Period 12 详细检查")
print("=" * 80)

period_12_idx = 11  # Period 12 (0-indexed) - 保存这个值，避免被循环覆盖

print(f"\nPeriod {period_12_idx + 1} 的出力情况:")
print("-" * 80)

for unit_id in [11, 13]:
    gen_col = f'Unit_{unit_id}_MW'
    status_col = f'Unit_{unit_id}_Status'
    
    gen_value = gen_schedule.loc[period_12_idx, gen_col]
    status_value = uc_schedule.loc[period_12_idx, status_col]
    params = unit_params[unit_id]
    
    print(f"\nUnit {unit_id}:")
    print(f"  状态: {status_value} ({'开机' if status_value == 1 else '停机'})")
    print(f"  出力: {gen_value:.6f} MW")
    print(f"  最小出力要求: {params['P_min']} MW")
    print(f"  最大出力限制: {params['P_max']} MW")
    
    if status_value == 1:  # 开机状态
        if gen_value < params['P_min'] - 1e-3:
            print(f"  ❌ 违反约束: 出力 {gen_value:.6f} < 最小出力 {params['P_min']}")
        elif gen_value > params['P_max'] + 1e-3:
            print(f"  ❌ 违反约束: 出力 {gen_value:.6f} > 最大出力 {params['P_max']}")
        else:
            print(f"  ✅ 满足约束")
    else:
        if gen_value > 1e-3:
            print(f"  ❌ 违反约束: 停机但出力 {gen_value:.6f} > 0")
        else:
            print(f"  ✅ 满足约束 (停机状态)")

# 检查所有时段的 Unit 13
print("\n" + "=" * 80)
print("Unit 13 所有时段的出力检查")
print("=" * 80)

violations = []
for t in range(24):  # 使用不同的变量名，避免覆盖 period_12_idx
    gen_value = gen_schedule.loc[t, 'Unit_13_MW']
    status_value = uc_schedule.loc[t, 'Unit_13_Status']
    
    if status_value == 1 and gen_value < 12 - 1e-3:
        violations.append((t + 1, gen_value, status_value))

if violations:
    print("\n❌ 发现违反最小出力约束的时段:")
    for period, gen, status in violations:
        print(f"  Period {period}: 出力 = {gen:.6f} MW < 12 MW (状态: {status})")
else:
    print("\n✅ Unit 13 所有时段的出力都满足最小出力约束 (≥12 MW)")

# 检查所有时段的 Unit 11
print("\n" + "=" * 80)
print("Unit 11 所有时段的出力检查")
print("=" * 80)

violations_11 = []
for t in range(24):  # 使用不同的变量名，避免覆盖 period_12_idx
    gen_value = gen_schedule.loc[t, 'Unit_11_MW']
    status_value = uc_schedule.loc[t, 'Unit_11_Status']
    
    if status_value == 1 and gen_value < 10 - 1e-3:
        violations_11.append((t + 1, gen_value, status_value))

if violations_11:
    print("\n❌ 发现违反最小出力约束的时段:")
    for period, gen, status in violations_11:
        print(f"  Period {period}: 出力 = {gen:.6f} MW < 10 MW (状态: {status})")
else:
    print("\n✅ Unit 11 所有时段的出力都满足最小出力约束 (≥10 MW)")

# 显示 Period 12 的完整数据
print("\n" + "=" * 80)
print("Period 12 完整数据")
print("=" * 80)
print(f"\n负荷需求: {gen_schedule.loc[period_12_idx, 'Load_MW']} MW")
print(f"总发电量: {gen_schedule.loc[period_12_idx, 'Total_Generation_MW']:.6f} MW")
print("\n各机组出力:")
for unit_id in [1, 2, 5, 8, 11, 13]:
    gen_col = f'Unit_{unit_id}_MW'
    gen_value = gen_schedule.loc[period_12_idx, gen_col]
    print(f"  Unit {unit_id}: {gen_value:.6f} MW")

