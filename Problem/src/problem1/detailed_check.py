"""
详细检查所有时段的出力情况，特别是接近最小出力限制的时段
"""
import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(project_root, "results", "problem1")
timestamp = "20251121_212827"

# 读取数据
gen_schedule = pd.read_csv(f"{results_dir}/generation_schedule_{timestamp}.csv")
uc_schedule = pd.read_csv(f"{results_dir}/uc_schedule_{timestamp}.csv")

# 参数
unit_params = {
    1: {'P_min': 50, 'P_max': 300},
    2: {'P_min': 20, 'P_max': 180},
    5: {'P_min': 15, 'P_max': 50},
    8: {'P_min': 10, 'P_max': 35},
    11: {'P_min': 10, 'P_max': 30},
    13: {'P_min': 12, 'P_max': 40},
}

print("=" * 100)
print("所有时段出力详细检查报告")
print("=" * 100)

# 检查 Unit 13
print("\n" + "=" * 100)
print("Unit 13 所有时段出力检查 (最小出力要求: 12 MW)")
print("=" * 100)
print(f"{'Period':<8} {'状态':<6} {'出力 (MW)':<15} {'最小要求':<10} {'状态':<10}")
print("-" * 100)

violations_13 = []
for period_idx in range(24):
    period = period_idx + 1
    gen_value = gen_schedule.loc[period_idx, 'Unit_13_MW']
    status_value = uc_schedule.loc[period_idx, 'Unit_13_Status']
    p_min = unit_params[13]['P_min']
    
    status_str = "开机" if status_value == 1 else "停机"
    check_str = ""
    
    if status_value == 1:
        if gen_value < p_min - 1e-3:
            check_str = "❌ 违反"
            violations_13.append((period, gen_value, status_value))
        elif gen_value < p_min + 0.1:  # 接近最小出力
            check_str = "⚠️ 接近"
        else:
            check_str = "✅"
    else:
        check_str = "✅" if gen_value < 1e-3 else "❌ 违反"
    
    print(f"{period:<8} {status_str:<6} {gen_value:<15.6f} {p_min:<10} {check_str:<10}")

if violations_13:
    print(f"\n❌ Unit 13 发现 {len(violations_13)} 个违反最小出力约束的时段:")
    for period, gen, status in violations_13:
        print(f"  Period {period}: 出力 = {gen:.6f} MW < 12 MW")
else:
    print(f"\n✅ Unit 13 所有时段的出力都满足最小出力约束 (≥12 MW)")

# 检查 Unit 11
print("\n" + "=" * 100)
print("Unit 11 所有时段出力检查 (最小出力要求: 10 MW)")
print("=" * 100)
print(f"{'Period':<8} {'状态':<6} {'出力 (MW)':<15} {'最小要求':<10} {'状态':<10}")
print("-" * 100)

violations_11 = []
for period_idx in range(24):
    period = period_idx + 1
    gen_value = gen_schedule.loc[period_idx, 'Unit_11_MW']
    status_value = uc_schedule.loc[period_idx, 'Unit_11_Status']
    p_min = unit_params[11]['P_min']
    
    status_str = "开机" if status_value == 1 else "停机"
    check_str = ""
    
    if status_value == 1:
        if gen_value < p_min - 1e-3:
            check_str = "❌ 违反"
            violations_11.append((period, gen_value, status_value))
        elif gen_value < p_min + 0.1:  # 接近最小出力
            check_str = "⚠️ 接近"
        else:
            check_str = "✅"
    else:
        check_str = "✅" if gen_value < 1e-3 else "❌ 违反"
    
    print(f"{period:<8} {status_str:<6} {gen_value:<15.6f} {p_min:<10} {check_str:<10}")

if violations_11:
    print(f"\n❌ Unit 11 发现 {len(violations_11)} 个违反最小出力约束的时段:")
    for period, gen, status in violations_11:
        print(f"  Period {period}: 出力 = {gen:.6f} MW < 10 MW")
else:
    print(f"\n✅ Unit 11 所有时段的出力都满足最小出力约束 (≥10 MW)")

# 特别检查 Period 12
print("\n" + "=" * 100)
print("Period 12 详细数据")
print("=" * 100)
period_idx = 11
print(f"\n负荷需求: {gen_schedule.loc[period_idx, 'Load_MW']} MW")
print(f"总发电量: {gen_schedule.loc[period_idx, 'Total_Generation_MW']:.10f} MW")
print(f"\n各机组出力:")
for unit_id in [1, 2, 5, 8, 11, 13]:
    gen_col = f'Unit_{unit_id}_MW'
    status_col = f'Unit_{unit_id}_Status'
    gen_value = gen_schedule.loc[period_idx, gen_col]
    status_value = uc_schedule.loc[period_idx, status_col]
    params = unit_params[unit_id]
    
    status_str = "开机" if status_value == 1 else "停机"
    if status_value == 1:
        if gen_value < params['P_min'] - 1e-3:
            check = "❌ 违反"
        elif gen_value < params['P_min'] + 0.1:
            check = "⚠️ 接近"
        else:
            check = "✅"
    else:
        check = "✅" if gen_value < 1e-3 else "❌ 违反"
    
    print(f"  Unit {unit_id:2d}: {gen_value:12.6f} MW (状态: {status_str}, 范围: {params['P_min']}-{params['P_max']} MW) {check}")

print("\n" + "=" * 100)

