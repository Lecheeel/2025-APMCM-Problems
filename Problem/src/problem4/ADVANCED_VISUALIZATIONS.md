# Advanced Visualizations for Problem 4

## 概述

本模块为问题4生成高质量的可视化图表，包括3D图表和专业的2D图表，适合用于学术论文。所有图表都展示了问题规模缩减策略的效果和缩减前后解决方案的对比。

## 功能特性

### 自动生成的高级图表

运行 `main.py` 后，会自动生成以下14种高级可视化，包括8种3D和综合图表，以及6种原理性2D图表：

1. **3D Reduced vs Full Scale Comparison** - 3D缩减前后对比图
   - 展示缩减问题和完整规模问题的发电量3D表面图对比
   - 使用不同颜色方案区分缩减和完整规模解决方案

2. **Reduction Strategy Heatmap** - 缩减策略热力图
   - 展示机组选择策略（哪些机组被选中）
   - 展示时段选择策略（哪些时段被选中）
   - 清晰显示缩减决策

3. **3D Binary Variable Reduction** - 3D二进制变量缩减可视化
   - 三维柱状图展示原始问题、缩减问题和CIM限制的二进制变量数量
   - 直观展示缩减效果

4. **Generation Heatmap Comparison** - 发电量热力图对比
   - 缩减问题和完整规模问题的发电量热力图并排对比
   - 包含数值标注，便于详细分析

5. **3D Cost Surface Comparison** - 3D成本表面对比
   - 展示缩减问题和完整规模问题的成本分布3D表面
   - 显示总成本信息

6. **Cost Comparison Analysis** - 成本对比分析
   - 成本对比柱状图（包括与Problem 2的对比，如果可用）
   - 缩减比例指标可视化

7. **3D Generation Difference Surface** - 3D发电量差异表面
   - 展示完整规模解决方案与缩减解决方案之间的差异
   - 使用发散色彩映射，红色表示增加，蓝色表示减少

8. **Detailed Reduction Information Dashboard** - 详细缩减信息仪表板
   - 综合信息面板，包含：
     - 二进制变量数量对比
     - 缩减比例指标
     - 成本对比
     - 机组选择可视化
     - 时段选择可视化
     - 负荷需求对比
     - 发电量对比（选定时段）

### 原理性2D图表（适合论文）

9. **Problem Reduction Flow Diagram** - 问题缩减流程图
   - 展示从原始问题到缩减问题的完整流程
   - 包含缩减策略、离散化、QUBO构建、解扩展等步骤
   - 清晰展示各阶段的数据规模和约束条件

10. **Discretization Encoding Principle** - 离散化编码原理图
   - 1-bit和2-bit编码的对比展示
   - 各机组的离散化步长可视化
   - 发电量范围编码示意图
   - 展示编码公式：p = P_min + bit × Δ

11. **Time Aggregation Principle** - 时间聚合原理图
   - 展示24个原始时段的选择策略
   - 聚合后时段的负荷对比
   - 时段映射关系可视化
   - 说明均匀聚合方法

12. **QUBO Constraint Transformation Principle** - QUBO约束转换原理图
   - 功率平衡约束的转换过程
   - 约束到惩罚项的数学公式
   - 发电量限制约束可视化
   - 约束违反惩罚机制展示

13. **Solution Expansion Strategy** - 解扩展策略图
   - 缩减解到完整解的扩展过程
   - 最近邻插值方法可视化
   - 选定时段和非选定时段的处理策略
   - 机组扩展方法说明

14. **Binary Variable Encoding Matrix** - 二进制变量编码矩阵
   - 二进制变量的索引结构
   - QUBO矩阵的结构可视化
   - 变量到发电量的映射关系
   - QUBO形式的数学表示

## 使用方法

### 基本使用

```bash
# 运行问题4求解器（会自动生成高级可视化）
python src/problem4/main.py
```

### 独立运行可视化

如果已有结果文件，可以单独运行可视化模块：

```bash
# 使用最新结果
python src/problem4/advanced_visualizations.py

# 指定结果目录和时间戳
python src/problem4/advanced_visualizations.py \
    --results-dir ../results/problem4 \
    --timestamp 20251124_060428 \
    --problem2-dir ../results/problem2
```

### 参数说明

- `--results-dir`: Problem 4结果目录（默认：`../results/problem4`）
- `--timestamp`: 特定时间戳（如果为None，使用最新结果）
- `--problem2-dir`: Problem 2结果目录（用于对比，可选）

## 输出文件

所有高级可视化图表保存在 `results/problem4/` 目录下，文件名格式：

**3D和综合图表：**
- `advanced_1_3d_reduced_vs_full_[timestamp].png`
- `advanced_2_reduction_strategy_heatmap_[timestamp].png`
- `advanced_3_3d_binary_vars_reduction_[timestamp].png`
- `advanced_4_generation_heatmap_comparison_[timestamp].png`
- `advanced_5_3d_cost_surface_[timestamp].png`
- `advanced_6_cost_comparison_[timestamp].png`
- `advanced_7_3d_difference_surface_[timestamp].png`
- `advanced_8_reduction_dashboard_[timestamp].png`

**原理性2D图表：**
- `principle_1_reduction_flow_[timestamp].png`
- `principle_2_discretization_[timestamp].png`
- `principle_3_time_aggregation_[timestamp].png`
- `principle_4_qubo_constraints_[timestamp].png`
- `principle_5_solution_expansion_[timestamp].png`
- `principle_6_binary_encoding_[timestamp].png`

## 图表特点

### 论文质量
- 高分辨率（300 DPI）
- 专业配色方案
- 清晰的标签和标题
- 适合学术发表

### 信息丰富
- 包含详细的数值标注
- 多角度展示数据
- 对比分析清晰
- 综合信息面板

### 3D可视化
- 多个3D图表展示不同维度
- 可调节视角
- 颜色映射清晰
- 适合展示复杂关系

## 技术细节

### 依赖库
- `matplotlib`: 基础绘图
- `numpy`: 数值计算
- `pandas`: 数据处理（可选，用于CSV读取）
- `mpl_toolkits.mplot3d`: 3D绘图支持

### 数据格式
可视化模块从以下文件读取数据：
- `summary_[timestamp].json`: 包含优化结果和缩减信息
- `reduced_generation_schedule_[timestamp].csv`: 缩减问题发电量计划（可选）
- `full_generation_schedule_[timestamp].csv`: 完整规模发电量计划（可选）

## 注意事项

1. 如果matplotlib库不可用，可视化会被跳过，但不会影响主程序运行
2. 如果Problem 2结果不可用，某些对比图表会相应调整
3. 所有图表使用白色背景，适合论文使用
4. 图表尺寸和字体大小已优化，适合打印和展示

## 示例输出

运行后会在控制台看到：
```
[Step 11] Generating advanced visualizations...
Loading Problem 4 results...
Creating Advanced Visualization 1: 3D Reduced vs Full Scale Comparison...
  ✓ Saved to: .../advanced_1_3d_reduced_vs_full_20251124_060428.png
Creating Advanced Visualization 2: Reduction Strategy Heatmap...
  ✓ Saved to: .../advanced_2_reduction_strategy_heatmap_20251124_060428.png
...
Advanced visualizations completed successfully!
```

## 扩展

如需添加新的可视化图表，可以在 `create_advanced_visualizations` 函数中添加新的可视化代码块，遵循现有的代码风格和命名规范。

