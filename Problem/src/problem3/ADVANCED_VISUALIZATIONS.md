# Advanced Visualizations for Problem 3

## 概述

本模块为问题3（QUBO转换和量子求解）生成高质量的可视化图表，包括3D图表、专业的2D图表以及理论原理图和流程图，特别针对离散化特性和QUBO解的特点，适合用于学术论文。

## 可视化类型

### 1. 高级3D可视化 (Advanced Visualizations)
生成13个高级3D可视化图表，展示QUBO解的3D特性和对比分析。

### 2. 理论原理图 (Theoretical Visualizations) ⭐ 新增
生成10个高级2D原理图和流程图，深入展示QUBO转换的理论基础和流程。

## 功能特性

### 自动生成的高级图表

运行 `main.py` 后，会自动生成以下高级可视化：

1. **3D QUBO Generation Surface** - 3D QUBO发电量表面图
   - 展示离散化后的发电量分布
   - 红色标记点显示离散值（QUBO特性）

2. **QUBO vs Continuous 3D Comparison** - QUBO与连续解3D对比
   - 并排展示QUBO解和Problem 2的连续解
   - 直观展示离散化带来的差异

3. **Discretization Error Surface** - 离散化误差表面图
   - 3D展示QUBO解与连续解之间的误差
   - 红色标记突出显示大误差区域

4. **Constraint Violation Analysis** - 约束违反分析（3D）
   - 展示功率平衡约束的违反情况
   - 3D表面图显示违反程度

5. **Discrete Value Distribution** - 离散值分布（3D柱状图）
   - 展示每个机组使用的离散值及其频率
   - 体现QUBO离散化的特点

6. **Multi-Panel QUBO Dashboard** - 多面板QUBO分析仪表板
   - 发电vs负荷对比
   - 功率平衡违反情况
   - 发电量热力图
   - 成本对比

7. **Contour Comparison** - 等高线对比图
   - QUBO解的等高线图
   - 与连续解的对比（如果可用）

8. **Multi-Angle 3D View** - 多角度3D视图
   - 四个不同视角的3D发电量景观图
   - 包括等轴测图、俯视图、正视图、侧视图

9. **Detailed Side-by-Side 3D Comparison** - 详细并排3D对比
   - 并排展示QUBO和连续解的3D表面图
   - 所有离散值点都用红色标记

10. **Comprehensive Heatmap Comparison** - 综合热图对比
    - 绝对差异热图（QUBO - 连续）
    - 相对差异热图（百分比误差）
    - QUBO发电量热图
    - 连续发电量热图

11. **3D Difference Surface** - 3D差异表面图
    - 绝对差异3D表面图
    - 相对差异3D表面图
    - 突出显示大误差区域

12. **Unit-wise Comparison** - 逐机组对比
    - 每个机组的QUBO vs 连续解对比
    - 每个机组的差异柱状图
    - 详细的逐时段分析

13. **Multi-Metric 3D Dashboard** - 多指标3D仪表板
    - 四个3D子图展示不同指标
    - 发电量对比、绝对误差、相对误差

### 理论原理图 (Theoretical Visualizations) ⭐ 新增

运行 `main.py` 后，会自动生成以下10个高级2D原理图和流程图：

1. **QUBO Transformation Flowchart** - QUBO转换流程图
   - 展示从UC问题到QUBO模型的完整转换流程
   - 包含离散化、约束转换、矩阵构建三个关键步骤
   - 清晰展示转换的每个环节

2. **Discretization Principle Diagram** - 离散化原理图
   - 展示连续变量到二进制变量的映射原理
   - 包含二进制编码公式和实际数值示例
   - 清晰展示离散化步长和离散值

3. **Constraint to Penalty Conversion Diagram** - 约束转惩罚项原理图
   - 展示所有约束类型如何转换为惩罚项
   - 包含功率平衡、爬坡、备用、N-1等约束
   - 显示惩罚系数和转换公式

4. **QUBO Matrix Structure Diagram** - QUBO矩阵结构图
   - 展示QUBO矩阵的对称结构
   - 包含矩阵块分解和变量映射
   - 清晰展示矩阵规模和组织方式

5. **Binary Variable Mapping Diagram** - 二进制变量映射图
   - 展示二进制变量的索引映射关系
   - 包含单位-时段-位数的三维映射
   - 提供变量索引计算公式

6. **Objective Function Construction Diagram** - 目标函数构建图
   - 展示目标函数的构建过程
   - 包含二次项、线性项、常数项的组成
   - 显示二进制变量替换过程

7. **Penalty Term Construction Diagram** - 惩罚项构建图
   - 详细展示每种约束的惩罚项构建方法
   - 包含四种主要约束类型的转换
   - 显示惩罚项在QUBO形式中的表达

8. **Solving Process Flowchart** - 求解流程流程图
   - 展示QUBO求解的完整流程
   - 包含矩阵检查、优化器、解码、验证等步骤
   - 清晰展示求解算法流程

9. **Discretization Error Analysis Diagram** - 离散化误差分析图
   - 展示离散化误差的来源和类型
   - 包含误差与位数关系的分析
   - 显示不同精度下的误差对比

10. **Complete System Architecture Diagram** - 完整系统架构图
    - 展示整个QUBO系统的完整架构
    - 包含数据加载、离散化、构建、求解、验证等模块
    - 清晰展示系统各组件的关系

## 使用方法

### 方法1：自动生成（推荐）

运行问题3求解器时会自动生成高级可视化：

```bash
# 运行问题3求解器（会自动生成高级可视化）
python src/problem3/main.py

# 使用2-bit离散化（更高精度）
python src/problem3/main.py --num-bits 2

# 使用Problem 2结果作为参考
python src/problem3/main.py --use-problem2-as-reference
```

### 方法2：从现有结果文件生成

如果已有结果文件，可以直接生成可视化：

```bash
# 为最新结果生成可视化
python src/problem3/advanced_visualizations.py

# 指定结果目录和时间戳
python src/problem3/advanced_visualizations.py \
    --results-dir ../results/problem3 \
    --timestamp 20251124_053435 \
    --problem2-dir ../results/problem2
```

### 输出文件

所有高级可视化图表保存在 `results/problem3/` 目录下：

**3D可视化文件** (文件名格式)：
- `advanced_1_3d_qubo_generation_[timestamp].png`
- `advanced_2_3d_comparison_[timestamp].png`
- `advanced_3_3d_error_surface_[timestamp].png`
- `advanced_4_3d_violations_[timestamp].png`
- `advanced_5_3d_discrete_dist_[timestamp].png`
- `advanced_6_dashboard_[timestamp].png`
- `advanced_7_contour_comparison_[timestamp].png`
- `advanced_8_multi_angle_3d_[timestamp].png`
- `advanced_9_detailed_3d_comparison_[timestamp].png`
- `advanced_10_heatmap_comparison_[timestamp].png`
- `advanced_11_3d_difference_surface_[timestamp].png`
- `advanced_12_unitwise_comparison_[timestamp].png`
- `advanced_13_multimetric_3d_[timestamp].png`

**理论原理图文件** (文件名格式) ⭐ 新增：
- `theoretical_1_qubo_flowchart_[timestamp].png`
- `theoretical_2_discretization_[timestamp].png`
- `theoretical_3_constraint_penalty_[timestamp].png`
- `theoretical_4_qubo_matrix_[timestamp].png`
- `theoretical_5_binary_mapping_[timestamp].png`
- `theoretical_6_objective_function_[timestamp].png`
- `theoretical_7_penalty_terms_[timestamp].png`
- `theoretical_8_solving_process_[timestamp].png`
- `theoretical_9_error_analysis_[timestamp].png`
- `theoretical_10_system_architecture_[timestamp].png`

## 图表特点

### QUBO特有可视化
- **离散值标记**: 在3D图中用红色点标记离散值
- **离散化误差**: 展示量化误差的分布
- **离散值分布**: 展示每个机组使用的离散值频率
- **约束违反**: 展示离散化导致的约束违反

### 对比分析
- **QUBO vs 连续**: 直观对比离散解和连续解
- **成本对比**: 展示离散化对成本的影响
- **误差分析**: 量化离散化误差

### 论文质量
- 高分辨率输出（300 DPI）
- 专业的配色方案
- 清晰的标签和标题
- 多维度数据展示

### 3D可视化
- 可调整视角
- 透明度和阴影效果
- 颜色映射清晰
- 多角度展示

## QUBO特性展示

### 离散化精度
- 1-bit: 每个机组每个时段只有2个离散值（P_min或P_max）
- 2-bit: 每个机组每个时段有4个离散值
- 可视化中清晰展示离散值的分布

### 约束违反
- 功率平衡违反：由于离散化，可能无法精确满足功率平衡
- 可视化中用红色/绿色柱状图展示违反情况
- 3D表面图展示违反的空间分布

### 成本影响
- 离散化可能导致成本增加
- 可视化中对比QUBO成本和连续成本
- 展示量化误差对成本的影响

## 依赖要求

- Python 3.7+
- matplotlib >= 3.0
- numpy >= 1.18
- pandas >= 1.0
- mpl_toolkits (通常随matplotlib安装)

安装依赖：
```bash
pip install matplotlib numpy pandas
```

## 故障排除

如果高级可视化无法生成：
1. 检查matplotlib版本是否支持3D绘图
2. 确保所有依赖已正确安装
3. 查看控制台输出的错误信息
4. 基本结果文件仍会正常保存

如果Problem 2结果不可用：
- 可视化仍会生成，但不会包含对比图表
- 可以稍后添加Problem 2结果目录重新生成

## 自定义

如需自定义可视化：
1. 编辑 `advanced_visualizations.py`
2. 修改颜色方案、图表大小、视角等参数
3. 添加新的可视化类型
4. 调整离散化分析参数

## 论文使用建议

### 推荐使用的图表

**3D可视化图表**：
- **3D QUBO Generation Surface**: 展示离散化后的发电模式
- **QUBO vs Continuous Comparison**: 展示离散化影响
- **Discretization Error Surface**: 量化离散化误差
- **Multi-Panel Dashboard**: 综合QUBO分析

**理论原理图** ⭐ 新增：
- **QUBO Transformation Flowchart**: 展示转换流程（适合方法部分）
- **Discretization Principle Diagram**: 展示离散化原理（适合理论部分）
- **Constraint to Penalty Conversion**: 展示约束转换（适合方法部分）
- **QUBO Matrix Structure**: 展示矩阵结构（适合技术细节）
- **System Architecture Diagram**: 展示系统架构（适合系统设计部分）

### 图表选择
- 3D图表适合展示复杂的三维关系
- 对比图适合展示离散化影响
- 误差图适合量化分析
- 多面板图适合综合展示

### QUBO分析重点
- 离散化精度的影响
- 约束违反情况
- 成本增加程度
- 离散值分布模式

## 示例

### 1-bit离散化
- 每个机组只有2个离散值
- 约束违反可能较大
- 成本可能显著增加

### 2-bit离散化
- 每个机组有4个离散值
- 约束违反较小
- 成本更接近连续解

### 可视化对比
- 1-bit: 离散值分布更集中
- 2-bit: 离散值分布更分散，更接近连续解

