# Advanced Visualizations for Problem 1

## 概述

本模块为问题1生成高质量的可视化图表，包括3D图表和专业的2D图表，适合用于学术论文。

## 功能特性

### 自动生成的高级图表

运行 `uc_gurobi.py` 后，会自动生成以下高级可视化：

1. **3D Generation Surface** - 3D发电量表面图
   - 展示各机组在不同时段的发电量分布
   - 使用viridis配色方案，适合论文发表

2. **3D Unit Commitment Bar Chart** - 3D机组组合柱状图
   - 三维柱状图展示机组启停状态
   - 绿色表示运行，红色表示停机

3. **3D Cost-Generation-Time Scatter** - 3D成本-发电量-时间散点图
   - 展示成本、发电量和时间的三维关系
   - 不同颜色代表不同机组

4. **Generation Contour Plot** - 发电量等高线图
   - 等高线图展示发电量分布
   - 叠加负荷需求曲线

5. **3D Cost Surface** - 3D成本表面图
   - 展示燃料成本在机组和时间维度的分布

6. **Multi-Angle 3D View** - 多角度3D视图
   - 四个不同视角的3D发电量景观图
   - 包括等轴测图、俯视图、正视图、侧视图

7. **Enhanced Heatmap** - 增强型热力图
   - 发电量热力图叠加负荷曲线
   - 显示关键时段的数值标注

8. **3D Cost Breakdown** - 3D成本分解图
   - 三维柱状图展示成本构成

## 使用方法

### 基本使用

```bash
# 运行问题1求解器（会自动生成高级可视化）
python src/problem1/uc_gurobi.py

# 或指定最小运行时间表
python src/problem1/uc_gurobi.py --min-up-time-table 2
```

### 输出文件

所有高级可视化图表保存在 `results/problem1/` 目录下，文件名格式：
- `advanced_1_3d_generation_surface_[timestamp].png`
- `advanced_2_3d_unit_commitment_[timestamp].png`
- `advanced_3_3d_cost_scatter_[timestamp].png`
- `advanced_4_contour_generation_[timestamp].png`
- `advanced_5_3d_cost_surface_[timestamp].png`
- `advanced_6_multi_angle_3d_[timestamp].png`
- `advanced_7_enhanced_heatmap_[timestamp].png`
- `advanced_8_3d_cost_breakdown_[timestamp].png`

## 图表特点

### 论文质量
- 高分辨率输出（300 DPI）
- 专业的配色方案
- 清晰的标签和标题
- 适合黑白打印的配色选项

### 3D可视化
- 可调整视角
- 透明度和阴影效果
- 颜色映射清晰
- 多角度展示

### 数据展示
- 关键数值标注
- 等高线和轮廓线
- 叠加多个数据维度
- 交互式友好的布局

## 依赖要求

- Python 3.7+
- matplotlib >= 3.0
- numpy >= 1.18
- mpl_toolkits (通常随matplotlib安装)

## 故障排除

如果高级可视化无法生成：
1. 检查matplotlib版本是否支持3D绘图
2. 确保所有依赖已正确安装
3. 查看控制台输出的错误信息
4. 基本可视化仍会正常生成

## 自定义

如需自定义可视化：
1. 编辑 `advanced_visualizations.py`
2. 修改颜色方案、图表大小、视角等参数
3. 添加新的可视化类型

## 论文使用建议

### 推荐使用的图表
- **3D Generation Surface**: 展示整体发电模式
- **Multi-Angle 3D View**: 多角度展示数据
- **Contour Plot**: 清晰的2D等高线图
- **Enhanced Heatmap**: 热力图适合展示时间序列数据

### 图表选择
- 3D图表适合展示复杂的三维关系
- 等高线图适合展示连续分布
- 热力图适合展示时间序列模式
- 多面板图适合综合展示

