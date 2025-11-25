# Advanced Visualizations for Problem 2

## 概述

本模块为问题2生成高质量的可视化图表，包括3D图表和专业的2D图表，特别针对网络约束和安全性分析，适合用于学术论文。

## 功能特性

### 自动生成的高级图表

运行 `uc_network_security.py` 后，会自动生成以下高级可视化：

1. **3D Generation Surface with Reserve** - 带旋转备用的3D发电量表面图
   - 展示发电量分布，叠加旋转备用容量
   - 红色半透明层表示备用容量

2. **3D Line Flow Surface** - 3D线路潮流表面图
   - 展示主要输电线路的潮流利用率
   - 显示线路容量使用情况

3. **3D Bus Voltage Angle Surface** - 3D节点电压角表面图
   - 展示关键节点的电压角变化
   - 基于DC潮流计算结果

4. **3D Spinning Reserve Landscape** - 3D旋转备用景观图
   - 展示可用备用vs需求备用
   - 线框表示需求备用

5. **Multi-Panel Network Dashboard** - 多面板网络分析仪表板
   - 线路潮流热力图
   - 节点电压角热力图
   - 旋转备用分析
   - 发电vs负荷对比

6. **3D Cost Analysis with Network** - 带网络约束的3D成本分析
   - 成本-发电量-时间关系
   - 颜色编码表示备用可用性

7. **Contour with Network Overlay** - 带网络叠加的等高线图
   - 发电量等高线图
   - 旋转备用等高线图

8. **3D Multi-Metric Dashboard** - 3D多指标仪表板
   - 四个3D子图展示不同指标
   - 发电量、备用、潮流、电压角

## 使用方法

### 基本使用

```bash
# 运行问题2求解器（会自动生成高级可视化）
python src/problem2/uc_network_security.py

# 启用惯性约束
python src/problem2/uc_network_security.py --enable-inertia

# 使用放松的N-1约束（调试用）
python src/problem2/uc_network_security.py --relax-n1
```

### 输出文件

所有高级可视化图表保存在 `results/problem2/` 目录下，文件名格式：
- `advanced_1_3d_generation_reserve_[timestamp].png`
- `advanced_2_3d_line_flow_[timestamp].png`
- `advanced_3_3d_bus_angles_[timestamp].png`
- `advanced_4_3d_reserve_landscape_[timestamp].png`
- `advanced_5_network_dashboard_[timestamp].png`
- `advanced_6_3d_cost_network_[timestamp].png`
- `advanced_7_contour_network_[timestamp].png`
- `advanced_8_3d_dashboard_[timestamp].png`

## 图表特点

### 网络分析专用
- **线路潮流可视化**: 展示输电线路的功率流动
- **节点电压角**: DC潮流分析结果可视化
- **旋转备用**: 安全性指标的可视化
- **N-1安全性**: 网络约束的图形化展示

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

## 网络特定可视化

### 线路潮流分析
- 显示主要输电线路的利用率
- 识别瓶颈线路
- 展示潮流随时间的变化

### 节点分析
- 电压角分布
- 关键节点识别
- 网络拓扑可视化

### 安全性分析
- 旋转备用可用性
- N-1约束满足情况
- 网络冗余度展示

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
4. 调整网络分析参数

## 论文使用建议

### 推荐使用的图表
- **3D Line Flow Surface**: 展示网络潮流分布
- **Multi-Panel Dashboard**: 综合网络分析
- **3D Reserve Landscape**: 安全性分析
- **Contour with Network**: 清晰的2D网络分析

### 图表选择
- 3D图表适合展示复杂的三维关系
- 多面板图适合综合展示多个指标
- 热力图适合展示时间序列模式
- 等高线图适合展示连续分布

### 网络分析重点
- 线路利用率分析
- 节点电压角变化
- 旋转备用充足性
- 网络安全性评估

