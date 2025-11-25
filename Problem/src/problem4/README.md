# Problem 4: Problem Scale Reduction under Quantum Hardware Constraints

## 概述

Problem 4 实现了有效的问题规模缩减策略，确保在CIM位容量限制下能成功求解QUBO模型。

## 关键设计原则

基于Problem 1和Problem 2的结论：
- **所有机组在所有时段保持在线（固定状态）**，这是全局最优解
- 因此只需要优化发电量 `p_{i,t}`，不需要优化机组状态 `u_{i,t}`
- 这显著减少了问题规模

## 缩减策略

### 1. 时间聚合（Time Aggregation）
- **uniform**: 均匀聚合，将24个时段均匀分组
- **peak_valley**: 峰谷聚合，选择负荷峰值和谷值时段

### 2. 机组选择（Unit Selection）
- 优先选择容量大的机组
- 可根据需要减少机组数量

### 3. 离散化精度调整
- 默认使用1-bit离散化（每个机组每个时段1个二进制变量）
- 可选2-bit离散化（精度更高但变量数翻倍）

## 模块结构

```
problem4/
├── data_loader.py          # 数据加载模块
├── reduction_strategy.py   # 缩减策略模块
├── discretization.py       # 离散化模块
├── qubo_builder.py         # QUBO构建模块
├── solver.py              # 求解器模块
├── verifier.py            # 验证器模块
└── main.py                # 主程序
```

## 使用方法

### 基本用法

```bash
python src/problem4/main.py
```

### 高级选项

```bash
python src/problem4/main.py \
    --max-binary-vars 200 \
    --num-bits 1 \
    --time-aggregation uniform \
    --optimizer simulated_annealing \
    --results-dir ../results/problem2 \
    --output-dir ../results/problem4
```

### 参数说明

- `--max-binary-vars`: 最大二进制变量数（默认：200，CIM限制）
- `--num-bits`: 每个机组每个时段的位数（1或2，默认：1）
- `--time-aggregation`: 时间聚合方法（uniform/peak_valley，默认：uniform）
- `--optimizer`: 求解器类型（simulated_annealing/cim，默认：simulated_annealing）
- `--results-dir`: Problem 2结果目录（用于对比）
- `--output-dir`: 输出目录
- `--num-solutions`: 求解的解数量（默认：3）

## 输出文件

运行后会生成以下文件：

1. `reduced_generation_schedule_*.csv`: 缩减后问题的发电计划
2. `full_generation_schedule_*.csv`: 扩展到全规模（24时段，6机组）的发电计划
3. `summary_*.json`: 完整的优化结果摘要

## 缩减效果示例

对于原始问题（6机组×24时段×2位=288变量）：
- 使用1-bit离散化：144变量
- 使用时间聚合（12时段）：72变量
- 使用时间聚合+机组选择（12时段×4机组）：48变量

## 约束处理

缩减后的QUBO模型包含以下约束（转换为惩罚项）：
1. **功率平衡约束**: 确保每个时段发电量等于负荷需求
2. **爬坡约束**: 限制机组出力变化速率
3. **旋转备用约束**: 确保足够的备用容量
4. **N-1安全约束**: 确保任意单台机组故障时系统仍可行

## 注意事项

1. **CIM硬件限制**: 8-bit INT空间[-128, 127]，QUBO矩阵会自动调整精度
2. **缩减损失**: 缩减会带来一定的精度损失，但能确保在硬件限制下求解
3. **解扩展**: 缩减问题的解会自动扩展到全规模（24时段，6机组）

## 与Problem 3的对比

| 特性 | Problem 3 | Problem 4 |
|------|-----------|-----------|
| 二进制变量数 | 288 (6×24×2) | 可配置（≤200） |
| 离散化精度 | 2-bit | 1-bit或2-bit |
| 时间聚合 | 否 | 是 |
| 机组选择 | 全部 | 可选择性 |
| CIM兼容性 | 可能超出限制 | 确保兼容 |

## Kaiwu SDK 文档

**官方文档（最新版本）：**
- 英文：https://kaiwu-sdk-docs.qboson.com/en/latest/
- 中文：https://kaiwu-sdk-docs.qboson.com/zh/v1.2.0/source/getting_started/index.html

**本地文档：**
- 详细使用说明：`KAIWU_SDK_USAGE.md`
- SDK 包文档：`../kaiwu/kaiwu-sdk-docs/`

## 参考文献

- Problem 1和Problem 2的结论：所有机组在线是最优解
- Problem 3的QUBO构建方法
- Kaiwu SDK文档：CIM硬件限制和精度要求

