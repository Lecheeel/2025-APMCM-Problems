# Kaiwu SDK 使用说明

## 官方文档

**Kaiwu SDK 官方文档（最新版本）：**
- 英文：https://kaiwu-sdk-docs.qboson.com/en/latest/
- 中文：https://kaiwu-sdk-docs.qboson.com/zh/v1.2.0/source/getting_started/index.html

## 安装

### 方法1：从 Wheel 文件安装

```bash
pip install kaiwu/kaiwu-1.3.0-cp310-none-win_amd64.whl
```

**注意：** Wheel 文件是为 Python 3.10 (cp310) 编译的。如果使用 Python 3.12，可能需要：
- 使用 Python 3.10 环境
- 或者从官方源安装最新版本（如果支持 Python 3.12）

### 方法2：从官方源安装

```bash
pip install kaiwu
```

## QUBO 矩阵 8-bit 精度调整

根据 Kaiwu SDK 文档，CIM 硬件要求 QUBO 矩阵元素必须在 8-bit 有符号整数范围 `[-128, 127]` 内。

### 使用 Kaiwu SDK 官方方法

```python
import kaiwu as kw
import numpy as np

# 构建 QUBO 模型
qubo_model = kw.qubo.QuboModel()
# ... 添加约束和目标函数 ...

# 提取矩阵
qubo_matrix = qubo_model.get_matrix()

# 检查 8-bit 精度
try:
    kw.qubo.check_qubo_matrix_bit_width(qubo_matrix, bit_width=8)
    print("✓ Matrix passes 8-bit check")
except ValueError as e:
    print(f"✗ Matrix fails 8-bit check: {e}")
    
    # 使用官方方法调整精度
    adjusted_matrix = kw.qubo.adjust_qubo_matrix_precision(
        qubo_matrix, 
        bit_width=8
    )
    
    # 再次验证
    kw.qubo.check_qubo_matrix_bit_width(adjusted_matrix, bit_width=8)
    print("✓ Adjusted matrix passes 8-bit check")
```

### 相关函数

根据文档，Kaiwu SDK 提供以下函数：

1. **`kw.qubo.check_qubo_matrix_bit_width(matrix, bit_width=8)`**
   - 检查 QUBO 矩阵是否符合指定的位宽要求
   - 如果不符合，抛出 `ValueError`

2. **`kw.qubo.adjust_qubo_matrix_precision(matrix, bit_width=8)`**
   - 自动调整 QUBO 矩阵精度以符合位宽要求
   - 返回调整后的矩阵

3. **`kw.preprocess.PrecisionReducer(optimizer, bit_width=8)`**
   - 作为优化器的预处理步骤，自动处理精度问题

## 脚本说明

### `generate_qubo_with_kaiwu.py`

使用 Kaiwu SDK 官方方法生成和调整 QUBO 矩阵的脚本。

**使用方法：**

```bash
python src/problem4/generate_qubo_with_kaiwu.py \
    --max-binary-vars 100 \
    --num-bits 1 \
    --target-max 80
```

**参数：**
- `--max-binary-vars`: 最大二进制变量数（默认：100）
- `--num-bits`: 每个机组每个时段的位数（默认：1）
- `--time-aggregation`: 时间聚合方法（可选：uniform, peak_valley）
- `--target-max`: 目标最大绝对值（默认：80）
- `--output-file`: 输出文件路径（可选）

### `generate_qubo_matrix_8bit.py`

独立计算 QUBO 矩阵并应用 8-bit 精度调整的脚本（不依赖 Kaiwu SDK 的矩阵构建）。

### `force_8bit_range.py`

对现有 QUBO 矩阵进行后处理，强制缩放到 8-bit 范围。

## 硬件约束

根据 CIM 硬件限制：

1. **最大二进制变量数：** 100
2. **QUBO 矩阵精度：** 8-bit 有符号整数 `[-128, 127]`
3. **矩阵对称性：** QUBO 矩阵必须是对称的

## 参考

- [Kaiwu SDK 官方文档](https://kaiwu-sdk-docs.qboson.com/en/latest/)
- [QUBO 包文档](kaiwu/kaiwu-sdk-docs/kaiwu.qubo-package.md)
- [CIM 包文档](kaiwu/kaiwu-sdk-docs/kaiwu.cim-package.md)

