kaiwu.ising package
Module contents
模块: Ising

功能: 提供Ising模型相关函数

class kaiwu.ising.IsingModel(variables, ising_matrix, bias)
Bases: dict

ising模型

get_variables()
获取模型中的变量

get_matrix()
获取Ising矩阵

get_bias()
获取QUBO转化时得到的常数偏置

fromkeys(value=None, /)
Create a new dictionary with keys from iterable and values set to value.

class kaiwu.ising.IsingExpression(variables=None, quadratic=None, linear=None, bias=0)
Bases: Expression

Ising 表达式基类，直接继承 Expression，保留扩展点。

clear() → None.  Remove all items from D.
fromkeys(value=None, /)
Create a new dictionary with keys from iterable and values set to value.

get_average_coefficient()
返回coefficient的平均值

get_max_deltas()
求出每个变量翻转引起目标函数变化的上界 返回值negative_delta，positive_delta分别为该变量1->0和0->1所引起的最大变化量

get_variables()
获取变量名集合

Returns:
variables: (tuple) 返回构成expression的变量集合

kaiwu.ising.calculate_ising_matrix_bit_width(ising_matrix, bit_width=8)
计算 ising 矩阵的参数位宽

Args:
ising_matrix (np.ndarray): ising 矩阵

bit_width (int): 最大位宽限制

Returns:
dict: 返回Ising矩阵的精度和缩放因子

precision (int): Ising矩阵精度

multiplier (float): 缩放因子

示例1:
import numpy as np
import kaiwu as kw
_matrix = -np.array([[ -0., 127., -12.,  -5.],
                     [127.,  -0., -12., -12.],
                     [-12., -12.,  -0.,  -9.],
                     [ -5., -12.,  -9.,  -0.]])
kw.ising.calculate_ising_matrix_bit_width(_matrix)
{'precision': 8, 'multiplier': np.float64(1.0)}
示例2（缩放后符合要求）:
import numpy as np
import kaiwu as kw
_matrix = -np.array([[ -0., 12.7, -1.2,  -0.5],
                     [12.7,  -0., -1.2, -1.2],
                     [-1.2, -1.2, -0.,   -0.9],
                     [-0.5, -1.2, -0.9,  -0.]])
kw.ising.calculate_ising_matrix_bit_width(_matrix)
{'precision': 8, 'multiplier': np.float64(10.0)}
示例3(缩放后也不符合要求):
import numpy as np
import kaiwu as kw
_matrix = -np.array([[-488.,  516.,  -48.],
                     [ 516., -516.,  -48.],
                     [ -48.,  -48.,   60.]])
kw.ising.calculate_ising_matrix_bit_width(_matrix)
{'precision': inf, 'multiplier': inf}
kaiwu.ising.adjust_ising_matrix_precision(ising_matrix, bit_width=8)
调整 ising 矩阵精度, 通过此接口调整后矩阵可能会有较大的精度损失，比如矩阵有一个数远大于其它数时，调整后矩阵精度损失严重无法使用

Args:
ising_matrix(np.ndarray): 目标矩阵

bit_width(int): 精度范围，目前只支持8位，有一位是符号位

Returns:
np.ndarray: 符合精度要求的 ising 矩阵

Examples:
import numpy as np
import kaiwu as kw
ori_ising_mat1 = np.array([[0, 0.22, 0.198],
                           [0.22, 0, 0.197],
                           [0.198, 0.197, 0]])
ising_mat1 = kw.ising.adjust_ising_matrix_precision(ori_ising_mat1)
ising_mat1
array([[  0, 127, 114],
       [127,   0, 114],
       [114, 114,   0]])
ori_ising_mat2 = np.array([[0, 0.22, 0.198],
                           [0.22, 0, 50],
                           [0.198, 50, 0]])
ising_mat2 = kw.ising.adjust_ising_matrix_precision(ori_ising_mat2)
ising_mat2  # The solutions obtained by qubo_mat2 and ori_qubo_mat2 matrices are quite different
array([[  0,   1,   1],
       [  1,   0, 127],
       [  1, 127,   0]])
class kaiwu.ising.Spin(name: str = '')
Bases: IsingExpression

自旋变量, 可能的取值只有-1,1.

Args:
name (str): 变量的唯一标识.

Returns:
dict: 名称为name的自旋变量.

Examples:
import kaiwu as kw
s = kw.ising.Spin("s")
s
2*s-1
clear() → None.  Remove all items from D.
fromkeys(value=None, /)
Create a new dictionary with keys from iterable and values set to value.

get_average_coefficient()
返回coefficient的平均值

get_max_deltas()
求出每个变量翻转引起目标函数变化的上界 返回值negative_delta，positive_delta分别为该变量1->0和0->1所引起的最大变化量

get_variables()
获取变量名集合

Returns:
variables: (tuple) 返回构成expression的变量集合