kaiwu.conversion package
Module contents
转换工具集合

kaiwu.conversion.ising_matrix_to_qubo_matrix(ising_mat, remove_linear_bit=True)
Ising矩阵转QUBO矩阵

Args:
ising_mat (np.ndarray): Ising矩阵

remove_linear_bit (bool): QUBO转Ising时会增加一个辅助变量表示线性项。是否移除最后一个自旋变量。默认为True。

Returns:
tuple: QUBO矩阵和bias

qubo_mat (np.ndarray): QUBO矩阵

bias (float): QUBO与Ising相差的常数项

Examples:
import numpy as np
import kaiwu as kw
matrix = -np.array([[ 0. ,  1. ,  0. ,  1. ,  1. ],
                    [ 1. ,  0. ,  0. ,  1.,   1. ],
                    [ 0. ,  0. ,  0. ,  1.,   1. ],
                    [ 1. ,  1.,   1. ,  0. ,  1. ],
                    [ 1. ,  1.,   1. ,  1. ,  0. ]])
_qubo_mat, _ = kw.conversion.ising_matrix_to_qubo_matrix(matrix)
_qubo_mat
array([[-4.,  8.,  0.,  8.],
       [-0., -4.,  0.,  8.],
       [-0., -0., -0.,  8.],
       [-0., -0., -0., -8.]])
kaiwu.conversion.qubo_matrix_to_ising_matrix(qubo_mat)
QUBO矩阵转Ising矩阵

Args:
qubo_mat (np.ndarray): QUBO矩阵

Returns:
tuple: Ising矩阵和bias
ising_mat (np.ndarray): Ising矩阵

bias (float): QUBO与Ising相差的常数项

Examples:
import numpy as np
import kaiwu as kw
matrix = -np.array([[-4.,  8.,  0.,  8.],
                    [-0., -4.,  0.,  8.],
                    [-0., -0., -0.,  8.],
                    [-0., -0., -0., -8.]])
_ising_mat, _ = kw.conversion.qubo_matrix_to_ising_matrix(matrix)
_ising_mat
array([[-0.,  1., -0.,  1.,  1.],
       [ 1., -0., -0.,  1.,  1.],
       [-0., -0., -0.,  1.,  1.],
       [ 1.,  1.,  1., -0.,  1.],
       [ 1.,  1.,  1.,  1., -0.]])
kaiwu.conversion.qubo_model_to_ising_model(qubo_model)
QUBO转CIM Ising模型.

Args:
qubo_model (QuboModel): QUBO Model.

Returns:
CimIsing: CIM Ising模型.

Examples:
import kaiwu as kw
b1, b2 = kw.core.Binary("b1"), kw.core.Binary("b2")
q = b1 + b2 + b1*b2
q_model = kw.qubo.QuboModel(q)
ci = kw.conversion.qubo_model_to_ising_model(q_model)
print(str(ci))
CIM Ising Details:
  CIM Ising Matrix:
    [[-0.    -0.125 -0.375]
     [-0.125 -0.    -0.375]
     [-0.375 -0.375 -0.   ]]
  CIM Ising Bias: 1.25
  CIM Ising Variables: b1, b2, __spin__