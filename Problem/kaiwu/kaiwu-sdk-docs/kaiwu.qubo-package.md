kaiwu.qubo package
Module contents
模块: qubo

功能: QUBO建模工具

kaiwu.qubo.check_qubo_matrix_bit_width(qubo_matrix, bit_width=8)
校验QUBO矩阵元素位宽

将QUBO矩阵转为伊辛矩阵，通过校验伊辛矩阵的元素位宽来实现对QUBO矩阵的校验

Args:
qubo_matrix (np.ndarray): QUBO矩阵

bit_width (int): 位宽

Raises:
ValueError: 当矩阵元素位宽超过指定位宽时抛出异常

Examples1:
import numpy as np
import kaiwu as kw
_matrix = -np.array([[-480., 508., -48.],
                     [ 508., -508., -48.],
                     [ -48., -48., 60.]])
kw.qubo.check_qubo_matrix_bit_width(_matrix)
Examples2（缩放后符合要求）:
import numpy as np
import kaiwu as kw
_matrix = -np.array([[-512.,  520.,  -48.],
                     [ 520., -520.,  -48.],
                     [ -48.,  -48.,   40.]])
kw.qubo.check_qubo_matrix_bit_width(_matrix)
Examples3(缩放后也不符合要求):
import numpy as np
import kaiwu as kw
_matrix = -np.array([[-488.,  516.,  -48.],
                     [ 516., -516.,  -48.],
                     [ -48.,  -48.,   60.]])
kw.qubo.check_qubo_matrix_bit_width(_matrix)
Traceback (most recent call last):
...
ValueError: CIM only supports signed 8-bit number
kaiwu.qubo.adjust_qubo_matrix_precision(qubo_matrix, bit_width=8)
调整矩阵精度, 通过此接口调整后矩阵可能会有较大的精度损失，比如矩阵有一个数远大于其它数时，调整后矩阵精度损失严重无法使用

Args:
qubo_matrix (np.ndarray): 目标矩阵

bit_width (int): 精度范围，目前只支持8位，有一位是符号位

Returns:
np.ndarray: 符合精度要求的QUBO矩阵

Examples:
import numpy as np
import kaiwu as kw
ori_qubo_mat1 = np.array([[0.89, 0.22, 0.198],
                     [0.22, 0.23, 0.197],
                     [0.198, 0.197, 0.198]])
qubo_mat1 = kw.qubo.adjust_qubo_matrix_precision(ori_qubo_mat1)
qubo_mat1
array([[348., 168., 152.],
       [ -0.,  92., 152.],
       [ -0.,  -0.,  80.]])
ori_qubo_mat2 = np.array([[0.89, 0.22, 0.198],
                          [0.22, 0.23, 0.197],
                          [0.198, 0.197, 100]])
qubo_mat2 = kw.qubo.adjust_qubo_matrix_precision(ori_qubo_mat2)
qubo_mat2  # The solutions obtained by qubo_mat2 and ori_qubo_mat2 matrices are quite different
array([[  8.,  -0.,  -0.],
       [ -0.,   4.,  -0.],
       [ -0.,  -0., 508.]])
class kaiwu.qubo.QuboModel(objective=None)
Bases: BinaryModel

支持添加约束的QUBO模型类

Args:
objective (QuboExpression, optional): 目标函数. 默认为None

invalidate_made_state()
Invalidate the made state when the model changes

make()
返回合并后的QUBO表达式 Returns:

BinaryExpression: 合并的约束表达式

get_matrix()
获取QUBO矩阵

Returns:
numpy.ndarray: QUBO矩阵

get_variables()
获取qubo模型的variables

get_offset()
获取qubo模型的offset

get_sol_dict(qubo_solution)
根据解向量生成结果字典.

add_constraint(constraint_in, name=None, constr_type: Literal['soft', 'hard'] = 'hard', penalty=None)
添加约束项（支持单个或多个约束）

Args:
constraint_in (ConstraintDefinition or iterable): 约束表达式或其可迭代对象 name (str or list, optional): 约束名称或名称列表，默认自动命名 constr_type (str, optional): 约束类型，可以设置为”soft”或”hard”，默认为”hard” penalty (float): 缺省惩罚系数

compile_constraints()
按照不同的风格转换约束项为Expression 对于不等式约束，目前支持的是罚函数方式，

get_constraints_expr_list()
获取当前所有的constraint。

Returns:
List of all constraints.

get_value(solution_dict)
根据结果字典将变量值带入qubo变量.

Args:
solution_dict (dict): 由get_sol_dict生成的结果字典。

Returns:
float: 带入qubo后所得的值

initialize_penalties()
自动初始化所有的惩罚系数

set_constraint_handler(constraint_handler)
设置约束项无约束化方法

Args:
constraint_handler: 设置约束项无约束化表示方法类

set_objective(objective)
设置目标函数

Args:
objective (BinaryExpression): 目标函数表达式

verify_constraint(solution_dict, constr_type: Literal['soft', 'hard'] = 'hard')
确认约束是否满足

Args:
solution_dict (dict): QUBO模型解字典

constr_type(str, optional): 约束类型，可以设置为”soft”或”hard”，默认为”hard”

Returns:
tuple: 约束满足信息
int: 不满足的约束个数

dict: 包含约束值的字典

kaiwu.qubo.calculate_qubo_value(qubo_matrix, offset, binary_configuration)
Q值计算器.

Args:
qubo_matrix (np.ndarray): QUBO矩阵.

offset (float): 常数项

binary_configuration (np.ndarray): 二进制配置

Returns:
output (float): Q值.

Examples:
import numpy as np
import kaiwu as kw
matrix = np.array([[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]])
offset = 1.8
binary_configuration = np.array([0, 1, 0])
qubo_value = kw.qubo.calculate_qubo_value(matrix, offset, binary_configuration)
print(qubo_value)
2.8
kaiwu.qubo.qubo_matrix_to_qubo_model(qubo_mat)
将qubo矩阵转化为qubo模型

Args:
qubo_mat (np.ndarray): QUBO矩阵

Returns:
QuboModel: QUBO模型

Examples:
import numpy as np
import kaiwu as kw
matrix = -np.array([[0, 8],
                    [0, 0]])
kw.qubo.qubo_matrix_to_qubo_model(matrix).objective
-8*b[0]*b[1]
exception kaiwu.qubo.QuboError(error_info)
Bases: KaiwuError

Exceptions in qubo module.

args
with_traceback()
Exception.with_traceback(tb) – set self.__traceback__ to tb and return self.