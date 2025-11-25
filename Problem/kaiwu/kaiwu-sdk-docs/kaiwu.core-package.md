kaiwu.core package
Module contents
模块: core

功能: 基础类的定义

class kaiwu.core.OptimizerBase
Bases: object

Ising求解器基类

set_matrix(ising_matrix)
设置矩阵并更新相关内容

on_matrix_change()
更新矩阵相关信息, 继承OptimizerBase时可以实现。当处理的ising矩阵发生变化时，这个函数的实现会被调用，从而有机会做相应动作

solve(ising_matrix=None)
求解

class kaiwu.core.SolverBase(optimizer)
Bases: object

Solver基类

Args:
optimizer (OptimizerBase): Ising求解器

solve_qubo(*args, **kwargs)
exception kaiwu.core.KaiwuError
Bases: Exception

Base class for exceptions in this module.

args
with_traceback()
Exception.with_traceback(tb) – set self.__traceback__ to tb and return self.

class kaiwu.core.ConstraintDefinition(expr_left, relation, expected_value=0)
Bases: object

约束定义 Args:

expr_left (Expression): 约束项左算子

relation (string): 关系运算符

expected_value(float): 约束项右算子， 缺省为0

is_satisfied(solution_dict)
验证约束满足情况

kaiwu.core.get_min_penalty(obj, cons)
返回约束项cons对应的最小惩罚系数，惩罚项优先满足

Args:
obj: 原目标函数的qubo表达式。

cons：约束项的qubo表达式

Returns:
float: 返回约束项cons对应的最小惩罚系数.

Examples：
import kaiwu as kw
x = [kw.core.Binary("b"+str(i)) for i in range(3)]
cons = kw.core.quicksum(x) - 1
obj = x[1] + 2 * x[2]
kw.core.get_min_penalty(obj, cons)
2.0
kaiwu.core.get_min_penalty_from_min_diff(cons, negative_delta, positive_delta)
根据objective的最大值，最小值估算约束项的最小惩罚系数

Args:
cons: 约束项

negative_delta: objective最小值

positive_delta: objective最大值

Returns:
找到的最小惩罚系数

kaiwu.core.get_min_penalty_for_equal_constraint(obj, cons)
返回一次等式约束项cons对应的最小惩罚系数：把满足这个约束的解的某一位比特翻转一下的最坏情况。 这个惩罚系数有效是指能够保证原问题的可行解是目标函数的局部最优（在一位比特翻转的局部意义下）。

Args：
obj: 原目标函数的qubo表达式。

cons：线性的等式约束cons=0中的线性表达式。

Returns:
float: 一次等式约束项cons对应的最小惩罚系数.

Examples：
import kaiwu as kw
x = [kw.core.Binary("b"+str(i)) for i in range(3)]
cons = kw.core.quicksum(x)-1
obj = x[1]+2*x[2]
kw.core.get_min_penalty_for_equal_constraint(obj,cons)
2.0
kaiwu.core.get_min_penalty_from_deltas(cons, neg_delta, pos_delta, obj_vars, min_delta_method='diff')
返回约束项cons对应的最小惩罚系数，惩罚项优先满足

Args：
cons：约束项的qubo表达式

neg_delta: 各个变量1变为0时最大变化量的dict

pos_delta: 各个变量0变为1时最大变化量的dict

obj_vars: 第三个元素为变量列表

min_delta_method: 声明在。分别用两种方法寻找最小变化值
MIN_DELTA_METHODS = {“diff”: _get_constraint_min_deltas_diff,
“exhaust”: _get_constraint_min_deltas_exhaust}

Examples:
import kaiwu as kw
x = [kw.core.Binary("b"+str(i)) for i in range(3)]
cons = kw.core.quicksum(x) - 1
obj = x[1]+2*x[2]
kw.core.get_min_penalty(obj, cons)
2.0
class kaiwu.core.PenaltyMethodConstraint(expr, penalty=1, parent_model=None)
Bases: object

有约束转无约束的penalty method方法 Args:

expr (Expression): 编译后的约束项表达式

penalty (float): 约束项惩罚系数

classmethod from_constraint_definition(name, constraint: ConstraintDefinition, parent_model)
Prepare QUBO expression for the given constraint, automatically determining slack variables if needed.

Args:
name: Name of the constraint.

constraint: The relation constraint to process.

parent_model: the model it belongs to.

set_penalty(penalty)
设置惩罚系数

penalize_more()
增加惩罚系数

penalize_less()
降低惩罚系数

is_satisfied(solution_dict)
验证约束满足情况

kaiwu.core.get_sol_dict(solution, vars_dict)
根据解向量和变量字典生成结果字典.

Args:
solution (np.ndarray): 解向量（spin）。

vars_dict (dict): 变量字典，用cim_ising_model.get_variables()生成。

Returns:
dict: 结果字典。键为变量名，值为对应的spin值。

Examples:
import numpy as np
import kaiwu as kw
a = kw.core.Binary("a")
b = kw.core.Binary("b")
c = kw.core.Binary("c")
d = a + 2 * b + 4 * c
d = kw.qubo.QuboModel(d)
d_ising = kw.conversion.qubo_model_to_ising_model(d)
vars = d_ising.get_variables()
s = np.array([1, -1, 1])
kw.core.get_sol_dict(s, vars)
{'a': np.float64(1.0), 'b': np.float64(0.0), 'c': np.float64(1.0)}
kaiwu.core.get_array_val(array, sol_dict)
根据结果字典将spin值带入qubo数组变量.

Args:
array (QUBOArray): QUBO数组

sol_dict (dict): 由get_sol_dict生成的结果字典。

Returns:
np.ndarray: 带入qubo数组后所得的值数组

Examples:
import kaiwu as kw
import numpy as np
x = kw.core.ndarray((2, 2), "x", kw.core.Binary)
y = x.sum()
y = kw.qubo.QuboModel(y)
y_ising = kw.conversion.qubo_model_to_ising_model(y)
ising_vars = y_ising.get_variables()
s = np.array([1, -1, 1, -1])
sol_dict = kw.core.get_sol_dict(s, ising_vars)
kw.core.get_array_val(x, sol_dict)
array([[1., 0.],
       [1., 0.]])
kaiwu.core.get_val(qubo, sol_dict)
根据结果字典将spin值带入qubo变量.

Args:
qubo (QUBO表达式): QUBO表达式

sol_dict (dict): 由get_sol_dict生成的结果字典。

Returns:
float: 带入qubo后所得的值

Examples:
import kaiwu as kw
import numpy as np
a = kw.core.Binary("a")
b = kw.core.Binary("b")
c = kw.core.Binary("c")
d = a + 2 * b + 4 * c
qubo_model = kw.qubo.QuboModel(d)
d_ising = kw.conversion.qubo_model_to_ising_model(qubo_model)
ising_vars = d_ising.get_variables()
s = np.array([1, -1, 1])
sol_dict = kw.core.get_sol_dict(s, ising_vars)
kw.core.get_val(d, sol_dict)
np.float64(5.0)
kaiwu.core.update_constraint(qubo_origin, qubo_result)
将qexp的约束以及降阶约束信息更新到q

class kaiwu.core.Expression(coefficient: dict | None = None, offset: float = 0)
Bases: dict

QUBO/Ising 通用表达式基类（提供默认二次表达式实现）

clear() → None.  Remove all items from D.
get_variables()
获取变量名集合

Returns:
variables: (tuple) 返回构成expression的变量集合

get_max_deltas()
求出每个变量翻转引起目标函数变化的上界 返回值negative_delta，positive_delta分别为该变量1->0和0->1所引起的最大变化量

fromkeys(value=None, /)
Create a new dictionary with keys from iterable and values set to value.

get_average_coefficient()
返回coefficient的平均值

kaiwu.core.expr_add(expr_left, expr_right, expr_result)
通用二次表达式相加

kaiwu.core.expr_mul(expr_left, expr_right, expr_result)
通用二次表达式相乘

kaiwu.core.expr_neg(expr_origin, expr_result)
通用二次表达式取负

kaiwu.core.expr_pow(expr_left, expr_right, expr_result)
通用二次表达式乘方，要求expr_right只能为1或者2

class kaiwu.core.BinaryModel(objective=None)
Bases: object

二值模型类

Args:
objective (BinaryExpression, optional): 目标函数. 默认为None

set_constraint_handler(constraint_handler)
设置约束项无约束化方法

Args:
constraint_handler: 设置约束项无约束化表示方法类

set_objective(objective)
设置目标函数

Args:
objective (BinaryExpression): 目标函数表达式

add_constraint(constraint_in, name=None, constr_type: Literal['soft', 'hard'] = 'hard', penalty=None)
添加约束项（支持单个或多个约束）

Args:
constraint_in (ConstraintDefinition or iterable): 约束表达式或其可迭代对象 name (str or list, optional): 约束名称或名称列表，默认自动命名 constr_type (str, optional): 约束类型，可以设置为”soft”或”hard”，默认为”hard” penalty (float): 缺省惩罚系数

get_value(solution_dict)
根据结果字典将变量值带入qubo变量.

Args:
solution_dict (dict): 由get_sol_dict生成的结果字典。

Returns:
float: 带入qubo后所得的值

verify_constraint(solution_dict, constr_type: Literal['soft', 'hard'] = 'hard')
确认约束是否满足

Args:
solution_dict (dict): QUBO模型解字典

constr_type(str, optional): 约束类型，可以设置为”soft”或”hard”，默认为”hard”

Returns:
tuple: 约束满足信息
int: 不满足的约束个数

dict: 包含约束值的字典

initialize_penalties()
自动初始化所有的惩罚系数

get_constraints_expr_list()
获取当前所有的constraint。

Returns:
List of all constraints.

compile_constraints()
按照不同的风格转换约束项为Expression 对于不等式约束，目前支持的是罚函数方式，

class kaiwu.core.BinaryExpression(coefficient: dict | None = None, offset: float = 0, name='')
Bases: Expression

QUBO表达式的基础数据结构

feed(feed_dict)
为占位符号赋值, 并返回赋值后的新表达式对象

Args:
feed_dict(dict): 需要赋值的占位符的值

Examples:
import kaiwu as kw
p = kw.core.Placeholder('p')
a = kw.core.Binary('a')
y = p * a
str(y)  
'(p)*a'
y= y.feed({'p': 2})
str(y)  
'2*a'
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

class kaiwu.core.Binary(name: str = '')
Bases: BinaryExpression

二进制变量, 只保存变量名，不继承 QuboExpression

clear() → None.  Remove all items from D.
feed(feed_dict)
为占位符号赋值, 并返回赋值后的新表达式对象

Args:
feed_dict(dict): 需要赋值的占位符的值

Examples:
import kaiwu as kw
p = kw.core.Placeholder('p')
a = kw.core.Binary('a')
y = p * a
str(y)  
'(p)*a'
y= y.feed({'p': 2})
str(y)  
'2*a'
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

kaiwu.core.quicksum(qubo_expr_list: list)
高性能的QUBO求和器.

Args:
qubo_expr_list (QUBO列表): 用于求和的QUBO表达式的列表.

Returns:
BinaryExpression: 约束QUBO.

Examples:
import kaiwu as kw
qubo_list = [kw.core.Binary("b"+str(i)) for i in range(10)] # Variables are also QUBO
output = kw.core.quicksum(qubo_list)
str(output)
'b0+b1+b2+b3+b4+b5+b6+b7+b8+b9'
class kaiwu.core.Placeholder(name: str = '')
Bases: BinaryExpression

占位符变量, 只保存变量名, 对决策

get_placeholder_set()
获取占位符集合

clear() → None.  Remove all items from D.
feed(feed_dict)
为占位符号赋值, 并返回赋值后的新表达式对象

Args:
feed_dict(dict): 需要赋值的占位符的值

Examples:
import kaiwu as kw
p = kw.core.Placeholder('p')
a = kw.core.Binary('a')
y = p * a
str(y)  
'(p)*a'
y= y.feed({'p': 2})
str(y)  
'2*a'
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

class kaiwu.core.Integer(name: str = '', min_value=0, max_value=127)
Bases: BinaryExpression

整数变量, 只保存变量名和范围，不继承 QuboExpression

clear() → None.  Remove all items from D.
feed(feed_dict)
为占位符号赋值, 并返回赋值后的新表达式对象

Args:
feed_dict(dict): 需要赋值的占位符的值

Examples:
import kaiwu as kw
p = kw.core.Placeholder('p')
a = kw.core.Binary('a')
y = p * a
str(y)  
'(p)*a'
y= y.feed({'p': 2})
str(y)  
'2*a'
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

kaiwu.core.ndarray(shape: int | Tuple[int, ...] | List[int], name, var_func, var_func_param=None)
基于 np.ndarray 的QUBO容器. 该容器支持各种 numpy 原生的向量化运算

Args:
shape (Union[int, Tuple[int, …]]): 形状

name (str): 生成的变量的标识符.

var_func (class for func): 用于生成元素的方法或类. 第一个参数必须是name

var_func_param (tuple): var_func除了name以外的参数

Returns:
np.ndarray: 多维容器.

Examples:
import numpy as np
import kaiwu as kw
A = kw.core.ndarray((2,3,4), "A", kw.core.Binary)
A
BinaryExpressionNDArray([[[A[0][0][0], A[0][0][1], A[0][0][2],
                           A[0][0][3]],
                          [A[0][1][0], A[0][1][1], A[0][1][2],
                           A[0][1][3]],
                          [A[0][2][0], A[0][2][1], A[0][2][2],
                           A[0][2][3]]],

                         [[A[1][0][0], A[1][0][1], A[1][0][2],
                           A[1][0][3]],
                          [A[1][1][0], A[1][1][1], A[1][1][2],
                           A[1][1][3]],
                          [A[1][2][0], A[1][2][1], A[1][2][2],
                           A[1][2][3]]]], dtype=object)
A[1,2]
BinaryExpressionNDArray([A[1][2][0], A[1][2][1], A[1][2][2], A[1][2][3]],
                        dtype=object)
A[:, [0,2]]
BinaryExpressionNDArray([[[A[0][0][0], A[0][0][1], A[0][0][2],
                           A[0][0][3]],
                          [A[0][2][0], A[0][2][1], A[0][2][2],
                           A[0][2][3]]],

                         [[A[1][0][0], A[1][0][1], A[1][0][2],
                           A[1][0][3]],
                          [A[1][2][0], A[1][2][1], A[1][2][2],
                           A[1][2][3]]]], dtype=object)
B = kw.core.ndarray(3, "B", kw.core.Binary)
B
BinaryExpressionNDArray([B[0], B[1], B[2]], dtype=object)
C = kw.core.ndarray([3,3], "C", kw.core.Binary)
C
BinaryExpressionNDArray([[C[0][0], C[0][1], C[0][2]],
                         [C[1][0], C[1][1], C[1][2]],
                         [C[2][0], C[2][1], C[2][2]]], dtype=object)
D = 2 * B.dot(C) + 2
str(D[0])
'2*B[0]*C[0][0]+2*B[1]*C[1][0]+2*B[2]*C[2][0]+2'
E = B.sum()
str(E)
'B[0]+B[1]+B[2]'
F = np.diag(C)
F
BinaryExpressionNDArray([C[0][0], C[1][1], C[2][2]], dtype=object)
kaiwu.core.dot(mat_left, mat_right)
矩阵乘法

Args:
mat_left (numpy.array): 矩阵1

mat_right (numpy.array): 矩阵2

Raises:
ValueError: 两个输入都必须是np.ndarray ValueError: 两个输入的维度必须匹配

Returns:
np.ndarray: 乘积矩阵

class kaiwu.core.BinaryExpressionNDArray
Bases: ndarray

基于 np.ndarray 的QUBO容器. 该容器支持各种 numpy 原生的向量化运算

is_array_less = <numpy.vectorize object>
is_array_less_equal = <numpy.vectorize object>
is_array_greater = <numpy.vectorize object>
is_array_greater_equal = <numpy.vectorize object>
is_array_equal = <numpy.vectorize object>
dot(b, out=None)
使用quicksum的矩阵乘法

Args:
b (BinaryExpressionNDArray): 另一个矩阵 out：可选输出数组，用于存储结果。需与预期输出形状一致。

Returns:
BinaryExpressionNDArray: 乘积

sum(axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True)
使用quicksum的求和方法

Args:
axis：指定求和的轴（维度）。默认为 None，表示对所有元素求和；若为整数或元组，则沿指定轴求和。 dtype：指定输出数据类型。若未提供，则默认使用输入数组的 dtype，但整数类型可能提升为平台整数精度。暂不支持。 out：可选输出数组，用于存储结果。需与预期输出形状一致。 keepdims：布尔值。若为 True，则保留被求和的轴作为长度为1的维度。暂不支持 。 initial：求和的初始值（标量），默认为0。暂不支持。 where：布尔数组，指定哪些元素参与求和（NumPy 1.20+支持）。暂不支持。

Returns:
BinaryExpressionNDArray: 乘积