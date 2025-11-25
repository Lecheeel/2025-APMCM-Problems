kaiwu.hobo package
Module contents
模块: hobo

功能: hobo建模工具

class kaiwu.hobo.HoboModel(objective=None)
Bases: BinaryModel

支持添加约束的HOBO模型类

reduce(predefined_pairs=None)
对Hobo Model高阶表达式进行降阶处理（降至二阶）

Args:
predefined_pairs (list): 预定义要合并的变量对列表，格式为[(var1, var2), …]

Returns:
QuboModel: 降阶后的QuboModel

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