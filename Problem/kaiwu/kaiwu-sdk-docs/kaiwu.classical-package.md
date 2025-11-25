kaiwu.classical package
Module contents
模块: classical

功能: 提供一系列经典求解器

class kaiwu.classical.SimulatedAnnealingOptimizer(initial_temperature=100, alpha=0.99, cutoff_temperature=0.001, iterations_per_t=10, size_limit=100, flag_evolution_history=False, verbose=False, rand_seed=None, process_num=1)
Bases: OptimizerBase, JsonSerializableMixin

求解CIM Ising模型的模拟退火求解器（输出结果具有随机性）.

Args:
initial_temperature (float): 初始温度.

alpha (float): 降温系数.

cutoff_temperature (float): 截止温度.

iterations_per_t (int): 每个温度迭代深度.

size_limit (int): 输出解的个数，默认输出100个解

flag_evolution_history (bool): 是否输出哈密顿量演化历史，默认False，当值为True时，通过get_ha_history方法获取演化历史

verbose (bool): 是否在控制台输出计算进度，默认False

rand_seed (int, optional): numpy随机数生成器的随机种子

process_num (int, optional): 并行进程数 (-1为自动调用所有可用核心，1为单进程). Defaults to 1.

Examples:
import numpy as np
import kaiwu as kw
matrix = -np.array([[ 0. ,  1. ,  0. ,  1. ,  1. ],
                    [ 1. ,  0. ,  0. ,  1.,   1. ],
                    [ 0. ,  0. ,  0. ,  1.,   1. ],
                    [ 1. ,  1.,   1. ,  0. ,  1. ],
                    [ 1. ,  1.,   1. ,  1. ,  0. ]])
worker = kw.classical.SimulatedAnnealingOptimizer(initial_temperature=100,
                                                  alpha=0.99,
                                                  cutoff_temperature=0.001,
                                                  iterations_per_t=10,
                                                  size_limit=10)
# This value is random and cannot be predicted
worker.solve(matrix) 
array([[-1,  1, -1,  1,  1],
       [-1,  1,  1, -1,  1],
       [-1,  1,  1, -1, -1],
       [ 1, -1, -1,  1,  1],
       [ 1, -1,  1, -1,  1],
       [ 1, -1,  1, -1, -1],
       [ 1, -1, -1, -1,  1],
       [ 1, -1,  1,  1, -1],
       [ 1,  1,  1, -1, -1],
       [-1, -1, -1,  1,  1]])
on_matrix_change()
更新矩阵相关信息

get_ha_history()
获取哈密顿量演化历史

Returns:
dict: 哈密顿量随时间演化历史，key是时间，单位为秒，value是 hamilton

single_process_solve(ising_matrix=None, init_solution=None, rand_seed=None)
单进程求解Ising矩阵

Args:
ising_matrix (np.ndarray, optional): Ising矩阵. Defaults to None.

init_solution (np.ndarray, optional): 初始解向量. Defaults to None.

rand_seed (int, optional): numpy随机数生成器的随机种子

Returns:
np.ndarray: 解向量

solve(ising_matrix=None, init_solution=None)
SA求解Ising矩阵solve接口

Args:
ising_matrix (np.ndarray, optional): Ising矩阵. Defaults to None.

init_solution (np.ndarray, optional): 初始解向量. Defaults to None.

Returns:
np.ndarray: 多个进程合并去重之后的解向量.

load_json_dict(json_dict)
从json文件读取的dict恢复对象

Returns:
dict: json字典

set_matrix(ising_matrix)
设置矩阵并更新相关内容

to_json_dict(exclude_fields=('_optimizer',))
转化为json字典

Returns:
dict: json字典

class kaiwu.classical.TabuSearchOptimizer(max_iter, recency_size=None, kmax=3, span_control_p1=3, span_control_p2=7, size_limit=1)
Bases: OptimizerBase

求解CIM Ising模型的禁忌搜索求解器.

Args:
max_iter (int): 最大迭代次数

recency_size (int): recency禁忌表的大小。如输入为空，则使用矩阵边长的1/10向上取整。

kmax (int): 模型参数变量k的最大值。默认值为3。

span_control_p1 (int): 影响span变化的参数p1.默认值为3。

span_control_p2 (int): 影响span变化的参数p2.默认值为7。

size_limit (int): 维护解集的大小

Examples:
import numpy as np
import kaiwu as kw
matrix = -np.array([[ 0. ,  1. ,  0. ,  1. ,  1. ],
                    [ 1. ,  0. ,  0. ,  1.,   1. ],
                    [ 0. ,  0. ,  0. ,  1.,   1. ],
                    [ 1. ,  1.,   1. ,  0. ,  1. ],
                    [ 1. ,  1.,   1. ,  1. ,  0. ]])
worker = kw.classical.TabuSearchOptimizer(10, size_limit=1)
worker.solve(matrix) 
array([[ 1,  1,  1, -1, -1]])
set_matrix(ising_matrix)
设置矩阵并更新相关内容

init_solution(solution)
初始化解向量

Args:
solution (np.ndarray): 初始解向量

solve(ising_matrix=None, solution=None)
求解Ising矩阵

Args:
ising_matrix (np.ndarray, optional): Ising矩阵. 默认为None.

solution (np.ndarray, optional): 初始解向量. 默认为None.

Returns:
np.ndarray: 解向量

on_matrix_change()
更新矩阵相关信息, 继承OptimizerBase时可以实现。当处理的ising矩阵发生变化时，这个函数的实现会被调用，从而有机会做相应动作

class kaiwu.classical.BruteForceOptimizer
Bases: OptimizerBase

求解Ising模型矩阵的暴力求解器，慢而准.

solve(ising_matrix=None)
求解Ising矩阵solve接口

Args:
ising_matrix (np.ndarray, optional): Ising矩阵. Defaults to None.

init_solution (np.ndarray, optional): 初始解向量. Defaults to None.

Returns:
np.ndarray: 1个或者多个能量最低的解向量.

Examples:

import kaiwu as kw
import numpy as np
mat = np.array([[0, 2, -3],[2, 0, -1],[-3, -1, 0]])
optimizer = kw.classical.BruteForceOptimizer()
optimizer.solve(mat)
array([[-1, -1,  1],
       [-1, -1,  1]])
on_matrix_change()
更新矩阵相关信息, 继承OptimizerBase时可以实现。当处理的ising矩阵发生变化时，这个函数的实现会被调用，从而有机会做相应动作

set_matrix(ising_matrix)
设置矩阵并更新相关内容