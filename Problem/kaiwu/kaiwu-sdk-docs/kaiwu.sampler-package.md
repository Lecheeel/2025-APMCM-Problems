kaiwu.sampler package
Module contents
模块: sampler

功能: 提供一系列数据后处理工具

class kaiwu.sampler.SimulatedAnnealingSampler(initial_temperature=100, alpha=0.99, cutoff_temperature=0.001, iterations_per_t=10, size_limit=100, flag_evolution_history=False, verbose=False, rand_seed=None, process_num=1)
Bases: SimulatedAnnealingSamplerBase

求解CIM Ising模型的模拟退火采样器。每个解都独立求解

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

single_process_solve(ising_matrix=None, init_solution=None, rand_seed=None, size_limit=None)
单进程求解Ising矩阵

Args:
ising_matrix (np.ndarray, optional): Ising矩阵. Defaults to None.

init_solution (np.ndarray, optional): 初始解向量. Defaults to None.

rand_seed (int, optional): numpy随机数生成器的随机种子

Returns:
np.ndarray: 解向量

set_matrix(ising_matrix)
设置矩阵并更新相关内容

solve(ising_matrix=None, init_solution=None, size_limit=None)
SA求解Ising矩阵solve接口

Args:
ising_matrix (np.ndarray, optional): Ising矩阵. Defaults to None.

init_solution (np.ndarray, optional): 初始解向量. Defaults to None.

Returns:
np.ndarray: 多个进程合并去重之后的解向量.