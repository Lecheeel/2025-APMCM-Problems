kaiwu.cim package
Module contents
模块: cim

功能: 提供一系列CIM求解器相关工具

class kaiwu.cim.PrecisionReducer(component: OptimizerBase, precision=8, target_bits=None, only_feasible_solution=True, truncated_precision=20)
Bases: OptimizerBase

降低精度装饰类

可打开日志看到详细输出：kw.common.set_log_level(“DEBUG”)

Args:
component (OptimizerBase): 基础类.

precision (int): 矩阵目标精度.

target_bits (int): 矩阵目标比特数. 默认为None，不进行控制。

only_feasible_solution (bool): 是否只要可行解，默认True，当only_feasible_solution=True且所有解都不是可行解时会抛出异常。

truncated_precision (int): 截断精度，可以计算的最大精度，建议使用默认值。

Examples:
import numpy as np
import kaiwu as kw
matrix = -np.array([[ 0. ,  1.23 ,  0. ,  1. ,  1. ],
                    [ 1.23 ,  0. ,  0. ,  1.,   1. ],
                    [ 0. ,  0. ,  0. ,  1.,   1. ],
                    [ 1. ,  1.,   1. ,  0. ,  1. ],
                    [ 1. ,  1.,   1. ,  1. ,  0. ]])
optimizer = kw.classical.SimulatedAnnealingOptimizer(initial_temperature=100,
                                                     alpha=0.99,
                                                     cutoff_temperature=0.001,
                                                     iterations_per_t=10,
                                                     size_limit=5)
new_optimizer = kw.cim.PrecisionReducer(optimizer, precision=4)
new_optimizer.solve(matrix) 
array([[ 1, -1, -1, -1, 1],
[ 1, -1, 1, -1, -1], [ 1, -1, 1, -1, 1], [-1, -1, -1, 1, 1], [ 1, 1, 1, -1, -1]])

solve(ising_matrix=None)
求解

on_matrix_change()
更新矩阵相关信息, 继承OptimizerBase时可以实现。当处理的ising矩阵发生变化时，这个函数的实现会被调用，从而有机会做相应动作

set_matrix(ising_matrix)
设置矩阵并更新相关内容

class kaiwu.cim.CIMOptimizer(task_name_prefix, wait=False, interval=1, project_no=None)
Bases: OptimizerBase

CIM Optimizer Interface

CIMOptimizer 是一种用于求解 Ising 计算问题的优化器 (Optimizer)，它通过提交任务到 相干光计算机 (CIM, Coherent Ising Machine) 真机进行计算，并返回最优解。

主要功能包括：

任务提交：将 Ising 矩阵任务上传至 CIM 计算平台，并创建计算任务。

任务查询：定期检查任务计算状态，获取计算结果。

缓存管理：本地缓存已计算任务的结果，避免重复提交。

Args:
task_name_prefix (str): 任务名称前缀，完整任务名称为 task_name_prefix_{ising_matrix_hash}

wait (bool, optional): 是否等待计算完成，默认为 False。

interval (int, optional): 轮询间隔时间（分钟），默认值为 1，最小值 1 分钟。

project_no (str, optional): 项目编号，值为CPQC-X中项目列表的项目ID, 用于创建项目下的任务

Example:
import numpy as np
import kaiwu as kw
kw.common.CheckpointManager.save_dir = '/tmp'
matrix = -np.array([[ 0. ,  1. ,  0. ,  1. ,  1. ],
                    [ 1. ,  0. ,  0. ,  1.,   1. ],
                    [ 0. ,  0. ,  0. ,  1.,   1. ],
                    [ 1. ,  1.,   1. ,  0. ,  1. ],
                    [ 1. ,  1.,   1. ,  1. ,  0. ]])
optimizer = kw.cim.CIMOptimizer(task_name_prefix='cim_optimizer_test')  
solution = optimizer.solve(matrix)  
print(solution)  
array([[-1,  1,  1, -1, -1],
       [-1,  1,  1,  1, -1],
       [-1,  1,  1, -1,  1],
       [ 1, -1, -1, -1,  1],
       [ 1, -1,  1,  1, -1],
       [ 1, -1,  1, -1,  1],
       [ 1, -1,  1, -1, -1],
       [ 1, -1, -1,  1,  1],
       [-1, -1, -1,  1,  1],
       [ 1,  1,  1, -1, -1]], dtype=int8)
kw.common.CheckpointManager.save_dir = None
Notes:
需要通过 CheckpointManager 设置中间文件保存路径 (save_dir)。

任务的唯一标识由 ising_matrix 和 task_name_prefix 共同决定，任何一项的变化都会创建新的任务。

同一个矩阵可以通过修改 task_name_prefix 创建不同的任务，若仅需查询结果，请确保 task_name_prefix 不变。

实例化CIMOptimizer时 task_name_prefix 必传，任务名: {task_name_prefix}_{hash(ising_matrix)}

solve(ising_matrix=None)
入口函数

Args:
ising_matrix (np.ndarray): Ising矩阵

Returns:
np.ndarray | None:
解向量集合（任务完成时）

None（任务仍在计算中）

get_task_result(ising_matrix: ndarray) → dict
获取任务结果

on_matrix_change()
更新矩阵相关信息, 继承OptimizerBase时可以实现。当处理的ising矩阵发生变化时，这个函数的实现会被调用，从而有机会做相应动作

set_matrix(ising_matrix)
设置矩阵并更新相关内容