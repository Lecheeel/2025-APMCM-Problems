kaiwu.solver package
Module contents
模块: solver

功能: 提供一系列基于cim/classical中的opitimizer进行求解的solver

class kaiwu.solver.PenaltyMethodSolver(optimizer, controller)
Bases: SolverBase, JsonSerializableMixin

罚函数法

设置 kw.common.CheckpointManager.save_dir = ‘/tmp’ 会打开缓存开关，当前状态和符合硬约束的惩罚系数组合都会被缓存 打开缓存后 PenaltyMethodSolver 可以在上一次程序终止的基础上继续迭代求解, 程序运行过程中的可行解会保存对应json文件里面 results 中.

Args:
optimizer (Optimizer): 优化器

controller (SolverLoopController): 循环控制器

Examples:
import kaiwu as kw
import numpy as np
kw.common.CheckpointManager.save_dir = '/tmp'
taskNums = 20
machineNums = 5
duration = np.array(range(1, taskNums + 1))
machine_start_time = np.array(range(1, machineNums + 1))
# Building a Qubo Model
qubo_model = kw.qubo.QuboModel()
X = kw.core.ndarray([taskNums, machineNums], "X", kw.core.Binary)
J_mean = (np.sum(duration) + np.sum(machine_start_time)) / machineNums
J = [machine_start_time[i] + duration.dot(X[:, i]) for i in range(machineNums)]
# Set objective function
qubo_model.set_objective(kw.core.quicksum([(J[i] - J_mean) ** 2 for i in range(machineNums)]) / machineNums)
# Set constraint
for j in range(taskNums):
    qubo_model.add_constraint((1 - kw.core.quicksum([X[j][i] for i in range(machineNums)])) ** 2 == 0,
                              f"c{j}", penalty=45)
# Loop control
controller = kw.common.SolverLoopController(max_repeat_step=5)
# optimizer
optimizer = kw.classical.SimulatedAnnealingOptimizer(initial_temperature=1e8,
                                                     alpha=0.9,
                                                     cutoff_temperature=0.01,
                                                     iterations_per_t=200,
                                                     size_limit=100)
solver = kw.solver.PenaltyMethodSolver(optimizer, controller)
sol_dict, hmt = solver.solve_qubo(qubo_model)
J_end = np.zeros(machineNums)
for i in range(machineNums):
    ifturn_temp = kw.core.get_val(kw.core.quicksum(X[:, i].tolist()), sol_dict)
    ifturn = 1 if ifturn_temp > 0 else 0
    J_temp = duration.dot(X[:, i]) + machine_start_time[i] * ifturn
    J_end[i] = kw.core.get_val(J_temp, sol_dict)
print("duration array: ", duration)  
duration array:  [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
print("Machine startup time: ", machine_start_time)  
Machine startup time:  [1 2 3 4 5]
print("End time of all machines：", J_end)  
End time of all machines： [43. 43. 48. 45. 46.]
print('final result：{}'.format(np.max(J_end)))  
final result：48.0
print('variance：', np.var(J_end))   
variance: 3.6
kw.common.CheckpointManager.save_dir = None
solve_qubo(*args, **kwargs)
solve_qubo_multi_results(qubo_model, size_limit=10)
求解QUBO模型, 并返回多个解

Args:
qubo_model (QuboModel): QUBO模型

size_limit (int): 返回解得数量

Returns:
list: QUBO模型解SolutionResult的字典、目标值，形如:

[
    {
        'sol_dict': {'x[0]': 1.0, 'x[1]': 0.0, 'x[2]': 0.0, 'x[3]': 0.0, 'x[4]': 1.0, 'x[5]': 0.0, ...}
        'objective': -15.01
    },
    ...
]
to_json_dict(exclude_fields=())
转化为json字典

Returns:
dict: json字典

load_json_dict(json_dict)
从json文件读取的dict恢复对象

Returns:
dict: json字典

class kaiwu.solver.SimpleSolver(optimizer)
Bases: SolverBase

实现用Optimizer直接对QuboModel进行求解 Examples:

import kaiwu as kw
n = 10
W = 5
p = [i + 1 for i in range(n)]
w = [(i + 2) / 2 for i in range(n)]
x = kw.core.ndarray(n, 'x', kw.core.Binary)
qubo_model = kw.qubo.QuboModel()
qubo_model.set_objective(sum(x[i] * p[i] * (-1) for i in range(n)))
qubo_model.add_constraint(sum(x[i] * w[i] for i in range(n)) <= W, "c", penalty=10)
solver = kw.solver.SimpleSolver(kw.classical.SimulatedAnnealingOptimizer(alpha=0.999, iterations_per_t=10))
sol_dict, qubo_val = solver.solve_qubo(qubo_model)
unsatisfied_count, result_dict = qubo_model.verify_constraint(sol_dict)
unsatisfied_count
0
Returns:
tuple: Result dictionary and Result dictionary.

dict: Result dictionary. The key is the variable name, and the value is the corresponding spin value.

float: qubo value.

solve_qubo(*args, **kwargs)