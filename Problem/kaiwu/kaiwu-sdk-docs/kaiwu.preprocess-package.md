kaiwu.preprocess package
Module contents
模块: preprocess

功能: 预处理相关功能，目前主要是针对ising矩阵的精度相关函数

kaiwu.preprocess.get_dynamic_range_metric(mat)
计算动态范围值（dynamic range, DR）

DR(Q) = log(max_{i,j}|Q_i - Q_j|/min_{Q_i \neq Q_j}|Q_i-Q_j|)

Args:
mat: Ising或QUBO矩阵

Returns:
float: 动态范围

Examples:
import numpy as np
mat = np.array([[0, 8, 1 ,1],
                [0, 0, 2, -1],
                [0, 0, 0, -8],
                [0, 0, 0, 0]])
import kaiwu as kw
kw.preprocess.get_dynamic_range_metric(mat)
np.float64(4.0)
kaiwu.preprocess.get_min_diff(mat)
计算最小差值

Args:
mat: Ising或QUBO矩阵

Returns:
float: 最小差值

Examples:
import numpy as np
mat = np.array([[0, 8, 1 ,1],
                [0, 0, 2, -1],
                [0, 0, 0, -8],
                [0, 0, 0, 0]])
import kaiwu as kw
kw.preprocess.get_min_diff(mat)
np.int64(1)
kaiwu.preprocess.lower_bound_parameters(ising_mat)
所有系数绝对值的和取负，确定Ising模型哈密顿量的下界

Args:
ising_mat (np.ndarray): Ising矩阵

Returns:
float: 哈密顿量的下界

Examples:
import numpy as np
import kaiwu as kw
mat = np.array([[0, 18, -12],
                [18, 0, 1],
                [-12, 1, 0]])
lb = kw.preprocess.lower_bound_parameters(mat)
lb
np.int64(-62)
kaiwu.preprocess.upper_bound_sample(ising_matrix, steps=10)
基于采样估计哈密顿量的上界

Args:
ising_matrix (np.ndarray): Ising矩阵

steps (int): 步数

Returns:
float: 哈密顿量的上界

Examples:
import numpy as np
import kaiwu as kw
mat = np.array([[0, 18, -12],
                [18, 0, 1],
                [-12, 1, 0]])
ub = kw.preprocess.upper_bound_sample(mat)
kaiwu.preprocess.upper_bound_simulated_annealing(ising_matrix)
基于模拟退火估计哈密顿量的上界

Args:
ising_matrix (np.ndarray): Ising矩阵

Returns:
float: 哈密顿量的上界

Examples:
import numpy as np
import kaiwu as kw
mat = np.array([[0, 18, -12],                            
                [18, 0, 1],
                [-12, 1, 0]])
ub = kw.preprocess.upper_bound_simulated_annealing(mat)
kaiwu.preprocess.perform_precision_adaption_mutate(ising_matrix, iterations=100, heuristic='greedy', decision='heuristic')
迭代减小Ising矩阵的动态范围，每次改变一个系数，保持最优解不变。思路参考 Mücke et al. (2023). 此方法会改变矩阵系数数值，但不保证一定能降低精度。可做探索性尝试

Args:
ising_matrix (np.ndarray): Ising矩阵

iterations (int, optional): 迭代次数，默认为100

heuristic (str, optional): 确定系数变化量的启发式方法，包括’greedy’和’order’。默认为’greedy’

decision (str, optional): 决定下一个修改位置的方法。包括’random’和’heuristic’。’heuristic’会优先选择直接影响动态范围的变量。 默认为’heuristic’。

Returns:
np.ndarray: 压缩参数的Ising矩阵

Examples:
import numpy as np
mat0 = np.array([[0., -10., 0., 20., 0.55],
   [-10., 0.,  6120., 0.5, 60.],
   [0.,  6120., 0.,   0., -5120.],
   [20.,  0.5,  0.,   0., 1.025],
   [0.55, 60., -5120., 1.025, 0.]])
import kaiwu as kw
kw.preprocess.perform_precision_adaption_mutate(mat0) 
array([[ 0.  , -2.05,  0.  , 40.  ,  0.  ],
       [-2.05,  0.  ,  4.1 ,  0.  ,  0.  ],
       [ 0.  ,  4.1 ,  0.  ,  0.  , -2.05],
       [40.  ,  0.  ,  0.  ,  0.  ,  2.05],
       [ 0.  ,  0.  , -2.05,  2.05,  0.  ]])
kaiwu.preprocess.perform_precision_adaption_split(ising_matrix: ndarray, param_bit=8, min_increment=None, penalty=None, round_to_increment=True)
将变量拆分, 使得QUBO表达式的系数范围缩小, 能够使用要求的比特数表达

Args:
ising_matrix (np.ndarray): Ising矩阵

param_bit (int): Ising矩阵元素可以使用的比特数，表示矩阵的参数精度。默认为8

min_increment (float): 矩阵元素的最小变化量，表示转化后矩阵的分辨率。默认值取矩阵元素间差值的最小正值

penalty (float): 惩罚项系数， 默认为min_increment * (2^(param_bit-1) - 1)

round_to_increment (bool): 将矩阵的所有元素转化为min_increment的整数倍，使得其可以用不超过param_bit的比特数表达

Returns:
tuple: 返回元组，包含新矩阵和变量索引
np.ndarray: 包含缩小范围的新矩阵。新矩阵单个元素并不一定在精度范围内，但是整体除以min_increment后会在精度范围内

np.ndarray: 变量分拆后该变量第一次出现的位置

Examples:
import numpy as np
import kaiwu as kw
mat = np.array([[0, 18, -12],
                [18, 0, 1],
                [-12, 1, 0]])
kw.preprocess.perform_precision_adaption_split(mat, param_bit=5, min_increment=1, penalty=4,
       round_to_increment=True)
(array([[ 0.,  4.,  3.,  5., -6.],
       [ 4.,  0.,  5.,  5., -6.],
       [ 3.,  5.,  0.,  4.,  0.],
       [ 5.,  5.,  4.,  0.,  1.],
       [-6., -6.,  0.,  1.,  0.]]), array([1, 3, 4]))
kaiwu.preprocess.restore_split_solution(solution, last_var_idx)
将缩小范围多项式的解转换回原来的表达式的解

Args:
solution(np.ndarray): 求得的解

last_var_idx(np.ndarray): 分拆后每个变量最后一次出现的位置

Returns:
np.ndarray: 原多项式的解

Examples:
import kaiwu as kw
import numpy as np
mat = np.array([[0, -15, 0, 30],
               [-15, 0, 0, 2],
               [0, 0, 0, 0],
               [30, 2, 0, 0]])
r, f = kw.preprocess.perform_precision_adaption_split(mat, 5, min_increment=0.5, round_to_increment=True)
worker = kw.classical.SimulatedAnnealingOptimizer()
opt = worker.solve(r)
sol = opt[0] * opt[0, -1]
kw.preprocess.restore_split_solution(sol, f)  
array([ 1, -1, -1,  1], dtype=int8)
kaiwu.preprocess.construct_split_solution(solution, last_var_idx)
将原矩阵分拆变量降低参数精度后，根据原矩阵的解构造新矩阵的解

Args:
solution (np.ndarray): 原矩阵的解

last_var_idx (np.ndarray): 分拆后每个变量最后一次出现的位置

Returns:
np.ndarray: 新矩阵的解

Examples:
import numpy as np
import kaiwu as kw
mat = np.array([[0, -15, 0, 40],
                [-15, 0, 0, 2],
                [0, 0, 0, 0],
                [40, 2, 0, 0]])
nmat, tail = kw.preprocess.perform_precision_adaption_split(mat, 5, min_increment=0.5,
                                                            round_to_increment=True,
                                                            penalty=0)
sol = np.array([1, 1, -1, -1])
ans = np.array([1, 1, 1, 1, -1, -1, -1, -1])
kw.preprocess.construct_split_solution(sol, tail)
array([ 1.,  1.,  1.,  1., -1., -1., -1., -1.])