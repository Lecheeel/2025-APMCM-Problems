# 概述

## Kaiwu SDK

Kaiwu SDK是一套软件开发套件，专为基于相干光量子计算机的QUBO问题求解而设计。该套件致力于为开发者提供一个便捷的Python环境，以构建适用于相干光量子计算机的软件算法，并直接通过物理接口（目前支持Ising矩阵）调用量子计算机真机进行计算。

Kaiwu SDK目前包含以下核心模块：qubo、cim、ising、preprocess、classical、sampler、solver和utils。在典型应用场景中，用户首先利用qubo模块进行问题建模，随后借助solver模块对构建的QUBO模型进行高效求解。solver模块不仅负责管理QUBO模型的系数，还支持调用用户指定的optimizer对最终的Ising模型矩阵进行深度求解。

我们特别将Ising模型矩阵的求解功能定义为optimizer。其中，基于经典计算实现的optimizer，如SA、tabu等经典模拟求解器，均集成在classical模块中；而直接对接真机的optimizer则位于cim模块。此外，其他辅助模块提供了日志记录、降阶处理和降精度处理等实用功能，为用户提供全方位的支持。

## 预备知识

### CIM

相干伊辛机(Coherent Ising Machine，简称CIM)，是目前玻色量子重点研发的一项量子计算机技术。CIM是一种基于简并光学参量振荡器(DOPO)的光量子计算机。在数学实践中，我们可以将其抽象为优化Ising模型的专用计算机。

### Ising模型

伊辛模型(Ising Model)，是一类描述物质相变的随机过程模型。抽象为数学形式为：

$$
H(\sigma) = -\sum_{i,j} J_{ij}\sigma_i\sigma_j - \mu\sum_i h_i\sigma_i
$$

其中$\sigma$为待求自旋变量，取值为$\{-1, 1\}$， $H$为哈密顿量， $J$为二次项系数，$\mu$和$h$为线性项系数，是已知量。

### QUBO

二次无约束二值优化问题(Quadratic unconstrained binary optimization，简称QUBO)，其数学形式如下：

$$
f_Q(x) = \sum_{i \leq j} q_{ij}x_i x_j
$$

其中$x$为待求二进制变量，取值为$\{0, 1\}$，$f$为目标函数，$q$为二次项系数，是已知量。写成线性代数的形式:

$$
f_Q(x) = x^T Q x
$$

其中，$x$为二进制向量，$Q$为QUBO矩阵，QUBO目标是找到使得$f$最小或最大的$x$，即:

$$
x^* = \arg\min_x f_Q(x)
$$

在Kaiwu SDK中，通过`kw.qubo.details`查看QUBO模型细节会显示offset和coefficients信息。其中offset表示QUBO模型中的常数项，与变量无关。coefficients表示QUBO模型中每个二值变量的系数取值，以及它们的交互项的系数取值。

### CIM求解模型

CIM求解QUBO或优化Ising模型的过程就是，将QUBO中的$q_{ij}$或Ising模型中的$J_{ij}$输入CIM，CIM返回$x$或$\sigma$的过程。

# 高阶问题求解

## 1. 高阶优化

QUBO（Quadratic Unconstrained Binary Optimization）二次无约束二元优化的目的是求得一组布尔变量的值 $(x_0, x_1, \dots x_n)$，使得二阶多项式 $x^T Q x$ 的值最小。

而HOBO（Higher Order Binary Optimization）高阶二元优化可以通过添加约束条件转化为QUBO问题，具体来说，即通过变量替换，令 $y = x_0 x_1$，将原式中的单项式阶数降低，并添加 $y = x_0 x_1$ 的约束。

而要使得约束成立的方式是在原式中添加惩罚项，即Rosenberg二次惩罚项，

$p(x_0, x_1, y) = x_0 x_1 - 2x_0 y - 2x_1 y + 3y$。该惩罚项满足

$y = x_0 x_1 \rightarrow p(x_0, x_1, y) = 0, y \neq x_0 x_1 \rightarrow p(x_0, x_1, y) > 0$

最终新的多项式为 $f(x, y) + k \sum p(x_i, x_j, y_{ij})$，其中k是惩罚项系数

## 2. 注意事项

新变量为原变量名字用下划线相连，如x0和x1被替换为b_x0_x1，b_x0_x1和b_x0_y1被替换为 b_x0_x1_y1

`b_`是内部保留符号, 不对用户开放用于命名变量.

## 3. 使用举例

### (1) 降阶

```python
import numpy as np
import kaiwu as kw

x = kw.core.ndarray(10, "x", kw.core.Binary)
y1 = x[0]*x[1] + x[2]*x[3] + x[8]
y2 = x[3]*x[4] + x[5]*x[6] + x[7]
y3 = y1 * y2
print(y3, "\n")
hobo_model = kw.hobo.HoboModel(y3)
qubo_model = hobo_model.reduce()
print(qubo_model)
```

执行以上代码后结果为

```
x[2]*x[3]*x[5]*x[6]+x[2]*x[3]*x[3]*x[4]+x[2]*x[3]*x[7]+x[0]*x[1]*x[5]*x[6]+x[0]*x[1]*x[3]*x[4]+x[0]*x[1]*x[7]+x[5]*x[6]*x[8]+x[3]*x[4]*x[8]+x[7]*x[8]

QUBO Details:
  Variables(Binary):b_x[2]_x[3], b_x[5]_x[6], b_x[0]_x[1], b_x[3]_x[4], x[4], x[7], x[8]
  QUBO offset:      0
  QUBO coefficients:
    b_x[2]_x[3], b_x[5]_x[6] : 1
    b_x[2]_x[3], x[4]        : 1
    b_x[2]_x[3], x[7]        : 1
    b_x[0]_x[1], b_x[5]_x[6] : 1
    b_x[0]_x[1], b_x[3]_x[4] : 1
    b_x[0]_x[1], x[7]        : 1
    b_x[5]_x[6], x[8]        : 1
    b_x[3]_x[4], x[8]        : 1
    x[7], x[8]               : 1
  HOBO Constraint:
    b_x[5]_x[6] : x[5], x[6]
    b_x[2]_x[3] : x[2], x[3]
    b_x[0]_x[1] : x[0], x[1]
    b_x[3]_x[4] : x[3], x[4]
```

### (2) 检查求得的结果是否满足降阶约束条件

```python
x1, x2, x3 = kw.core.Binary("x1"), kw.core.Binary("x2"), kw.core.Binary("x3")
p = x1*x2*x3
hobo_model = kw.hobo.HoboModel(p)
qubo_model = hobo_model.reduce()
solution = {"x1": 1, "x2": 1, "x3": 0, "b_x1_x2": 1}
err_cnt, _ = hobo_model.verify_constraint(solution)
print(err_cnt)  # 输出0，证明解满足降阶的约束
```

# 参数精度

## 参数精度转化要求

在对实际问题进行建模时，常常得到QUBO形式，再将其转化成Ising模型就可以应用于物理机了。QUBO模型需要转化为 Ising模型 $\mathcal{H} = -\sum_{i,j} J_{ij}s_i s_j - \mu\sum_i h_i s_i$ 以用于实际的物理计算。

实际计算中，系数位宽受到物理上硬件条件的限制，只能取有限的范围。

1.  CIM真机只支持8bit INT空间[-128, 127]
2.  用户建模要保证转换完的Ising矩阵符合要求
3.  提供的转换逻辑供参考，用户可以根据自己的矩阵使用更适合的方法

## QUBO转化为Ising

本节介绍QUBO如何转化为Ising，以便于理解动态范围检查如何限制QUBO矩阵。

QUBO模型如下：

$$
\mathcal{H} = \sum_i q_{ii}x_i + \sum_{i \ne j} q_{ij}x_i x_j
$$

在计算之前需要将其转化为Ising矩阵。令 $s_i = 2x_i - 1 \in \{1, -1\}$,

$$
\begin{aligned}
\mathcal{H} &= \sum_i q_{ii} \cdot \frac{s_i + 1}{2} + \sum_{i \ne j} q_{ij} \cdot \frac{(s_i + 1)(s_j + 1)}{4} \\
&= \sum_{i \ne j} \frac{q_{ij}}{4} s_i s_j + \sum_i \left( \frac{q_{ii}}{2} + \frac{\sum_{k \ne i} (q_{ik} + q_{ki})}{4} \right) s_i + \left( \frac{\sum_i q_{ii}}{2} + \frac{\sum_{i \ne j} q_{ij}}{4} \right)
\end{aligned}
$$

QUBO变量满足 $x^2 = x$，可以用矩阵的对角线元素表达一次项。而Ising模型 $x^2 = 1$，不能这样表示。

上式中的一次项通过添加辅助变量 $s$ 化为二次项，辅助变量取1和-1时可以分别对应到添加辅助变量前的两组解。故添加辅助变量后与原问题等价。

常数项单独记录，在计算最终哈密顿量时加上即可。

**举例：**

$$
x^\top \begin{pmatrix} 0 & 2 \\ 1 & 1 \end{pmatrix} x
$$

经过变换之后为

$$
s^\top \begin{pmatrix} 0 & 3/8 & 3/8 \\ 3/8 & 0 & 5/8 \\ 3/8 & 5/8 & 0 \end{pmatrix} s + 3/2
$$

## kaiwuSDK降低精度的方法

kaiwuSDK提供了两种降低参数精度的方法，`perform_precision_adaption_mutate` 和 `perform_precision_adaption_split`。mutate方法在修改矩阵的同时能够保持矩阵的解不变，但能够改变精度的程度取决于矩阵本身的可下降空间。split方法能够将矩阵修改到任意精度，但是新矩阵的比特数随着精度变化量增长较快。

### perform_precision_adaption_mutate

#### 1 动态范围

##### 1.1 QUBO矩阵相关定义

**定义** Q为qubo矩阵，$Q \subseteq Q'$ 表示Q的最优解包含于Q'的最优解

**定义** $[Q]$ 为对Q的每个元素round取整

##### 1.2 动态范围相关定义

**定义** $X$ 为系数矩阵

$\hat{D}(X)$ 为X元素的最大距离

$\check{D}(X)$ 为X元素的最小非0距离

$DR(X) := log_2(\frac{\hat{D}(X)}{\check{D}(X)})$ 为X的动态范围

**命题** $\hat{D}(Q) = 1$, 那么 $\forall \alpha > 0, DR([\alpha Q]) \le log_2(1+\alpha)$

动态范围的定义可以适用于QUBO矩阵和Ising矩阵。通过减小动态范围，可以降低原矩阵所需要的参数精度。

由于最终在Ising矩阵上进行计算，所以kaiwuSDK降低动态范围的操作直接作用于Ising矩阵

#### 2. 应用举例

```python
import kaiwu as kw
import numpy as np

mat0 = np.array([[0, -20, 0, 40, 1.1],
                [0, 0, 12240, 1, 120],
                [0, 0, 0, 0, -10240],
                [0, 0, 0, 0, 2.05],
                [0, 0, 0, 0, 0]])
mutated_mat = kw.preprocess.perform_precision_adaption_mutate(mat0)
print(mutated_mat)
```

### perform_precision_adaption_split

#### 1. 参数精度

由于当前量子计算机对Ising矩阵系数的存储方式为定点数，且只有8位精度，对于最大最小值的比值超过 $2^8$ 的多项式，需要将系数大的项分拆。

实现方式为将原式中的比特替换成值相等的多个等价比特，相等条件由约束项实现，从而使得每一项的系数都能够缩小。

常用QUBO建模后转化为Ising模型，这里以QUBO表达式为例，拆分的方式为将 $f(x)$ 转化为

$$
\begin{aligned}
f(\mathbf{x}, \mathbf{x'}) + M\sum_i (x_i - x_{i1})^2 + M\sum_{i,j} (x_{ij} - x_{i(j+1)})^2 = \\
f(\mathbf{x}, \mathbf{x'}) + M\sum_i (x_i + x_{i1} - 2x_i x_{i1}) + M\sum_{i,j} (x_{ij} + x_{i(j+1)} - 2x_{ij} x_{i(j+1)})
\end{aligned}
$$

例如，对于 $x_1 + 2x_2 + 200x_3$，要求多项式系数最大最小值的比值不能超过150，阈值设置为那么将多项式修改为 $x_1 + 2x_2 + 100x_3 + 100x_{31} + 50(x_3 - x_{31})^2 = x_1 + 2x_2 + 100x_3 + 100x_{31} + 50(x_3 + x_{31} - 2x_3 x_{31})$

#### 2. 应用举例

```python
import kaiwu as kw
import numpy as np

mat = np.array([[0, -15,0, 40],
                [-15,0, 0, 1],
                [0,  0, 0, 0],
                [40, 1, 0, 0]])
splitted_ret, last_idx = kw.preprocess.perform_precision_adaption_split(mat, 4)
print(splitted_ret)
```

min_increment计算得到默认值为1。精度设置为4个比特，范围在-7到7，绝对应该小于7。

通过这样拆分变量，可以在保持原矩阵的解的情况下，将参数精度降低。

降低精度的过程通过param_bit, min_increment, penalty, round_to_increment等参数来调节。

矩阵的值都是min_increment的整数倍，默认取矩阵元素间的最小正差值

round_to_increment表示在这个过程中，通过调整各个元素，使得拆分后的元素的和等于原值。如15拆分成7.5+7.5，若分别近似取整会变成8+8。设置reduce_error=True，15会自动拆分为7+8

```python
splitted_ret2, last_idx2 = kw.preprocess.perform_precision_adaption_split(mat, 4, min_increment=3, round_to_increment=True)
print(splitted_ret2)
```

结果为

```python
[[ 0. 21. -6.  0.  3. 12.]
[21.  0. -9.  0. 12. 12.]
[-6. -9.  0.  0.  0.  0.]
[ 0.  0.  0.  0.  0.  0.]
[ 3. 12.  0.  0.  0. 21.]
[12. 12.  0.  0. 21.  0.]]
```

对拆分后精度满足要求的矩阵进行求解后，可以通过`restore_split_solution`将其恢复成原矩阵的解

```python
worker = kw.classical.SimulatedAnnealingOptimizer()
output = worker.solve(splitted_ret)
sol = output[0]
org_sol = kw.preprocess.restore_split_solution(sol, last_idx)
print(org_sol)
```

结果为

```python
[-1.  1. -1. -1.]
```

## PrecisionReducer

为了方便用户在SDK中迭代调用CIM量子计算机，SDK提供了PrecisionReducer，它应用了装饰模式（一种软件设计模式）。当用户需要在提交量子计算机求解时，可以在CIMOptimizer之外再套一个PrecisionReducer，然后把PrecisionReducer作为Optimizer传给Solver。


# 自动确定惩罚系数

## 1. 惩罚系数

在通过QUBO对实际问题建模的过程中，约束条件的一种常见处理方式是将其转化为惩罚项，添加到目标函数中。确定约束条件的合适的惩罚系数有时是一个花费时间的事情。

对于传统方法，过大的惩罚系数会导致算法忽略了目标函数，导致求到的可行解没有一个好的目标函数。对于真机，过大的惩罚系数也可能导致精度方面的问题。与此相反，过小的惩罚系数可能会导致约束没有被优先满足。因此，估计出合适的惩罚系数是有意义的。

## 2. 估计方法

当某个离散变量的值变化时，如果在目标函数引起的变化量恒小于在惩罚项引起的变化量，则惩罚项能够被优先满足。所以只要估计出每个变量引起的变化量的比例，就能得到惩罚系数。

目标函数的最大变化量可以通过上界来估计

$$
x^\top Q x = f(x_i) = \sum_j q_{ij} x_i x_j + \sum_{k \ne i,j} q_{kj} x_k x_j
$$

QUBO表达式中，与 $x_i$ 相关的项可以分离出来。

当 $x_i$ 从0变化为1时

$$
\Delta f(x_i) = \sum_j q_{ij} x_j < \sum_{q_{ij}>0} q_{i,j}
$$

当 $x_i$ 从1变化为0时

$$
\Delta f(x_i) = -\sum_j q_{ij} x_j < \sum_{q_{ij}<0} q_{i,j}
$$

估计惩罚项的最小非零变化量，二者比值可以作为惩罚系数

## 3. 例子

```python
import kaiwu as kw
x = kw.core.ndarray(4, 'x', kw.core.Binary)
cons = (x[0] + x[1] + 2*x[3] - 2)**2 + 2 * x[2]
obj = 3*(x[0] - x[1])**2 - 2 * x[2]

penalty = kw.core.get_min_penalty(obj, cons)
qubo_expr = obj + penalty * cons
```
