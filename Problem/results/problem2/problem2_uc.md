## 一、模型基本信息

- **时间范围**：24 小时（\( t = 1,\dots,24 \)）
- **发电机组**：6 台（编号为 1, 2, 5, 8, 11, 13）
- **网络结构**：30 个节点、41 条支路（含线路参数）
- **负载数据**：每小时总负荷 \( D_t \)（单位：MW）

---

## 二、决策变量

| 变量 | 类型 | 含义 |
|------|------|------|
| \( u_{i,t} \) | 二元变量 | 机组 \( i \) 在时段 \( t \) 是否在线（1=是，0=否） |
| \( v_{i,t} \) | 二元变量 | 机组 \( i \) 在时段 \( t \) 启动（1=启动） |
| \( w_{i,t} \) | 二元变量 | 机组 \( i \) 在时段 \( t \) 停机（1=停机） |
| \( p_{i,t} \) | 连续变量 | 机组 \( i \) 在时段 \( t \) 的出力（MW） |
| \( \theta_{b,t} \) | 连续变量 | 节点 \( b \) 在时段 \( t \) 的电压相角（弧度）（参考节点 \( b=1 \) 固定为 0） |
| \( P^{\text{flow}}_{l,t} \) | 连续变量 | 线路 \( l \) 在时段 \( t \) 的有功潮流（MW） |

---

## 三、目标函数

最小化总运行成本（燃料成本 + 启停成本）：

\[
\min \sum_{i=1}^6 \sum_{t=1}^{24} \left[
a_i p_{i,t}^2 + b_i p_{i,t} + c_i
+ S_i^{\text{start}} v_{i,t} + S_i^{\text{shut}} w_{i,t}
\right]
\]

其中：
- \( a_i, b_i, c_i \)：机组 \( i \) 的二次成本系数（来自 Table 2）
- \( S_i^{\text{start}}, S_i^{\text{shut}} \)：启停成本（来自 Table 1）

---

## 四、核心约束（Problem 1 基础）

### 1. **系统功率平衡**
\[
\sum_{i=1}^6 p_{i,t} = D_t, \quad \forall t
\]

### 2. **发电上下限（与启停状态耦合）**
\[
P_{\min,i} \cdot u_{i,t} \le p_{i,t} \le P_{\max,i} \cdot u_{i,t}, \quad \forall i, t
\]

### 3. **启停逻辑一致性**
\[
v_{i,t} - w_{i,t} = 
\begin{cases}
u_{i,t} - u_{i,t-1}, & t \ge 2 \\
u_{i,t} - u_{i,0}^{\text{init}}, & t = 1
\end{cases}
\quad \forall i
\]

其中 \( u_{i,0}^{\text{init}} \in \{0,1\} \) 为初始状态。

### 4. **最小开机时间（Min-Up Time）**
- 若机组在 \( t \) 启动（\( v_{i,t}=1 \)），则后续 \( T^{\text{up}}_i - 1 \) 小时必须保持开机。
- 初始开机时间已满足的情况下，前若干时段强制 \( u_{i,t}=1 \)。

数学形式（简化）：
\[
u_{i,t} \ge \sum_{\tau = \max(1, t - T_i^{\text{up}} + 1)}^{t} v_{i,\tau}, \quad \forall i, t
\]

> 注：实际使用 **Table 2** 中的 \( T_i^{\text{up}} \)（可通过命令行切换，但默认为 Table 2）

### 5. **最小停机时间（Min-Down Time）**
- 若机组在 \( t \) 停机（\( w_{i,t}=1 \)），则后续 \( T^{\text{down}}_i - 1 \) 小时必须保持关机。
- 初始停机时间已满足的情况下，前若干时段强制 \( u_{i,t}=0 \)。

数学形式（简化）：
\[
u_{i,t} \le 1 - \sum_{\tau = \max(1, t - T_i^{\text{down}} + 1)}^{t} w_{i,\tau}, \quad \forall i, t
\]

### 6. **爬坡约束（Ramp Limits）**
- **开机时**：\( p_{i,1} - p_{i,0}^{\text{init}} \le R_i^{\text{up}} \)
- **运行中**：
  \[
  p_{i,t} - p_{i,t-1} \le R_i^{\text{up}} \cdot u_{i,t-1} + P_{\max,i} \cdot (1 - u_{i,t-1})
  \]
  \[
  p_{i,t-1} - p_{i,t} \le R_i^{\text{down}} \cdot u_{i,t} + P_{\max,i} \cdot (1 - u_{i,t})
  \]

### 7. **启停互斥**
\[
v_{i,t} + w_{i,t} \le 1, \quad \forall i, t
\]

---

## 五、Problem 2 新增约束（网络与安全）

### 8. **DC 潮流模型（线路约束）**

- **参考节点**：节点 1 的相角固定为 0：
  \[
  \theta_{1,t} = 0, \quad \forall t
  \]

- **线路潮流计算**（对每条支路 \( l \) 连接节点 \( m \to n \)）：
  \[
  P^{\text{flow}}_{l,t} = B_l (\theta_{m,t} - \theta_{n,t}), \quad \forall l, t
  \]
  其中 \( B_l = 1 / X_l \)（\( X_l \) 为线路电抗）。

- **线路容量限制**：
  \[
  -P^{\max}_l \le P^{\text{flow}}_{l,t} \le P^{\max}_l, \quad \forall l, t
  \]

### 9. **节点功率平衡（含负荷分布）**

- **负荷分配**：总负荷 \( D_t \) 按固定比例分配至 24 个负荷节点（非发电机节点），比例和为 1。
- 对每个节点 \( b \)：
  \[
  \sum_{i: \text{unit } i \text{ at bus } b} p_{i,t}
  - D_t \cdot \alpha_b
  = \sum_{l \in \text{out}(b)} P^{\text{flow}}_{l,t}
  - \sum_{l \in \text{in}(b)} P^{\text{flow}}_{l,t}
  \]
  其中 \( \alpha_b \) 为节点 \( b \) 的负荷分配系数。

### 10. **旋转备用约束（Spinning Reserve）**

- 每时段必须有足够的在线备用容量：
  \[
  \sum_{i=1}^6 (P_{\max,i} \cdot u_{i,t} - p_{i,t}) \ge R_t, \quad \forall t
  \]
- 其中 \( R_t = \max(10\% \cdot D_t,\ P_{\max}^{\text{largest}}) \)，但不超过 N-1 可行上限。

### 11. **N-1 安全约束**

#### (a) **发电机 N-1**
- 任意一台机组故障退出后，剩余机组最大出力仍需满足：
  \[
  \sum_{j \ne i} P_{\max,j} \cdot u_{j,t} \ge D_t + R_t, \quad \forall i, t
  \]
  （若启用 `--relax-n1`，则右侧仅保留 \( D_t \)）

#### (b) **线路 N-1（简化处理）**
- **单线潮流限制**：任一线路潮流不超过总负荷的 80%：
  \[
  |P^{\text{flow}}_{l,t}| \le 0.8 \cdot D_t, \quad \forall l, t
  \]
- **总传输容量冗余**：移除任意一条线路后，剩余线路总容量仍 ≥ \( D_t \)

### 12. **（可选）最小惯量约束**

- 若启用 `--enable-inertia`，则系统惯量需满足：
  \[
  \sum_i H_i P_{\max,i} u_{i,t} \ge H_{\min} \cdot \sum_i P_{\max,i} u_{i,t}, \quad \forall t
  \]
  其中 \( H_{\min} = \frac{\Delta P}{2 \cdot \text{ROCOF}_{\text{set}}} = \frac{600}{2 \times 0.5} = 600 \)

> 注：该式在代码中写作 \( \text{total\_inertia} \ge H_{\min} \cdot \text{total\_capacity} \)

---

## 六、补充说明

- **初始状态**：由 `Initial_Up_Time` 决定（>0 表示初始在线）
- **备用与 N-1 耦合**：备用需求 \( R_t \) 的设定考虑了 N-1 可行性
- **负荷分配**：虽未明确指定，脚本采用等比例分配至 24 个负荷节点
- **可视化输出**：包含启停状态、出力曲线、备用、线路潮流、电压角等 8 类图表

