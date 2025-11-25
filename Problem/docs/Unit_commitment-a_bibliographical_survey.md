# Unit Commitment—A Bibliographical Survey

**Abstract—With the fast-paced changing technologies in the power industry, new power references addressing new technologies are coming to the market. So there is an urgent need to keep track of international experiences and activities taking place in the field of modern unit-commitment (UC) problem. This paper gives a bibliographical survey, mathematical formulations, and general backgrounds of research and developments in the field of UC problem for past 35 years based on more than 150 published articles. The collected literature has been divided into many sections, so that new researchers do not face any difficulty in carrying out research in the area of next-generation UC problem under both the regulated and deregulated power industry.**

**Index Terms—Ant colony systems, artificial neural networks, branch and bound, deregulation, dynamic programming, enumeration, evolutionary computation, expert system, fuzzy logic, genetic algorithms, hybrid models, integer programming, interior point, Lagrangian Relaxation, linear programming, priority list, simulated annealing, tabu search, unit commitment.**

---

### I. INTRODUCTION

MANY utilities have daily load patterns which exhibit extreme variation between peak and offpeak hours because people use less electricity on Saturday than on weekdays, less on Sundays than on Saturdays, and at a lower rate between midnight and early morning than during the day [5], [13], [64], [66], [104]. If sufficient generation to meet the peak is kept on line throughout the day, it is possible that some of the units will be operating near their minimum generating limit during the offpeak period. The problem confronting the system operator is to determine which units should be taken offline and for how long.

In most of the interconnected power systems, the power requirement is principally met by thermal power generation. Several operating strategies are possible to meet the required power demand, which varies from hour to hour over the day. It is preferable to use an optimum or suboptimum operating strategy based on economic criteria. In other words, an important criterion in power system operation is to meet the power demand at minimum fuel cost using an optimal mix of different power plants. Moreover, in order to supply high-quality electric power to customers in a secured and economic manner, thermal unit commitment (UC) is considered to be one of best available options. It is thus recognized that the optimal UC of thermal systems, which is the problem of determining the schedule of generating units within a power system, subject to device and operating constraints results in a great saving for electric utilities. So the general objective of the UC problem is to minimize system total operating cost while satisfying all of the constraints so that a given security level can be met [61], [77], [139].

This paper summarizes different methods used in the UC problem-solving technique. It also presents a direction on which the new solution techniques evolve with time. 

### II. GENERAL BACKGROUND AND CONCEPTS

Various approaches have been developed to solve the optimal UC problem. These approaches have ranged from highly complex and theoretically complicated methods to simple rule-of-thumb methods. The scope of operations scheduling problem will vary strongly from utility to utility depending on their mix of units and particular operating constraints [11], [35], [47], [65], [72], [75].

The economic consequences of operation scheduling are very important. Since fuel cost is a major cost component, reducing the fuel cost by little as 0.5% can result in savings of millions of dollars per year for large utilities [15], [110].

A very important task in the operation of a power system concerns the optimal UC considering technical and economical constraints over a long planning horizon up to one year. The solution of the exact long-term UC [73], [118], [138] is not possible due to exorbitant computing time and, on the other hand, the extrapolation of short-term UC to long-term period is inadequate because too many constraints are neglected such as maintenance time and price increases, etc.

Energy management systems have to perform more complicated and timely system control functions to operate a large power system reliably and efficiently. In the case of a power pool that involves several generation areas interconnected by tie line [111], the objective is to achieve the most economical

generation policy that could supply the local demands without violating tie-line capacity constraints [16], [20], [76]. Although the thermal and hydro thermal UC of a single area has been studied extensively, the multiarea generation schedule has not been given enough attention. The available literature for the UC involving multiareas reveals that scheduling should be considered together with a viable economic dispatch to preserve the tie-line constraints.

In the past, demand forecast advised power system operators of the amount of power that needed to be generated [117]. But under partially or fully deregulated environment, in the future, bilateral spot and forward contracts will make part of the total demand known *a priori* [23]. The remaining part of the demand will be predicted as in the past. However, the generating companies (GENCOs) share of this remaining demand may be difficult to predict since it will depend on how its price compares to that of other suppliers. The GENCO’s price will depend on the prediction of its share of this remaining demand as that will determine how many units they have switched on. The UC schedule directly affects the average cost and indirectly the price, making it an essential input to any successful bidding strategy. There may be a tendency to think that maximizing the profit is essentially the same as minimizing the cost. This is not necessarily the case. We have to remember that since we no longer have the obligation to serve the demand, the GENCOs may choose to generate less than the demand. This allows a little more flexibility and makes the problem complex in the UC schedules under the deregulated environment. Finally, the profit depends, not only on the cost, but also on revenue. If revenue increases more than the cost does, the profit will increase. So for the next-generation UC problem, researchers have to still play an important role.

If the bid functions are nonconvex or nondifferentiable in nature, which is commonly seen in both regulated and deregulated power industry, then the above problem becomes complex. Further, the complexity increases if the competition is encouraged in both suppliers and buyers side including emission constraints. So it has been observed that the hybrid models, which are the combination of both classical and nonclassical methods, can handle the present day complex UC problem commonly seen within developed countries.

With the available standard software products, electric utilities have to enhance, evolve, and upgrade or add new applications such as UC solutions for modern deregulated power industry in conjunction with energy management systems [41], [44], [85].

### III. UC UNDER DEREGULATED POWER INDUSTRY

Since the mid-1980s, the electrical power-supply industry around the world has experienced a period of rapid and critical changes regarding the way electricity is generated, transmitted, and distributed. The need for more efficiency in power production and delivery has led to privatization, restructuring, and, finally, deregulation of the power sectors in several countries traditionally under control of federal and state governments. Many countries like England, the U.S., Canada, Australia, New Zealand, Chile, Argentina, Peru, Colombia, and Scandinavian are already exercising with the deregulated electricity industry. Though there have been some pitfalls here and there, the end users of the system are enjoying the fruits of the deregulated electricity industry tree. So it is the high time for both the developed and developing countries to modify or replace their traditional algorithms based on the requirements of the modern power industry.

In any restructured or deregulated power industry, the pool implements a power action based on a UC model. Suppliers submit their bids to supply the forecasted daily inelastic demand [12]. Each bid consists of a cost function and a set of parameters that define the operative limits of the generating unit. After the pool solves the UC problem, the system marginal price is determined for each time period. The system marginal price is nothing but the maximum average cost among the scheduled generators. Several scheduling and pricing concerns have been raised with the use of UC models to conduct power pool auctions [99], [127]. It is reported that the cost minimization model does not always lead to lower prices when they are defined as maximum average costs. Cost suboptimal solutions that result in lower prices may exist and, therefore, the applicability of cost minimization UC models for power pool auctions is questioned.

Chattopadhyay *et al.* [26] presented a model, capable of performing the following tasks: generation scheduling, interutility transmission scheduling, and nonutility generation purchase planning, etc. It is required to update the UC algorithm as the electric industry restructures. In [23], a price/profit based UC problem has been formulated which considers the softer demand constraint and allocates fixed and transitional costs to the scheduled hours. In August 2001, M. Madrigal *et al.* [82] investigated the existence, determination, and effects of competitive market equilibrium for UC power pool auctions to avoid the conflict of interest and revenue deficiency. New formulations to the UC problems suitable for an electric power producer in an deregulated market has been provided by Valenzuela *et al.* [60] and Larsen *et al.* [129] in 2001.

### IV. UC PROBLEM FORMULATION

The generic unit commitment problem can be formulated as

**Minimize Operational Cost (OC)**

$$
OC = \sum_{i=1}^{N} \sum_{t=1}^{T} FC_{it}(P_{it}) + MC_{it}(P_{it}) + ST_{it} + SD_{it} \: \$/\text{hr} \tag{1.1}
$$

where $FC_{it}(P_{it})$ (Fuel cost) is the input/output(I/O) curve that is modeled with a curve (normally quadratic).

$$
FC_{it}(P_{it}) = a_i * {P_{it}}^2 + b_i * P_{it} + c_i \: \$/\text{hr} \tag{1.2}
$$

$a_i$, $b_i$, and $c_i$ are the cost coefficients.
The maintenance cost ($MC_i(P_i)$) is described by

$$
MC_{it}(P_{it}) = BM_{it} + IM_{it} * P_{it} \: \$/\text{hr} \tag{1.3}
$$

where $BM_i$ is the base maintenance cost, and $IM_i$ is the incremental maintenance cost.
The startup cost ($ST_{it}$) is described by

$$
ST_{it} = TS_{it} + (1 - e^{(D_{it}/AS_{it})})BS_{it} + MS_{it} \: \$/\text{hr} \tag{1.4}
$$

TS$_{it}$ turbine startup cost;  
BS$_{it}$ boiler startup cost;  
MS$_{it}$ startup maintenance cost;  
D$_{it}$ number of hours down;  
AS$_{it}$ boiler cool down coefficient.

Similarly, the shut-down cost (SD$_{it}$) is described by

$$
\text{SD}_{it} = \text{KP}_{it} \quad \$/\text{hr} \tag{1.5}
$$

where K is the incremental shut-down cost.

**Subject to the following constraints:**

- **Minimum up-time**  
  a unit must be ON for a certain number of hours before it can be shut off;

- **Minimum downtime**  
  a unit must be OFF for a certain number of hours before it can be brought online;

- **maximum and minimum output limits on generators**  
  $$
  \text{P}_{it}^{\text{min}} \le \text{P}_{it} \le \text{P}_{it}^{\text{max}}.
  $$

- **Ramp rate**  
  $$
  \nabla\text{P}_{it} \le \nabla\text{P}_{it}^{\text{max}}.
  $$

- **Power balance**  
  $$
  \sum_{i=1}^{N}(U_{it}.P_{it}) = D_{t}^{f} + \text{losses}.
  $$

- **Must run units**  
  these units include prescheduled units which must be online, due to operating reliability and/or economic considerations;

- **Must out units**  
  units which are on forced outages and maintenance are unavailable for commitment;

- **Spinning reserve**  
  spinning reserve requirements are necessary in the operation of a power system if load interruption is to be minimal. This necessity is due partly to certain outages of equipment. Spinning reserve requirements may be specified in terms of excess megawatt capacity or some form of reliability measures;

- **Crew constraints**  
  certain plants may have limited crew size which prohibits the simultaneous starting up and/or shutting down of two or more units at the same plant. Such constraints would be specified by the times required to bring a unit online and to shut down the unit.

Redefining the UC problem for the deregulated environment [23] involves changing the demand constraints from an equality to less than or equal, and changing the objective function from cost minimization to profit (revenue-operational cost) maximization. Now the generic UC problem under deregulated environment can be formulated as

**Maximize Profit (P)**

$$
P = \sum_{i=1}^{N}\sum_{t}^{T}(P_{it}.f_{pt}).U_{it} - OC. \tag{1.6}
$$

**Subject to the following constraints:**

**New Power balance**

$$
\sum_{i=1}^{N}(U_{it}.P_{it}) \le D_{t}^{f} \tag{1.7}
$$

where

- $N$ number of units;
- $T$ number of time periods;
- $f_{pt}$ forecasted price for for period t;
- $U_{it}$ up/down status of unit i;
- $P_{it}$ power generation of unit i during time period t;
- $D_{t}^{f}$ forecasted demand during time period t.

Reserve power and transmission losses are as per contract and the rest of the constraints are the same as generic UC problem.

Similarly, network-constrained UC problem under both regulated and deregulated environment can be extended by incorporating the following system constraint parallel with (1.1) and (1.6)

- **Power-flow equation of the power network**  
  $$
  g(V,\phi) = 0
  $$

  where

  $$
  g(V,\phi) = 
  \begin{cases}
  P_i(V,\phi) - P_i^{net} & \} \leftarrow \text{For each PQ bus i} \\
  Q_i(V,\phi) - Q_i^{net} & \} \leftarrow \text{For each PV bus} \\
  P_m(V,\phi) - P_m^{net} & \} \leftarrow \text{m, not including the ref. bus.}
  \end{cases}
  $$

  where

  - $P_i$ and $Q_i$ respectively calculated real and reactive power for PQ bus i;
  - $P_i^{net}$ and $Q_i^{net}$ respectively specified real and reactive power for PQ bus i;
  - $P_m$ and $P_m^{net}$ respectively calculated and specified real power for PV bus m;
  - $V$ and $\phi$ voltage magnitude and phase angles at different buses.

- **The inequality constraint on reactive power generation $Qg_i$ at each PV bus**  
  $$
  Qg_i^{\text{min}} \le Qg_i \le Qg_i^{\text{max}}
  $$

  where $Qg_i^{\text{min}}$ and $Qg_i^{\text{max}}$ are, respectively, minimum and maximum value of reactive power at PV bus $i$.

- **The inequality constraint on voltage magnitude V of each PQ bus**  
  $$
  V_i^{\text{min}} \le V_i \le V_i^{\text{max}}
  $$

  where $V_i^{\text{min}}$ and $V_i^{\text{max}}$ are, respectively, minimum and maximum voltage at bus $i$.

- **The inequality constraint on phase angle $\phi_i$ of voltage at all of the buses $i$**  
  $$
  \phi_i^{\text{min}} \le \phi_i \le \phi_i^{\text{max}}
  $$

  where $\phi_i^{\text{min}}$ and $\phi_i^{\text{max}}$ are, respectively, minimum and maximum voltage angles allowed at bus $i$.

- **MVA flow limit on transmission line**  
  $$
  MVA_{fij} \le MVA_{fij}^{\text{max}}
  $$

  where $MVA_{fij}^{\text{max}}$ is the maximum rating of transmission line connecting bus $i$ and $j$.

### V. METHODOLOGIES AND ANALYSIS

#### A. Exhaustive Enumeration

The UC problem has been earlier solved by enumerating all possible combinations of the generating units and then the combinations that yield the least cost of operation are chosen as the optimal solution. In [104], Kerr, *et al.*, and in [66], Hara, *et al.*, solved the UC problem successfully including Florida Power Corporation by using the exhaustive enumeration method. Even though the method was not suitable for a large size electric utility, it was capable of providing an accurate solution.

#### B. Priority Listing

Priority listing method initially arranges the generating units based on lowest operational cost characteristics. The predetermined order is then used for UC such that the system load is satisfied. Burns *et al.* [105] and Lee [38] handled the UC problem, using priority order. Shoults, *et al.* [109] presented a straightforward and computationally efficient algorithm using priority order including import/export constraints. Lee [36] and Lee *et al.* [34] solved the single and multiarea UC problem using priority order based on a classical index.

#### C. Dynamic Programming

Stated in power system parlance, the essence of dynamic programming is for the total running cost of carrying $x$ megawatt (MW) of load on $N$ generating units to be a minimum, the load $y$ MW carried by unit $N$ must be such that the remaining load of $(x - y)$ MW is carried by the remaining $(N - 1)$ units also at minimum cost. In mathematical form $F_N(x) = \text{Min}[g_N(y) + f_{N-1}(x - y)]$ where

- $F_N(x)$: minimum running cost of carrying $x$ MW load on $N$ generating units;
- $g_N(y)$: cost of carrying $y$ MW load on unit $N$;
- $f_{N-1}(x - y)$: minimum cost of carrying the remaining $(x - y)$ MW load on the remaining $(N - 1)$ units.

Dynamic programming was the earliest optimization-based method to be applied to the UC problem. It is used extensively throughout the world. It has the advantage of being able to solve problems of a variety of sizes and to be easily modified to model characteristics of specific utilities [107], [136]. It is relatively easy to add constraints that affect operations at an hour since these constraints mainly affect the economic dispatch and solution method [33]. It is more difficult to include constraints [57] that affect a single-units operation over time. The disadvantage of the dynamic programming are its requirement to limit the commitments considered at any hour and its suboptimal treatment of minimum up and downtime constraints and time-dependent startup costs [22].

In [98], Lowery discussed the practical applicability of dynamic programming for UC solutions. In 1971, Happ [46] reported the advantages of personal-computer solutions over manual commitment solutions and claimed that the savings obtained are in excess of 1% of the total fuel cost which translates into U.S. $7000 for a 100-machine system. Pang *et al.* [14] compared the performance of four UC methods, three of which are based on the dynamic programming approach.

#### D. Integer and Linear Programming

Dillon *et al.* [130] developed an integer programming method for practical size scheduling problem based on the extension and modification of the branch-and-bound method. The UC problem can be partitioned into a nonlinear economic dispatch problem and a pure integer nonlinear UC problem based on benders approach. Whereas the mixed integer programming approach solves the UC problem by reducing the solution search space through rejecting infeasible subsets. A linear programming UC problem can be solved either by decomposing the whole problem into subproblems with help of Dantzig–Wolfe decomposition principle and then each subproblem is solved using linear programming or the problem can be solved directly by revised simplex technique [44].

#### E. Branch and Bound

Lauer *et al.* [42] and Cohen *et al.* [10] presented a new approach for solving UC problem based on branch-and-bound method, which incorporates all time-dependent constraints and does not require a priority ordering of units. In [74], Huang *et al.* proposed a constraint logic programming along with the branch-and-bound technique to provide an efficient and flexible approach to the UC problem.

The branch-and-bound procedure consists of the repeated application of the following steps. First, that portion of the solution space (i.e., set of decision variables under consideration) in which the optimal solution is known to lie is partitioned into subsets. Second, if all of the elements in a subset violate the constraints of the minimization problem, then that subset is eliminated from further consideration (fathomed). Third, an upper bound on the minimum value of the objective function is computed. Finally, lower bounds are computed on the value of the objective function when the decision variables are constrained to lie in each subset still under consideration. A subset is then fathomed if its lower bound exceeds the upper bound of the minimization problem, since the optimal decision variable cannot lie in that subset. Convergence takes place when only one subset of decision variables remains, and the upper and lower bounds are equal for that subset.

#### F. Lagrangian Relaxation

Based on Lagrangian Relaxation approach, the UC problem can be written in terms of 1) a cost function that is the sum terms each involving a single unit, 2) a set of constraints involving a single unit, and 3) a set of coupling constraints (the generation and reserve constraints), one for each hour in the study period, involving all of the units. Formally, we can write the UC problem as follows [30], [86], [126], [128]:

**Minimize**
$$
\sum_{t} \sum_{i} OC(G_i(t), U_i(t)).
$$

**Subject to the unit constraints** $L_i(G_i, U_i) \le 0$.

For all units I, where $G_i = (G_i(1), \dots, G_i(T))$ and $U_i = (U_i(1), \dots, U_i(T))$; and the coupling generation and reserve constraints

$$
\sum_{i} R_{i,n}(G_i(t), U_i(t)) \ge \text{Re } s - \text{Re } q(t).
$$

For all times t and requirement n. Everett showed that an approximate solution to this problem can be obtained by adjoining the coupling constraints onto the cost using Lagrange multipliers. The resulting “relaxed” problem is to minimize the so-called Lagrangian subject to the unit constraints

$$
D(G_i, U_i) = \sum_{t} \sum_{i} \left[ OC(G_i(t), U_i(t)) - \sum_{n} \lambda_n(t) R_{i,n}(G_i(t), U_i(t)) \right]
$$

where $\lambda_n(t)$ are the multipliers associated with the $n$th requirement for time t. Describing the Lagrangian Relaxation method requires answering the following questions: 1) how do we find the multipliers $\lambda_n(t)$ so that the solution to the relaxed problem is near the optimum, 2) how close to the optimum is the solution, and 3) how do we solve the relaxed problem? Everett and dual theory together provides us the insight to the above questions and solutions.

Lagrangian Relaxation is also being used regularly by some utilities [9], [106], [112]. Its utilization in production UC programs is much more recent than the dynamic programming. It is much more beneficial for utilities with a large number of units since the degree of suboptimality goes to zero as the number of units increases. It has also the advantage of being easily modified to model characteristics of specific utilities. It is relatively easy to add unit constraints. The main disadvantage of Lagrangian Relaxation is its inherent suboptimality.

In [7], Merlin *et al.* proposed a new method for UC using Lagrangian Relaxation method and validated at Electricite De France. Aoki *et al.* [62], [63] applied Lagrangian Relaxation method for a large-scale optimal UC problem, which includes three types of units such as usual thermal units, fuel-constrained thermal units, and pumped storage hydro units. A three-phase new Lagrangian Relaxation algorithm for UC is proposed by Zhuang *et al.* [32]. In the first phase, the Lagrangian dual of the UC is maximized with standard subgradient technique, the second phase finds a reserve feasible dual solution, and followed by a third phase of economic dispatch. Wang *et al.* [17] presented a rigorous mathematical method for dealing with the ramp rate limits in UC and the rotor fatigue effect in economic scheduling. Ma *et al.* [52] incorporated optimal power flow in the UC formulation. Using benders decomposition, the formulation is decomposed into a master problem and a subproblem. The master problem solves UC with prevailing constraints, except transmission security and voltage constraints, by augmented Lagrangian Relaxation. The refinement or reduction in complexity of Lagrangian Relaxation method has also been suggested by Takriti [120] and Cheng [24].

#### G. Interior Point Optimization

Interior point methods have not only been used to successfully solve very large linear and nonlinear programming problems, but also to solve combinatorial and nondifferentiable problems. The interior point method has now been applied to solve scheduling problems in electric power systems. Madrigal *et al.* [81] applied the interior point method for solving the UC problem, based on his observation of the interior-point method to have two main advantages such as they have better convergence characteristics and they do not suffer with parameter tuning.

#### H. Tabu Search

Tabu search is a powerful optimization procedure that has been successfully applied to a number of combinatorial optimization problems. It has the ability to avoid entrapment in local minima by employing a flexible memory system [1]. Mori *et al.* [48], [149] presented an algorithm, embedding the priority list into tabu search for unit commitment. Rajan *et al.* [148] solved UC problem using neural-based tabu search method. In [150], Lin *et al.* developed an improved tabu search algorithm for solving economic dispatch problems. Mantawy *et al.* [1], [3] presented UC solutions using tabu search and also solved long-term hydroscheduling problems very efficiently using a new tabu search algorithm [151].

#### I. Simulated Annealing

Simulated annealing was independently introduced by Kirkpatrick, Gela, and Vecchi in 1982 and Cerny in 1985. Annealing, physically, refers to the process of heating up a solid to a high temperature followed by slow cooling achieved by decreasing the temperature of the environment in steps [68], [133]. By making an analogy between the annealing process and the optimization problem, a great class of combinatorial optimization problems can be solved following the same procedure of transition from equilibrium state to another, reaching minimum energy of the system. In solving the UC problem, two types of variables need to be determined. The unit status (binary) variable U and V and the units output power (continuous) variables, P. The problem can be decomposed into two subproblems, a combinatorial optimization problem in U and V and a nonlinear optimization problem in P. So, simulated annealing can be suitably used to solve the UC problem. Mantawy *et al.* [2] presented a simulated annealing algorithm to solve the UC problem and concluded that even though simulated annealing algorithm has the disadvantage of taking long CPU time, it has other strong features like being independent of the initial solution and mathematical complexity.
