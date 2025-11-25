# Quantum-based Optimization for Unit Commitment Problems in Power Systems

Unit commitment (UC) is a fundamental optimization problem in power system operation. Its goal is to determine the on/off status and generation levels of thermal units over a given time horizon to minimize operating costs while satisfying demand and system constraints. The UC problem is known to be a large-scale mixed integer programming (MIP) challenge, whose computational complexity grows exponentially with system size.

With the emergence of quantum optimization technologies such as the coherent Ising Machine (CIM), the UC problem can be reformulated as a quadratic unconstrained binary optimization (QUBO) model. This transformation enables the use of quantum-inspired solvers for fast and parallel search of optimal solutions, paving the way for real-time power system operation on quantum hardware.

We invite participants to bridge classical power system optimization with quantum computing through a three-stage workflow:

1. building a classical unit commitment model;
2. transforming it into a QUBO formulation;
3. designing a scalable reduction strategy to accommodate current quantum

---

## Problem Description

Participants are required to construct a complete UC optimization framework that integrates classical and quantum methods.

The classical UC formulation is typically based on standard economic dispatch principles, including fuel costs, startup/shutdown costs, and operational constraints.

Padhy et al. [1] provides a comprehensive survey of UC modeling and solution approaches. In addition, Mehta et al. [2] analyze the computational hardness of QUBO problems, offering insights relevant for mapping and solving UC problems in combinatorial optimization frameworks.

Participants are required to construct a complete UC optimization framework that integrates both classical and quantum optimization methods. All generator, network, and load parameters required for UC modeling are provided in Tables 1-5. Using these data, participants must complete the following four tasks in sequence, and are allowed to simplify network and unit parameters as needed when the QUBO matrix is beyond the bit-width limitations of the target quantum hardware.

Participants are tasked with a multi-step process:

1. Constructing and solving a classical UC model.

2. Extending this model with network and security constraints.

3. Transforming the complete UC formulation into a QUBO model and solving it using quantum-inspired methods.

4. Designing a reduction strategy to ensure the problem scale complies with qubit constraints.

Each task below is weighted according to its difficulty and importance in the quantum optimization workflow.

---

## Problem 1. Classical UC Modeling and Optimization

Formulate and solve a classical Unit Commitment model using conventional optimization tools (e.g., Gurobi or CPLEX). To ensure consistency and fairness across teams, the classical UC model must include at least the following components:

**1. Objective Function**
* **(1) Fuel cost:** A quadratic or piecewise-linear production cost for each generating unit.
* **(2) Startup/shutdown cost:** A cost linked to unit startup/shutdown transitions.

**2. Constraints**
* **(1) Power balance constraint:** Total generation must match the system load in each time period.
* **(2) Generation limits:** Each unit must operate within its minimum and maximum output limits whenever committed.
* **(3) Minimum up/down time constraints:** After being turned on, a unit must remain online for a minimum number of periods; after shutdown, it must remain offline for a minimum number of periods.
* **(4) Ramp-rate constraints:** The change in unit output between consecutive periods must not exceed ramp-up and ramp-down limits.
* **(5) Startup/shutdown logic constraints:** Logical relationships ensuring that commitment status, startup indicators, and shutdown indicators are consistent.

---

## Problem 2. UC Modeling with Network and Security Constraints

Based on the classical UC formulation developed in Problem 1, participants shall further enhance the model by incorporating system-level network and security constraints. In this task, the UC model must be extended to include:

* **(1) Network power flow constraints:** Such as DC power flow limits on all transmission lines.
* **(2) N-1 security constraints:** Ensuring that the system remains operationally feasible under any single generator or transmission line outage.
* **(3) Spinning-reserve requirements:** Guaranteeing sufficient reserve capacity in each time period to maintain reliability.

**The following constraint is not mandatory for completing Problem 2:**
* **(4) Minimum safety inertia constraint (optional):** Enforcing a system-wide minimum inertia requirement using from the reference [3]. Only units that are already online in each period contribute to the inertia ($H$).

These additional constraints aim to bring the UC model closer to real-world operational conditions and increase its modeling depth and complexity. Participants shall use the same optimization solver adopted in Problem 1 (e.g., Gurobi or CPLEX) to solve this extended UC model, and are required to compare the results of Problem 1 and Problem 2.

---

## Problem 3. QUBO Transformation and Kaiwu SDK Quantum Solution

Transform the UC model in Problem 2 into a QUBO representation and solve it using the Kaiwu SDK. Participants are required to compare the quantum-based solutions with the traditional optimization results.

---

## Problem 4. Problem Scale Reduction under Quantum Hardware Constraints

Participants shall design and implement an effective problem-scale reduction strategy specifically for the Problem 2 UC model (or corresponding QUBO model in Problem 3), whose QUBO representation is typically too large to be executed directly on current quantum hardware. The reduction method should ensure that, under the bit-capacity limitation of the CIM (e.g., XXX bits), the resulting QUBO can be successfully solved.

---

## Data Tables

### Table 1: Parameters of each unit (Part A)

| Unit (bus) | Maximum power generation (MW) | Minimum power generation (MW) | Minimum Up Time (h) | Startup Cost ($) | Shutdown Cost ($) | Ramp-Up Limit (MW/h) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 300 | 50 | 8 | 180 | 180 | 80 |
| 2 | 180 | 20 | 8 | 180 | 180 | 80 |
| 5 | 50 | 15 | 5 | 40 | 40 | 50 |
| 8 | 35 | 10 | 5 | 60 | 60 | 50 |
| 11 | 30 | 10 | 6 | 60 | 60 | 60 |
| 13 | 40 | 12 | 3 | 40 | 40 | 60 |

### Table 2: Parameters of each unit (Part B)

| Unit (bus) | Ramp-Down Limit (MW/h) | Minimum Up Time (h) | Minimum Down Time (h) | Initial $U_P$ Time (h) | Initial Down Time (h) | a | b | c | H |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 80 | 5 | 3 | 5 | 0 | 0.02 | 2.00 | 0 | 7.0 |
| 2 | 80 | 4 | 2 | 4 | 0 | 0.0175 | 1.75 | 0 | 4.5 |
| 5 | 50 | 3 | 2 | 3 | 0 | 0.0625 | 1.00 | 0 | 4.5 |
| 8 | 50 | 3 | 2 | 3 | 0 | 0.00834 | 3.25 | 0 | 3.2 |
| 11 | 60 | 1 | 1 | 3 | 0 | 0.025 | 3.00 | 0 | 3.0 |
| 13 | 60 | 4 | 2 | 4 | 0 | 0.025 | 3.00 | 0 | 3.0 |

### Table 3: Grid parameters

| Branch number | From bus | To Bus | R* | X | b | Max power transmission (MW) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1 | 2 | 0.02 | 0.06 | 0.03 | 650 |
| 2 | 1 | 3 | 0.05 | 0.19 | 0.02 | 650 |
| 3 | 2 | 4 | 0.06 | 0.17 | 0.02 | 325 |
| 4 | 3 | 4 | 0.01 | 0.04 | 0 | 650 |
| 5 | 2 | 5 | 0.05 | 0.20 | 0.02 | 650 |
| 6 | 2 | 6 | 0.06 | 0.18 | 0.02 | 325 |
| 7 | 4 | 6 | 0.01 | 0.04 | 0 | 450 |
| 8 | 5 | 7 | 0.05 | 0.12 | 0.01 | 350 |
| 9 | 6 | 7 | 0.03 | 0.08 | 0.01 | 650 |
| 10 | 6 | 8 | 0.01 | 0.04 | 0 | 160 |
| 11 | 6 | 9 | 0 | 0.21 | 0 | 325 |
| 12 | 6 | 10 | 0 | 0.56 | 0 | 160 |
| 13 | 9 | 11 | 0 | 0.21 | 0 | 325 |
| 14 | 9 | 10 | 0 | 0.11 | 0 | 325 |
| 15 | 4 | 12 | 0 | 0.26 | 0 | 325 |
| 16 | 12 | 13 | 0 | 0.14 | 0 | 325 |
| 17 | 12 | 14 | 0.12 | 0.26 | 0 | 160 |
| 18 | 12 | 15 | 0.07 | 0.13 | 0 | 160 |
| 19 | 12 | 16 | 0.09 | 0.20 | 0 | 160 |
| 20 | 14 | 15 | 0.22 | 0.20 | 0 | 80 |
| 21 | 16 | 17 | 0.08 | 0.19 | 0 | 80 |
| 22 | 15 | 18 | 0.11 | 0.22 | 0 | 80 |
| 23 | 18 | 19 | 0.06 | 0.13 | 0 | 80 |
| 24 | 19 | 20 | 0.03 | 0.07 | 0 | 80 |
| 25 | 10 | 20 | 0.09 | 0.21 | 0 | 80 |
| 26 | 10 | 17 | 0.03 | 0.08 | 0 | 80 |
| 27 | 10 | 21 | 0.03 | 0.07 | 0 | 80 |
| 28 | 10 | 22 | 0.07 | 0.15 | 0 | 80 |
| 29 | 21 | 22 | 0.01 | 0.02 | 0 | 160 |
| 30 | 15 | 23 | 0.10 | 0.20 | 0 | 160 |
| 31 | 22 | 24 | 0.12 | 0.18 | 0 | 160 |
| 32 | 23 | 24 | 0.13 | 0.27 | 0 | 160 |
| 33 | 24 | 25 | 0.19 | 0.33 | 0 | 80 |
| 34 | 25 | 26 | 0.25 | 0.38 | 0 | 80 |
| 35 | 25 | 27 | 0.11 | 0.21 | 0 | 80 |
| 36 | 27 | 28 | 0 | 0.40 | 0 | 80 |
| 37 | 27 | 29 | 0.22 | 0.42 | 0 | 80 |
| 38 | 27 | 30 | 0.32 | 0.60 | 0 | 80 |
| 39 | 29 | 30 | 0.24 | 0.45 | 0 | 80 |
| 40 | 8 | 28 | 0.06 | 0.20 | 0.02 | 160 |
| 41 | 6 | 28 | 0.02 | 0.06 | 0.01 | 160 |

*\*Note: Column "R" is inferred based on standard grid parameter formats $(R, X, B)$ and the provided data.*

### Table 4: Load demands of 24 periods

| Time period (h) | Load demand (MW) |
| :--- | :--- |
| 1 | 166 |
| 2 | 196 |
| 3 | 229 |
| 4 | 257 |
| 5 | 263 |
| 6 | 252 |
| 7 | 246 |
| 8 | 213 |
| 9 | 192 |
| 10 | 161 |
| 11 | 147 |
| 12 | 160 |
| 13 | 170 |
| 14 | 185 |
| 15 | 208 |
| 16 | 232 |
| 17 | 246 |
| 18 | 241 |
| 19 | 236 |
| 20 | 225 |
| 21 | 204 |
| 22 | 182 |
| 23 | 161 |
| 24 | 131 |

### Table 5: Inertia-related security parameters

| ROCOF (Hz/s) | F | load step (MW) |
| :--- | :--- | :--- |
| 0.5 | 2.118 | 600 |

---

## References

[1] Padhy, Narayana Prasad. "Unit commitment-a bibliographical survey." *IEEE Transactions on Power Systems*. 19.2 (2004): 1196-1205.  

[2] Mehta V, Jin F, Michielsen K, et al. "On the hardness of quadratic unconstrained binary optimization problems[J]." *Frontiers in Physics*, 2022, 10: 956882.  

[3] Xia W, Wang Z, Wang J, et al. "Evaluation of minimum safety inertia of new energy power system considering rate of change of frequency [C]//2022 12th International Conference on Power and Energy Systems (ICPES)." *IEEE*, 2022: 618-622.  


[4] Kaiwu SDK Documentation and Resources  
    - SDK Documentation