# Evaluation of Minimum Safety Inertia of New Energy Power System Considering Rate of Change of Frequency
2022 the 12th International Conference on Power and Energy Systems

## Abstract
Due to the rapid development of the new energy power system, the increase of new energy permeability, and the grid-connected new energy inverter does not have the inertia support ability, so the frequency stability of the system is seriously affected, and the frequency indicators are gradually close to the safety boundary. In order to solve this problem, this paper first summarizes the inertia index of conventional units, and combs out the effective calculation method of inertia of conventional units. Secondly, this method is extended to the inertia estimation of the new energy power system to evaluate the inertia of the new energy power system. Then, considering the minimum safety inertia evaluation of new energy power system constrained by the rate of change of frequency, the minimum safety inertia coefficient of new energy is defined and taken as the evaluation index. Finally, the IEEE 10-machine 39-node New England Power system model is used to verify the effectiveness of the proposed method.

**Keywords**: new energy power system, inertia estimation, Rate of change of frequency, minimum safety inertia

## I. INTRODUCTION
At the UN General Assembly in 2020, China proposed to achieve "carbon peaking" by 2030 and "carbon neutrality" by 2060. In order to achieve the above dual-carbon goal, the construction of a high-proportion new energy power system has become the primary task at present [1]. Since most of the new energy units do not have the inertia support capability of the synchronous units, with the gradual increase of new energy permeability, the level of inertia of the system continuously decreases and presents the characteristics of spatial and temporal distribution. Coupled with the maximum power tracking control of the new energy power system, the frequency modulation capability and power response of the system are weakened.

At present, some foreign power grids have experienced frequency problems caused by the reduction of inertia due to the access of a high proportion of new energy. For example, in the "8.9" power failure in the UK [2], a series of failures led to the system power deficit exceeding the maximum frequency adjustment capacity of the grid, and finally triggered the low-frequency load reduction. Due to the relative reduction of inertia of the new energy power system, the rate of change of frequency of the system under disturbance increases and the lowest frequency decreases [3]. In order to make the rate of change of frequency after disturbance lower than the set value of the system frequency protection, the new energy power system needs to maintain sufficient inertia. The minimum safety inertia is the minimum inertia required by the frequency of power system to maintain safe and stable operation. Moreover, the minimum safety inertia is the key factor affecting the frequency problem, so it is urgent to solve the evaluation problem of the minimum safety inertia.

In order to solve the problem caused by the relatively reduced inertia support to the frequency of power grid, a large number of literatures have carried out relevant analysis and research on this problem. Literature [4] uses synchronous phasor measurement technology to measure the frequency transformation rate at the initial moment of grid disturbance to evaluate the system inertia, but this literature does not evaluate the minimum safety inertia constraint. Literature [5] evaluated the inertia of the power system by defining the concept of the inertia graph center, but did not specify the relationship between the inertia of the system and the rate of change of frequency. In literature [6], the frequency response of the system under different inertia levels was obtained by simulation. Through comparative analysis, it was found that the lowest frequency point had very obvious nonlinear characteristics with the system inertia, but its mathematical model was not sought.

From the perspective of power system power response, this paper summarizes the calculation method of inertia of conventional units, and then analyzes the calculation method of inertia of high proportion new energy power system. Based on the swing equation of synchronous motor, the mathematical model is deduced and solved. Combined with the inertia characteristics of high proportion new energy power system under different permeability, the minimum safety inertia evaluation method of the system with the constraint of rate of change of frequency is proposed. The ratio of the minimum safety inertia constraint is designed to judge the stability of the system power disturbance, and the proposed method is verified by simulation.

## II. INERTIAL CHARACTERISTIC INDEX OF POWER SYSTEM
The inertia of conventional units mostly comes from synchronous generators, so the indicators of rotating inertia of conventional units shall be sorted out first. After fully understanding its relevant contents, we turn our attention to the inertia evaluation and calculation of new energy power system[7].

### A. Rotational Inertia Index of Conventional Unit
When the synchronous generator is in operation, the expression of generator unit moment of inertia is:
$$J=\int r^{2} dm \quad (1)$$
where $r$ is the radius of rotation, $m$ is the mass of rigid body, and the unit of moment of inertia $J$ is $kg \cdot m^{2}$ [8].

When the system frequency fluctuates due to power imbalance, we pay more attention to the change of unit energy, and the speed change will be expressed in the form of energy. For synchronous generators, the stored kinetic energy is:
$$E_{k}=\frac{1}{2} J \omega^{2}$$
where $J$ is the moment of inertia; $\omega$ is the mechanical angular velocity.

The inertia constant \(H\) is defined as the ratio of kinetic energy to rated capacity, namely:
$$H=\frac{E_{k}}{S_{B}}=\frac{J \omega_{n}^{2}}{2 S_{B}}$$

The relationship between the inertia time constant \(T_{j}\) and the inertia constant is:
$$T_{j}=2 H \quad (4)$$

### B. Inertia Calculation Method of New Energy Power System
The above basic inertia index of conventional units is further extended to the inertia calculation method of high proportion new energy power system [9].

1. Including total kinetic energy of new energy power system
According to Formula, the total kinetic energy of new energy power system is:
$$E_{s}=\sum_{i=1}^{n} H_{G i} \cdot S_{G i}+\sum_{j=1}^{m} H_{P V_{j}} \cdot S_{P V_{j}}$$
where $H_{G i}$ and $S_{G i}$ respectively represent the inertia constant and rated capacity of synchronous unit $i$; $H_{P V j}$ and $S_{P V j}$ respectively represent the inertia constant and rated capacity of new energy unit $j$; $n$ and $m$ represent the number of synchronous units and new energy units respectively.

2. Including the inertia constant of new energy power system
According to Formula, the inertia constant of new energy power system is:
$$H_{s}=\frac{\sum_{i=1}^{n} H_{G i} \cdot S_{G i}+\sum_{j=1}^{m} H_{P V_{j}} \cdot S_{P V_{j}}}{S_{s}}$$
where $S_{s}$ refers to the total capacity of the system.

3. Instantaneous permeability of asynchronous power generation
For Denmark and other countries, they use the instantaneous penetration rate of asynchronous power generation to represent the inertia level of new energy power systems.
$$M=\frac{P_{RE }+P_{import }}{P_{load }+P_{export }} × 100 \%$$
where $P_{RE}$ is power output for new energy; $P_{import}$ is power of external electric input; $P_{load}$ and $P_{export}$ are load power and outgoing power respectively.

Since wind power, photovoltaic and other new energy units operate decoupled from the system, and the external power input does not have inertial response, if the new energy units do not have additional virtual inertia control, the stability of the grid is inversely proportional to $M$. If the value of $M$ is larger, it means that the smaller the system inertia is, the more unstable the system is.

The above are several common inertial estimation methods for new energy systems, but the above methods are not comprehensive enough, and the obtained inertia is theoretical inertia, so the minimum safe inertia of new energy power systems cannot be accurately calculated. In the inertia response stage, the relative size of the inertia is an important factor that hinders the frequency change, and the rate of change of frequency can better reflect the change of the system inertia. Therefore, this paper uses the rate of change of frequency as the constraint index to estimate the inertia.

## III. INERTIA ESTIMATION BASED ON FREQUENCY CHANGE RATE
### A. Analysis of the Frequency Change Process of the Electrical System
The frequency response of the electric power system mainly includes inertia response, primary frequency modulation, secondary frequency modulation, etc. [10]. Several indicators are mainly involved in the frequency variation curve of the power system, including the maximum frequency change rate $ROCOFmax$, lowest frequency point $f_{min}$, steady-state deviation $\Delta f$ and other indicators. This paper mainly studies the inertia response of the system, so the corresponding indicators are mainly the frequency change rate and the lowest point of frequency. The two values are used as inertia constraint index to estimate the inertia of the system.

### B. Mathematical Model of Power System Inertia Estimation
Because the synchronous generator occupies the main position in the power system, and the synchronous generator is also the main body of inertia response. The inertia response can be described according to the swing equation of the generator, as shown in the formula:
$$2 H \frac{d \Delta f(t)}{d t}+D \Delta f(t)=\Delta P_{e}(t)-\Delta P_{m}(t)$$
where $H$ is the inertial constant; $\Delta f(t)$ is the frequency deviation; $D$ is the damping coefficient; $\Delta P_{m}(t)$ is the amount of mechanical power change; $\Delta P_{e}(t)$ is the amount of electromagnetic power change.

It can be simplified to:
$$2 H \frac{d \Delta f(t)}{d t}+D \Delta f(t)=\Delta P$$
where $\Delta P$ is the system disturbance power, and it's a per-unit value.

And solve the differential equation about the frequency deviation \(\Delta f(t)\) of the system, which yields:
$$\Delta f(t)=\frac{\Delta P}{D}\left(1-e^{-\frac{D}{2 H}t}\right)$$

Deriving it, the system frequency change rate is:
$$ROCOF =\frac{\Delta P}{2 H} e^{-\frac{D}{2 H}t} \quad (11)$$

At the beginning of the system disturbance, the frequency drops the fastest and the frequency change rate is the largest, so substitute $t=0$ into the above equation to obtain the maximum frequency change rate after the system disturbance:
$$ROCOF_{max} =\frac{\Delta P}{2 H}$$

In summary, due to the access of a high proportion of new energy, when the system inertia level is low, the maximum frequency change rate after power disturbance, that is, the minimum safe inertia, will become lower. When the inertia of the system exceeds a certain limit, it will make the system protection device operate, which will cause serious damage to the power system. Some countries propose a standard that the frequency change rate is not higher than 0.4Hz/s or 0.6Hz/s, and 0.5Hz/s is taken in this article [11].

Therefore, the minimum safe inertia coefficient of the new energy is defined as $F$, which is the ratio of the critical frequency change rate and the maximum frequency change rate:
$$F=\frac{ROCOF_{set}}{ROCOF_{max}}=\frac{2 H \cdot ROCOF_{set}}{\Delta P}$$

$F$ is used to judge whether the inertia is sufficient when the new energy power system is disturbed. According to the size of the $F$ value, the stability of the system after disturbance can be judged:
- When $F ≥1$, it was believed that the system inertia was sufficient and the system was relatively stable;
- When $F<1$, it was believed that the system inertia was insufficient and it was easy to trigger the action of the frequency protection device.

Considering the problem of new energy penetration rate, it can be seen from equation that different new energy output ratios will have an impact on the inertia constant. Therefore, substituting equation into equation can obtain a minimum safe inertia coefficient that considers new energy penetration rate.
$$F=\frac{2 H \cdot ROCOF_{set}}{\Delta P}=\frac{2 ROCOF_{set} \cdot\left(\sum_{i=1}^{n} H_{G i} \cdot S_{G i}+\sum_{j=1}^{m} H_{P V_{j}} \cdot S_{P V_{j}}\right)}{S_{s} \cdot \Delta P}$$

## IV. MODEL BUILDING AND SIMULATION VERIFICATION
In order to verify the minimum safe inertia estimation method based on the frequency change rate proposed in this paper, the IEEE 10 generators 39 nodes New England power system model was built on DIgSILENT/PowerFactory software. The system has 10 generators; The rated frequency of the system is 60Hz, and its generator G1 is the equivalent of the external grid. The rated capacity and inertia constant of each generator are as follows.

### TABLE I. GENERATOR RATED CAPACITY AND INERTIA CONSTANT
| Generator number | Capacity, $S/MW$ | Inertia constant, $H / s$ |
| --- | --- | --- |
| 1 | 1000 | 5.000 |
| 2 | 700 | 4.329 |
| 3 | 800 | 4.475 |
| 4 | 800 | 3.575 |
| 5 | 300*2 | 4.333 |
| 6 | 800 | 4.350 |
| 7 | 700 | 3.771 |
| 8 | 700 | 3.471 |
| 9 | 1000 | 3.450 |
| 10 | 1000 | 4.200 |

### A. Analysis of the Influence of the Inertial Constant on the Lowest Point of Frequency
Increasing the power of load 29 from 250MW to 500MW causes a power deficit in the system [12]. When the inertial constant of the system is $H$, $1.5H$, $2H$, or $3H$ sequentially, with the increase of the inertia constant of the power system, the lowest point of frequency of the bus connected to the grid increases, and the frequency change rate decreases.

### B. Analysis of the Influence of Load Step on the Minimum Safe Inertia Coefficient
In order to prove the effectiveness of the minimum safe inertia coefficient proposed in this paper, different load step conditions are set to estimate the minimum safe inertia of the system, under the condition that the inertia constant of the power system is unchanged, and the system is simulated and verified according to the frequency change rate.

### TABLE II. FREQUENCY RESPONSE AT DIFFERENT LOAD STEPS
| Load step/MW | Minimum safe inertia/S | $F$ | System stability | Inertia deficiency/S |
| --- | --- | --- | --- | --- |
| 100 | 0.84 | 5.556 | stable | 0.00 |
| 200 | 1.69 | 2.778 | stable | 0.00 |
| 400 | 3.42 | 1.389 | stable | 0.00 |
| 600 | 5.04 | 0.924 | Instability | 0.95 |

The analysis of the contents of the table shows that when the system suffers from different power disturbances, the critical inertia of the system is different, and when $F<1$, it is considered that the system inertia is insufficient and the frequency instability problem is prone to occur.

### C. Analysis of the Minimum Safe Inertia Coefficient Considering PV Access
G5 in the 10 generators 39 nodes system is equivalent to PV generator controlled by inverters, and the PV penetration rate is adjusted by controlling the number of PV inverters. In this way, the effectiveness of the new energy power system can be verified. The increase in PV penetration rate leads to a decrease in the system inertia, an increase in the frequency change rate, and a decrease in the lowest point of frequency. Therefore, in the new energy power system, if there is no virtual inertia control for PV, it will lead to insufficient system inertia and easily cause system failure.

In the power system containing PV penetration rate of 13.5%, under the condition that the inertia constant is constant, different load step conditions are set to estimate the minimum safe inertia of the system, and the system is simulated and verified according to the frequency change rate. Get the table below.

### TABLE III. FREQUENCY RESPONSE AT DIFFERENT LOAD STEPS (With PV Access)
| Load step/MW | Minimum safe inertia/S | $F$ | System stability | Inertia deficiency/S |
| --- | --- | --- | --- | --- |
| 100 | 0.84 | 5.102 | stable | 0.00 |
| 200 | 1.69 | 2.551 | stable | 0.00 |
| 400 | 3.42 | 1.276 | stable | 0.00 |
| 600 | 5.64 | 0.638 | Instability | 1.32 |

By comparing Table II and Table III, it can be seen that due to the increase in PV penetration rate, if the load step is unchanged, the minimum safe inertia will also remain unchanged. However, the frequency change rate increases, the minimum safe inertia coefficient decreases, and the system is more likely to have insufficient inertia.

## V. CONCLUSION
In this paper, the calculation and evaluation methods of inertia in new energy power systems are summarized, and the minimum safe inertia evaluation method of the system constrained by the rate of frequency change is proposed, and the conclusions are as follows:
1. Through the derivation of the mathematical model, the relationship between inertia and frequency change rate is further explained, and obtain that inertia is an important parameter that hinders frequency change. The maximum value of the frequency change rate is positively correlated with the amount of power change and inversely correlated with the system inertia.
2. Based on the swing equation of the generator, this paper derives and solves its mathematical model, and proposes a minimum safe inertia evaluation method for the system constrained by the frequency change rate. The contrast value of the minimum inertia constraint is designed to judge the stability of the system power disturbance. When $F<1$, it was believed that the system inertia was insufficient and was prone to frequency instability. The proposed method is verified by simulation.
3. This paper also compares the minimum safe inertia coefficient of PV access system with the minimum safe inertia coefficient without PV access system, and concludes that the larger the minimum safe inertia coefficient is, the more sufficient the system inertia is and the more stable the system is.
