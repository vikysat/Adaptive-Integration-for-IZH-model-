# Adaptive Integration for the Izhikevich Neuron Model

> *Fewer steps. No missed spikes. Better simulation.*

---

## Overview

This repository explores and benchmarks dynamic time-step integration methods for the Izhikevich spiking neuron model. The goal is to drastically reduce the computational cost of neural simulations without sacrificing temporal fidelity — spike timing, subthreshold voltage accuracy, or anything else that matters.

Several experimental solvers are included, documenting the progression toward the proposed method: **Exponential State-Dependent Integration**.

---

## The Core Contribution — Exponential $dv/dt$ Integration

**Function:** `adaptive_dvdt_exp`

Most adaptive solvers adjust step size based on threshold crossings or generic error tolerances. This method does something different — it scales the discrete array jump directly from the instantaneous derivative of the membrane potential ($dv/dt$).

An action potential is defined by an explosive rate of change. By mapping the time step to $dv/dt$ through an exponential decay, the solver gets ultra-fine resolution exactly when the voltage trajectory begins to curve — and maximum acceleration everywhere else.

$$N_{\text{jump}} = \left\lfloor N_{\text{min}} + (N_{\text{max}} - N_{\text{min}}) \exp\left(-k\left|\frac{dv}{dt}\right|\right) \right\rfloor$$

When $dv/dt \approx 0$ (flat, subthreshold regime), the exponential evaluates to 1 and the simulation jumps by $N_{\text{max}}$. The moment voltage begins rising toward a spike, the jump size collapses — no timing precision lost.

---

## Experimental Methods

The following methods were explored before arriving at the exponential approach. They're included to document the reasoning and show where each falls short.

---

### Sigmoid Voltage Mapping

**Function:** `adaptivesig`

Scales the time step by passing membrane voltage $v$ through a sigmoid centered at threshold $v_0$.

$$N_{\text{jump}} = \left\lfloor N_{\text{min}} + (N_{\text{max}} - N_{\text{min}}) \left( 1 - \frac{1}{1 + \exp(-k(v - v_0))} \right) \right\rfloor$$

**Where it breaks down:** Mapping to absolute voltage rather than rate of change means the solver can't distinguish between a heavily depolarized-but-stable neuron and one about to fire. High $v$, low $dv/dt$ — the step shrinks unnecessarily and wastes computation.

---

### Inverse Voltage Mapping

**Function:** `adaptive_dt`

Uses an inverse proportional relationship to membrane voltage, offset by a constant $c$.

$$N_{\text{jump}} = \left\lfloor \max\left(N_{\text{min}}, \frac{N_{\text{max}}}{1 + k|v + c|}\right) \right\rfloor$$

**Where it breaks down:** Same fundamental problem as the sigmoid — the step size is bound to the absolute state variable rather than its derivative. The solver doesn't respond to the explosive trajectory of a spike until the voltage is already high, leading to numerical instability and delayed spike times.

---

## Baselines

Performance is benchmarked against two standard implementations.

| | |
|---|---|
| **Reference** `reference` | Gold standard. Standard Euler integration at a fixed step of 0.001 ms — maximally granular, maximally expensive. |
| **Interpolated** `interpolated` | Izhikevich's proposed hybrid. Runs at a larger fixed step (e.g. 0.025 ms) but mathematically interpolates the exact spike peak between steps. |

---

*Computational neuroscience shouldn't have to choose between speed and accuracy.*
