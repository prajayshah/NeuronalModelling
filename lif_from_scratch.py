import numpy as np
import matplotlib.pyplot as plt

## setup parameters and state variables
T = 50
dt = 0.125

time = np.arange(0, T+dt, dt)
t_rest = 0

## LIF properties
Vm = np.zeros(len(time))
Rm = 1
Cm = 10
tau_m = Rm*Cm
tau_ref =4
Vth = 1
V_spike = 0.5

## Input stimulus
I = 1.5
## iterate over each time step
for i, t in enumerate(time):
    if t > t_rest:
        Vm[i] = Vm[i - 1] + (-Vm[i - 1] + I * Rm) / tau_m * dt
        if Vm[i] >= Vth:
            Vm[i] += V_spike
            t_rest = t + tau_ref

## plot membrane potential trace
plt.plot(time, Vm)
plt.title('Leaky Integrate-and-Fire Example')
plt.ylabel('Membrane Potential (V)')
plt.xlabel('Time (msec)')
plt.ylim([0,2])
plt.show()

