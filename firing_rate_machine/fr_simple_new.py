import numpy as np
import scipy.integrate
from functools import partial
import matplotlib.pyplot as plt
from funcsforprajay import funcs
import os
import pickle
from firing_rate_machine.utils import *


x_range = np.linspace(0, 8, 18)
result = np.random.rand(len(x_range))
funcs.make_general_scatter([x_range], [result], alpha=1, s=60, figsize=(3,3), ax_titles=['Random Plot'])


# %% from LOW RANK PAPER - same results but wayyy faster than the for loop implementation
# Network
N = 1000  # Number of units
Mm = 1.1  # Mean of m
Mn = 2.  # Mean of n
Sim = 1.  # Std of m
Sin = 1.  # Std of n

# connectivity matrix base
R = 1 * GetBulk(N)  # random matrix
m = GetGaussianVector(Mm, Sim, N)
n = GetGaussianVector(Mn, Sin, N)
M = np.outer(m, n) / N  # structured matrix

# Input
mean_i = 1
# I = GetGaussianVector(mean_i, abs(mean_i), N)
I = 0

# initial values
t0 = GetGaussianVector(1, 1, N)


print(f"mean Input value: {np.mean(I)}")
print(f"mean base connectivity weight value: {round(np.mean(R + M), 4)}, min: {round(np.min(R + M), 4)}, "
      f"max: {round(np.max(R + M), 4)}")

print('\n ** Simulating... **')

g = 0.8           # Random strength
s = 2.4           # structure strength
h = s / ((np.dot(m, n)) / N)

# Simulation

T = 40         # Total time of integration, expressed in time constants of single units
deltat = 0.1
t = np.linspace( 0, T, int(T/deltat) )

# Build connectivity matrix
J = g * R + h * M  # final connectivity matrix of the simulation

### Simulate
# initial values
# t0 = GetGaussianVector(0, 1, N)
# r_init = np.array([[1]*N]) # vector of firing rates of Nn neurons, initialized at 0 for all neurons
# t0 = r_init[0]*2

t_max = 500  # simulation length
t_stim = 0  # stimulus length (relative to total simulation length)
time_steps = int(t_max / 10) # resolution


t = np.linspace(0, t_max, t_max*time_steps)

## Simulate
F = SimulateActivity(t=t, x0=t0, J=J, I=I)

plot_network_simulation(data=F, J=J, title=f"struct & rand strength: {s}, {g}")
fravg = pop_firing_rate(F, plot=True)


# %% multi simulations

pkl_path_dir = '/home/pshah/Documents/NeuronalModelling/sim-exports'
os.makedirs(pkl_path_dir, exist_ok=True) if not os.path.exists(pkl_path_dir) else None
pkl_path_full =  f"{pkl_path_dir}/lowrank_g vs h strength, positive input.pkl"

size1 = size2 = 10
x_range = np.linspace(0, 5, size1)
# y_range = np.linspace(0, 5, size2)
y_range = [1]
size1=len(x_range)
size2=len(y_range)
result = np.empty([size1, size2])  # collect corr coefficients

# Network
N = 1000  # Number of units
Mm = 1.1  # Mean of m
Mn = 2.  # Mean of n
Sim = 1.  # Std of m
Sin = 1.  # Std of n

# connectivity matrix base
R = 1 * GetBulk(N)  # random matrix
m = GetGaussianVector(Mm, Sim, N)
n = GetGaussianVector(Mn, Sin, N)
M = np.outer(m, n) / N  # structured matrix

# Input
mean_i = 1
I = GetGaussianVector(mean_i, abs(mean_i), N)
# I = 0

# initial values
t0 = GetGaussianVector(1, 1, N)


print(f"mean Input value: {np.mean(I)}")
print(f"mean base connectivity weight value: {round(np.mean(R + M), 4)}, min: {round(np.min(R + M), 4)}, "
      f"max: {round(np.max(R + M), 4)}")

print('\n ** Simulating... **')

count = 0
for i, x in enumerate(x_range):
    for j, y in enumerate(y_range):
        count += 1

        if count % 5 == 0: save = False
        else: save = False

        g = x           # Random strength
        h = y           # structure strength

        # Simulation

        T = 40         # Total time of integration, expressed in time constants of single units
        deltat = 0.1
        t = np.linspace( 0, T, int(T/deltat) )

        # Build connectivity matrix
        J = g * R + h * M  # final connectivity matrix of the simulation

        ### Simulate
        # initial values
        # t0 = GetGaussianVector(0, 1, N)
        # r_init = np.array([[1]*N]) # vector of firing rates of Nn neurons, initialized at 0 for all neurons
        # t0 = r_init[0]*2

        t_max = 500  # simulation length
        t_stim = 0  # stimulus length (relative to total simulation length)
        time_steps = int(t_max / 10) # resolution


        t = np.linspace(0, t_max, t_max*time_steps)

        ## Simulate
        F = SimulateActivity(t=t, x0=t0, J=J, I=I)

        # plot_network_simulation(data=F, title=f"h param: {x}") if plot else None
        fravg = pop_firing_rate(F, plot=False)
        result[i, j] = fravg

        if save:
            to_save = {'avg_fr': result,
                       'variables':
                           {'g_range': x_range,
                            'h_range': y_range,
                            'Input val': np.mean(I),
                            'Init val': np.mean(t0),
                            },
                       'title': f"g vs h strength",
                       'experiment': 'g vs h strength, positive input',
                       'pkl_path_full': pkl_path_full
                       }

            with open(to_save['pkl_path_full'], 'wb') as f:
                print(f"saving to ({count}): {to_save['pkl_path_full']}")
                pickle.dump(to_save, f)


# plot network response vs. conn. param strength
funcs.make_general_scatter([x_range], [result], alpha=1, s=60, figsize=(3,3), x_label='conn param', y_label='Avg FR (norm.)',
                           ax_titles=[f"h strength: {round((Mm*Mn)/N, 2)}"])

# %% TODO heatmap of the simulation results
arr = load_(pkl_path=pkl_path_full)

var1 = list(arr['variables'].keys())[0]
var2 = list(arr['variables'].keys())[1]
var1_label = 'Random strength'
var2_label = 'Struct. strength'

fig, ax = plt.subplots(figsize=(3, 3))
ylabels = [round(i, 1) for i in arr['variables'][var1]]
xlabels = [round(i, 1) for i in arr['variables'][var2]]
plot_heatmap(data=arr['avg_fr'], xlabel=var2_label, ylabel=var1_label,
                colorlabel=None, vmin=0, vmax=20, title='Avg FR', fig=fig, ax=ax, show=False,
                xlabels=xlabels, ylabels=ylabels)
fig.tight_layout()
fig.show()
