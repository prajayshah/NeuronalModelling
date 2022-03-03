# vanilla firing rate type model of a recurrent neuronal network based on some simple initial guidance from Thijs
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.integrate
from functools import partial

# Euler's method function
def odeEuler(f,y0,t):
    """
    Approximate the solution of y'=f(y,t) by Euler's method.

    Parameters
    ----------
    f : function
        Right-hand side of the differential equation y'=f(t,y), y(t_0)=y_0
    y0 : array
        Initial Numpy array of rates r(t0)=r0 where r0 is the entry at index 0 of the array t
    t : array
        1D NumPy array of t values to simulate where we approximate y values. Time step
        at each iteration is given by t[n+1] - t[n].

    Returns
    -------
    y : 1D NumPy array
        Approximation y[n] of the solution y(t_n) computed by Euler's method.
    """

    y = y0
    for n in range(0,len(t)-1):
        y = np.append(y, np.array([y[n] + f(y[n],t[n])*(t[n+1] - t[n])]), axis=0)  # you should really do this so that you set up a matrix which contains the necessary number of entries, and then enter each iterative solution in the appropriate slot
    return y

def rectify(input):
    return np.multiply(input, (input > 0))

#%%
#################################################################################
### NETWORK BUILD BELOW
#################################################################################

# input
Ni = 3 # number of neurons from feedforward layer
N_projections = 400
x = np.array([10,0.0,0.0]) # vector of inputs (can also add a time dimension to have time varying inputs)
x_stop = np.array([0,0,0])  # empty vector to use for stopping projection inputs after a given time
I = np.repeat(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), repeats=N_projections / Ni, axis=0) # matrix of input connection weights

# recurrent layer
Nn = 400-1 # number of neurons in the recurrent layer
r_init = np.array([[1]*Nn]) # vector of firing rates of Nn neurons, initialized at 0 for all neurons
# W = np.diag(np.ones(Nn))  # matrix of recurrent connection weights, should be a Nn x Nn size array
W = np.zeros((Nn, Nn))  # matrix of recurrent connection weights, should be a Nn x Nn size array

# modifying recurrent connection weights as you want to for the network
# start with building random E-E connectivity
ee_p= 0.02  # probability of e-e connections
ie_p = 0.02 # probability of I-E connections
i_p = 0.2 # probability of Inh. neurons
e_strength = 1
i_strength = -1.5

e_neurons = random.sample(range(Nn), int((1-i_p)*Nn))
i_neurons = [i for i in range(Nn) if i not in e_neurons]

for i in e_neurons:
    # set E - E connectivity
    idx = random.sample(range(len(W[i])), int(ee_p*len(W[i])))  # select random indexes to setup as connections based on selected probability
    W[idx, i] = e_strength

for i in i_neurons:
    # set I - E connectivity
    idx = random.sample(range(len(W[i])), int(ie_p*len(W[i])))  # select random indexes to setup as connections based on selected probability
    W[idx, i] = i_strength


# # set I -> all E connectivity -- doesn't need for loop because these will make Inh connections to all neurons
# i = random.sample(range(Nn), int(i_p*Nn))
# W[:, i] = i_strength



# output layer - null for now

b = np.array([0]*Nn)  # external input
tau = 1  # time-constant


#################################################################################
### SIMULATION PARAMETERS
#################################################################################

t_max = 1000  # simulation length
t_stim = 0  # stimulus length (relative to total simulation length)
time_steps = 100 # resolution

# simulation equations and plots
'''drdt = 1/tau * (-r + np.tanh(W@r + I@x + b))'''

# stimulation function
f = lambda y, t: 1/tau*(-y + W @ np.tanh(y) + I @ np.tanh(x) + b)
# f = lambda y, t: 1/tau*(-y + np.tanh(W @ y + I @ x + b))
# f = lambda y, t: 1/tau*(-y + rectify(W@y + I@x + b))
t = np.linspace(0, t_max, t_max*time_steps)

# remaining (non-stim) function
f_2 = lambda y, t: 1/tau*(-y + np.tanh(W @ y + I @ x_stop + b))
# f_2 = lambda y, t: 1/tau*(-y + rectify(W@y + I@x_stop + b))
# t_2 = np.linspace(t_stim, t_max, t_max*time_steps)


# %% from LOW RANK PAPER - same results but wayyy faster than the for loop implementation
#################################################################################
### RUN SIMULATION OF THE NETWORK
#################################################################################

def Integrate (X, t, J, I):
    dXdT = -X + np.dot( J, np.tanh(X) ) + I
    return dXdT

def SimulateActivity (t, x0, J, I):
    print(' ** Simulating... **')
    return scipy.integrate.odeint( partial(Integrate, J=J, I=I), x0, t )

r_2 = SimulateActivity(t=t, x0=r_init[0]*50, J=W, I=0)

## plot connectivity matrix
f, axs = plt.subplots(nrows=1, ncols=2, figsize=[10,5])
im = axs[0].imshow(W)
axs[0].set_title('Sources (left) x Targets (bottom) connectivity matrix')


# plotting firing rate as heatmap of Neurons x Time

to_plot=r_2
# f, ax = plt.subplots(figsize=[10,2])
im = axs[1].imshow(to_plot.T, cmap='RdBu_r',aspect=144, vmin=-10, vmax=10)
f.colorbar(im, orientation='vertical', fraction=0.08)
axs[1].set_title('Firing Rate: Neurons x Time')
# ax.set_colorbar(fraction=0.008)
f.tight_layout()
f.show()


# %%
#################################################################################
### RUN SIMULATION OF THE NETWORK
#################################################################################
# r = odeEuler(f=f, y0=r_init, t=t)

# for loop from odeEuler solution to temporarily handle transient stimulation
y = r_init
for n in range(0,len(t)-1):
    if n < t_stim*time_steps:
        y = np.append(y, np.array([y[n] + f(y[n],t[n])*(t[n+1] - t[n])]), axis=0)  # you should really do this so that you set up a matrix which contains the necessary number of entries, and then enter each iterative solution in the appropriate slot
    else:
        y = np.append(y, np.array([y[n] + f_2(y[n],t[n])*(t[n+1] - t[n])]), axis=0)  # you should really do this so that you set up a matrix which contains the necessary number of entries, and then enter each iterative solution in the appropriate slot
r_2 = y

