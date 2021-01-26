# vanilla firing rate type model of a recurrent neuronal network based on some simple initial guidance from Thijs
import numpy as np
import matplotlib.pyplot as plt
import random

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
N_projections = 210
x = np.array([10,0.0,0.0]) # vector of inputs (can also add a time dimension to have time varying inputs)
x_stop = np.array([0,0,0])  # empty vector to use for stopping projection inputs after a given time
J = np.repeat(np.array([[1,0,0], [0,1,0], [0,0,1]]), repeats=N_projections/Ni, axis=0) # matrix of input connection weights

# recurrent layer
Nn = 210 # number of neurons in the recurrent layer
r_init = np.array([[0]*Nn]) # vector of firing rates of Nn neurons, initialized at 0 for all neurons
W = np.diag(np.ones(Nn))  # matrix of recurrent connection weights, should be a Nn x Nn size array

# modifying recurrent connection weights as you want to for the network
# start with building random E-E connectivity
ee_p= 0.2  # probability of e-e connections
i_p = 0.2 # probability of Inh. neurons
e_strength = 1
i_strength = -0.9
for i in range(len(W)):
    # set E - E connectivity
    idx = random.sample(range(len(W[i])), int(ee_p*len(W[i])))  # select random indexes to setup as connections based on selected probability
    W[i, idx] = e_strength

# set I - E connectivity -- doesn't need for loop because these will make Inh connections to all neurons
idx = random.sample(range(Nn), int(i_p*Nn))
W[:, idx] = i_strength



# output layer - null for now

b = np.array([0]*Nn)  # external input
tau = 1/2  # time-constant


#%%
#################################################################################
### SIMULATION PARAMETERS
#################################################################################

t_max = 50  # simulation length
t_stim = 0  # stimulus length (relative to total simulation length)
time_steps = 100 # resolution

# simulation equations and plots
'''drdt = 1/tau * (-r + np.tanh(W@r + J@x + b))'''

# stimulation function
f = lambda y, t: 1/tau*(-y + np.tanh(W@y + J@x + b))
# f = lambda y, t: 1/tau*(-y + rectify(W@y + J@x + b))
t = np.linspace(0, t_max, t_max*time_steps)

# remaining (non-stim) function
f_2 = lambda y, t: 1/tau*(-y + np.tanh(W@y + J@x_stop + b))
# f_2 = lambda y, t: 1/tau*(-y + rectify(W@y + J@x_stop + b))
# t_2 = np.linspace(t_stim, t_max, t_max*time_steps)


#################################################################################
### SIMULATION OF THE NETWORK
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

#%% plotting firing rate as heatmap of Neurons x Time
to_plot=r_2
plt.figure(figsize=[10,10])
plt.imshow(to_plot.T); plt.title('Firing Rate: Neurons x Time'); plt.colorbar(fraction=0.008)
plt.show()



