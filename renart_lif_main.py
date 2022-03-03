# imports and some general functions
# replicating the Renart et al., 2010, Science balanced E-I network of spiking neurons

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit

def delta(x):
    return 1 if x == 0 else 0

def set_to_one(array, probability):
    '''
    Set indices from the given array to 1 based on the given probability
    :param array: NumPy array
    :param probability: probability with which to change indices to 1
    :return: NumPy array with some indices set to 1 based on a given probability
    '''

    for i in np.arange(len(array)):
        indices_to_change = np.random.choice(np.arange(len(array[i])), round(probability * len(array[i])),
                                             replace=False)
        array[i][indices_to_change] = 1

    return array

def set_from_gaus(array, mean):
    '''
    Set indices from the given array to a value drawn from a gaussian distribution defined with mu = mean and sigma = 0.5*mu
    :param array: NumPy array
    :param mean: mean of thea gaussian distribution
    :return: NumPy array with some indices set to 1 based on a given probability
    '''

    gaussian = np.random.normal(mean, 0.5*mean, array.shape[1])
    for i in np.arange(len(array)):
        randomized_indices = np.random.choice(np.arange(len(array[i])), len(array[i]), replace=False)
        array[i][randomized_indices] = gaussian[randomized_indices]
    return array

def set_from_uniform(array, start, end):
    '''
    Set indices from the given array to a value drawn from a uniform distribution defined between a range of values (sampled at 0.05ms resolution)
    :param array: NumPy array
    :param start: lower value of distribution
    :param end: higher value of distribution
    :return: NumPy array with some indices set to 1 based on a given probability
    '''

    values = np.arange(start, end, 0.05)
    for i in np.arange(len(array)):
        choices = np.random.choice(values, array.shape[1],replace=True)
        array[i] = choices
    return array

#%% NETWORK BUILD

# set some initial constants

Ne = 400  # number of E neurons
Ni = 100  # number of I neurons
Nx = 400  # number of X (external population) neurons

Cm = 0.25e-6 #mFaraday
gL = 16.7e-6 #mSeimens
tau = 15.  #ms -- membrane constant, defined by Cm/gL
Vl = -70. #mV -- resting potential
V_t = -50. #mV -- threshold
V_refrac = -60.  #mV -- refractory voltage
V_rev_E = 0. #mV
V_rev_X = 0. #mV
V_rev_I = -80. #mV

ext_freq = 10  #Hz - external pop'n spiking frequency
time_simulation = 1*1000 #ms
dt = 0.1
time = np.linspace(0, time_simulation, int(time_simulation*(1/dt)))


# define equations for spiking
''' eqs:
if V_(e/i) < V_t then,
    dVe/dt = -gL/Cm * (Ve - Vl) + IeE(t) + IeI(t) + IeX(t)
    dVi/dt = -gL/Cm * (Vi - Vl) + IiE(t) + IiI(t) + IiX(t)
    where,
        IeE(t) = -[p_ij_eE @ g_ij_eE @ s_ij_eE(t)] * (Ve - V_rev_E)
        IeI(t) = -[p_ij_eI @ g_ij_eI @ s_ij_eI(t)] * (Ve - V_rev_I)
        IeX(t) = -[p_ij_eX @ g_ij_eX @ s_ij_eX(t)] * (Ve - V_rev_X)
        
        and
        
        IiE(t) = -[p_ij_iE @ g_ij_iE @ s_ij_iE(t)] * (Vi - V_rev_E)
        IiI(t) = -[p_ij_iI @ g_ij_iI @ s_ij_iI(t)] * (Vi - V_rev_I)
        IiX(t) = -[p_ij_iX @ g_ij_iX @ s_ij_iX(t)] * (Vi - V_rev_X)
        
        where, (stated below are definitions for ds: synaptic gating variable that opens and closes synaptic conductance)
            ds_ij_eE = 1/tau_d * (x_ij_eE - s_ij_eE)
            dx_ij_eE = 1/tau_r * sum_over_t_j_E[delta(t - t_j_E - d_ij_iE)] - x_ij_eE  
        
        and,
            p_x_x = 1 or 0 with P(1) = 0.2  # connectivity binary variable
            g_x_x = value from a Gaussian distribution of mean g_II, gEE, gIE, gEI, g_IX or g_EX 
                g_II = 40  nS
                g_EE = 2.4 nS
                g_IE = 4.8 nS
                g_EI = 40  ns
                g_IX = 5.4 nS
                g_EX = 5.4 nS 
            V_rev_E =   0 mV
            V_rev_X =   0 mV
            V_rev_I = -80 mV
else,
    Ve = -60 #mV for 2ms  
    Vi = -60 #mV for 1ms
'''

# setup data matrices to hold voltage values for E, I and X populations
Ve = np.full([Ne, int(time_simulation*(1/dt))], Vl); spikes_e = np.zeros([Ne, int(time_simulation*(1/dt))])
Vi = np.full([Ni, int(time_simulation*(1/dt))], Vl); spikes_i = np.zeros([Ni, int(time_simulation*(1/dt))])

# define some random spikes to add to the external pop'n - doesn't have Poisson statistics yet.
spikes_x = np.zeros([Nx, int(time_simulation*(1/dt))])  # set external input spikes
spikes_x = set_to_one(spikes_x, probability=ext_freq/1000*dt)



# variables for connections/synapses
probability=0.5  # general probability of connectivity, same for all types of connections
p_ij_eE = np.zeros((Ne, Ne)); p_ij_eE = set_to_one(p_ij_eE, probability=probability)
p_ij_eI = np.zeros((Ne, Ni)); p_ij_eI = set_to_one(p_ij_eI, probability=probability)
p_ij_eX = np.zeros((Ne, Nx)); p_ij_eX = set_to_one(p_ij_eX, probability=probability)

p_ij_iE = np.zeros((Ni, Ne)); p_ij_iE = set_to_one(p_ij_iE, probability=probability)
p_ij_iI = np.zeros((Ni, Ni)); p_ij_iI = set_to_one(p_ij_iI, probability=probability)
p_ij_iX = np.zeros((Ni, Nx)); p_ij_iX = set_to_one(p_ij_iX, probability=probability)


# variables for synapse strength matrices
g_ij_eE = np.zeros((Ne, Ne)); g_ij_eE = set_from_gaus(g_ij_eE, mean=2.4e-6)  #mSeimens
g_ij_eI = np.zeros((Ne, Ni)); g_ij_eI = set_from_gaus(g_ij_eI, mean=40e-6)  #mSeimens
g_ij_eX = np.zeros((Ne, Nx)); g_ij_eX = set_from_gaus(g_ij_eX, mean=5.4e-6)  #mSeimens

g_ij_iE = np.zeros((Ni, Ne)); g_ij_iE = set_from_gaus(g_ij_iE, mean=4.8e-6)  #mSeimens
g_ij_iI = np.zeros((Ni, Ni)); g_ij_iI = set_from_gaus(g_ij_iI, mean=40e-6)  #mSeimens
g_ij_iX = np.zeros((Ni, Nx)); g_ij_iX = set_from_gaus(g_ij_iX, mean=5.4e-6)  #mSeimens




# TODO - functions for calculating the synaptic conductance changes (essentially specifies *when* a post-synaptic current occurs)
tau_d = 5 #ms
tau_r = 1 #ms

s_ij_eE = np.zeros((Ne, Ne, int(time_simulation*(1/dt))))
x_ij_eE = np.zeros((Ne, Ne, int(time_simulation*(1/dt))))
d_ij_eE = np.zeros((Ne, Ne)); set_from_uniform(d_ij_eE, 0.5, 1.5)  # synaptic conductance delay - !!!! - this needs to be a specific uniform distribution (see Renart suppl)

s_ij_eI = np.zeros((Ne, Ni, int(time_simulation*(1/dt))))
x_ij_eI = np.zeros((Ne, Ni, int(time_simulation*(1/dt))))
d_ij_eI = np.zeros((Ne, Ni)); set_from_uniform(d_ij_eE, 0.1, 0.9)  # synaptic conductance delay - !!!! - this needs to be a specific uniform distribution (see Renart suppl)

s_ij_eX = np.zeros((Ne, Nx, int(time_simulation*(1/dt))))
x_ij_eX = np.zeros((Ne, Nx, int(time_simulation*(1/dt))))
d_ij_eX = np.zeros((Ne, Nx))  # synaptic conductance delay - !!!! - this needs to be a specific uniform distribution (see Renart suppl)

x_ij_iX = np.zeros((Ni, Nx, int(time_simulation*(1/dt))))
s_ij_iX = np.zeros((Ni, Nx, int(time_simulation*(1/dt))))
d_ij_iX = np.zeros((Ni, Nx))  # synaptic conductance delay - !!!! - this needs to be a specific uniform distribution (see Renart suppl)

x_ij_iI = np.zeros((Ni, Ni, int(time_simulation*(1/dt))))
s_ij_iI = np.zeros((Ni, Ni, int(time_simulation*(1/dt))))
d_ij_iI = np.zeros((Ni, Ni)); set_from_uniform(d_ij_eE, 0.1, 0.9)  # synaptic conductance delay - !!!! - this needs to be a specific uniform distribution (see Renart suppl)

x_ij_iE = np.zeros((Ni, Ne, int(time_simulation*(1/dt))))
s_ij_iE = np.zeros((Ni, Ne, int(time_simulation*(1/dt))))
d_ij_iE = np.zeros((Ni, Ne)); set_from_uniform(d_ij_eE, 0.5, 1.5)  # synaptic conductance delay - !!!! - this needs to be a specific uniform distribution (see Renart suppl)


#%% equations
# this is two coupled differential equations, where y = [s_ij_aB, x_ij_aB]
# @jit
def synaptic_gate_eE(i, j, n):
    '''
    :param i: post-synaptic cell i that is being computed on currently
    :param j: pre-synaptic cell j that needs to be evaluated for its input onto cell i
    :param n: current time-step of the simulation
    :return: the synaptic gate/conductance for cell j to cell i
    '''

    spk = 0
    spikes_j = np.where(spikes_e[j]==1)
    if int(n - d_ij_eE[i, j]) in spikes_j[0]:
        spk = 1

    dxdt = lambda x : 1/tau_r * spk - x
    dsdt = lambda x, s: 1 / tau_d * (x - s)

    x_ij_eE[i, j, n + 1] = x_ij_eE[i, j, n] + dxdt(x = x_ij_eE[i, j, n]) * (time[n + 1] - time[n])
    s_ij_eE[i, j, n + 1] = s_ij_eE[i, j, n] + dsdt(x = x_ij_eE[i, j, n+1], s = s_ij_eE[i, j, n]) * (time[n + 1] - time[n])

    return s_ij_eE[i, j, n+1]
# @njit
def synaptic_gate_eI(i, j, n):
    '''
    :param i: post-synaptic cell i that is being computed on currently
    :param j: pre-synaptic cell j that needs to be evaluated for its input onto cell i
    :param n: current time-step of the simulation
    :return: the synaptic gate/conductance for cell j to cell i
    '''

    spk = 0
    spikes_j = np.where(spikes_i[j]==1)
    if int(n - d_ij_eI[i, j]) in spikes_j[0]:
        spk = 1

    dxdt = lambda x : 1/tau_r * spk - x
    dsdt = lambda x, s: 1 / tau_d * (x - s)

    x_ij_eI[i, j, n + 1] = x_ij_eI[i, j, n] + dxdt(x = x_ij_eI[i, j, n]) * (time[n + 1] - time[n])
    s_ij_eI[i, j, n + 1] = s_ij_eI[i, j, n] + dsdt(x = x_ij_eI[i, j, n+1], s = s_ij_eI[i, j, n]) * (time[n + 1] - time[n])

    return s_ij_eI[i, j, n+1]
# @njit
def synaptic_gate_eX(i, j, n):
    '''
    :param i: post-synaptic cell i that is being computed on currently
    :param j: pre-synaptic cell j that needs to be evaluated for its input onto cell i
    :param n: current time-step of the simulation
    :return: the synaptic gate/conductance for cell j to cell i
    '''

    spk = 0
    spikes_j = np.where(spikes_x[j]==1)
    if int(n - d_ij_eX[i, j]) in spikes_j[0]:
        spk = 1

    dxdt = lambda x : 1/tau_r * spk - x
    dsdt = lambda x, s: 1 / tau_d * (x - s)

    x_ij_eX[i, j, n + 1] = x_ij_eX[i, j, n] + dxdt(x = x_ij_eX[i, j, n]) * (time[n + 1] - time[n])
    s_ij_eX[i, j, n + 1] = s_ij_eX[i, j, n] + dsdt(x = x_ij_eX[i, j, n+1], s = s_ij_eX[i, j, n]) * (time[n + 1] - time[n])

    return s_ij_eX[i, j, n+1]
# @njit
def synaptic_gate_iE(i, j, n):
    '''
    :param i: post-synaptic cell i that is being computed on currently
    :param j: pre-synaptic cell j that needs to be evaluated for its input onto cell i
    :param n: current time-step of the simulation
    :return: the synaptic gate/conductance for cell j to cell i
    '''

    spk = 0
    spikes_j = np.where(spikes_e[j]==1)
    if int(n - d_ij_iE[i, j]) in spikes_j[0]:
        spk = 1

    dxdt = lambda x : 1/tau_r * spk - x
    dsdt = lambda x, s: 1 / tau_d * (x - s)

    x_ij_iE[i, j, n + 1] = x_ij_iE[i, j, n] + dxdt(x = x_ij_iE[i, j, n]) * (time[n + 1] - time[n])
    s_ij_iE[i, j, n + 1] = s_ij_iE[i, j, n] + dsdt(x = x_ij_iE[i, j, n+1], s = s_ij_iE[i, j, n]) * (time[n + 1] - time[n])

    return s_ij_iE[i, j, n+1]
# @njit
def synaptic_gate_iI(i, j, n):
    '''
    :param i: post-synaptic cell i that is being computed on currently
    :param j: pre-synaptic cell j that needs to be evaluated for its input onto cell i
    :param n: current time-step of the simulation
    :return: the synaptic gate/conductance for cell j to cell i
    '''

    spk = 0
    spikes_j = np.where(spikes_i[j]==1)
    if int(n - d_ij_iI[i, j]) in spikes_j[0]:
        spk = 1

    dxdt = lambda x : 1/tau_r * spk - x
    dsdt = lambda x, s: 1 / tau_d * (x - s)

    x_ij_iI[i, j, n + 1] = x_ij_iI[i, j, n] + dxdt(x = x_ij_iI[i, j, n]) * (time[n + 1] - time[n])
    s_ij_iI[i, j, n + 1] = s_ij_iI[i, j, n] + dsdt(x = x_ij_iI[i, j, n+1], s = s_ij_iI[i, j, n]) * (time[n + 1] - time[n])

    return s_ij_iI[i, j, n+1]
# @njit
def synaptic_gate_iX(i, j, n):
    '''
    :param i: post-synaptic cell i that is being computed on currently
    :param j: pre-synaptic cell j that needs to be evaluated for its input onto cell i
    :param n: current time-step of the simulation
    :return: the synaptic gate/conductance for cell j to cell i
    '''

    spk = 0
    spikes_j = np.where(spikes_x[j]==1)
    if int(n - d_ij_iX[i, j]) in spikes_j[0]:
        spk = 1

    dxdt = lambda x : 1/tau_r * spk - x
    dsdt = lambda x, s: 1 / tau_d * (x - s)

    x_ij_iX[i, j, n + 1] = x_ij_iX[i, j, n] + dxdt(x = x_ij_iX[i, j, n]) * (time[n + 1] - time[n])
    s_ij_iX[i, j, n + 1] = s_ij_iX[i, j, n] + dsdt(x = x_ij_iX[i, j, n+1], s = s_ij_iX[i, j, n]) * (time[n + 1] - time[n])

    return s_ij_iX[i, j, n+1]

# @njit
def synaptic_current(i, n, V, alpha_cell=None):
    '''
    :param i: cell i from the network - post-synaptic cell
    :param n: current time step of the network
    :param alpha_cell: cell represented by 'alpha' in Renart Equations
    :return: total synaptic current for post-synaptic cell i
    '''

    if alpha_cell=='E':
        Ve = V

        # e <-- E current
        IeE_t = 0
        for j in np.arange(Ne):
            if p_ij_eE[i,j] > 0: # check if there is a synaptic connection, and only if there is then calculate the synaptic gate/current for this synapse
                gate = synaptic_gate_eE(i, j, n)
                IeE_t += p_ij_eE[i,j] * g_ij_eE[i,j] * gate
        IeE_t = -IeE_t * (Ve - V_rev_E)

        # e <-- I current
        IeI_t = 0
        for j in np.arange(Ni):
            if p_ij_eE[i, j] > 0:
                gate = synaptic_gate_eI(i, j, n)
                IeI_t += p_ij_eI[i, j] * g_ij_eI[i, j] * gate
        IeI_t = -IeI_t * (Ve - V_rev_I)

        # e <-- X current
        IeX_t = 0
        for j in np.arange(Nx):
            if p_ij_eE[i, j] > 0:
                gate = synaptic_gate_eX(i, j, n)
                IeX_t += p_ij_eX[i, j] * g_ij_eX[i, j] * gate
        IeX_t = -IeX_t * (Ve - V_rev_X)

        return np.sum([IeE_t, IeI_t, IeX_t])

    elif alpha_cell == 'I':
        Vi = V

        # i --> E current
        IiE_t = 0
        for j in np.arange(Ne):
            if p_ij_eE[i, j] > 0:
                gate = synaptic_gate_iE(i, j, n)
                IiE_t += p_ij_iE[i, j] * g_ij_iE[i, j] * gate
        IiE_t = -IiE_t * (Vi - V_rev_E)

        # i --> I current
        IiI_t = 0
        for j in np.arange(Ni):
            if p_ij_eE[i,j] > 0:
                gate = synaptic_gate_iI(i, j, n)
                IiI_t += p_ij_iI[i, j] * g_ij_iI[i, j] * gate
        IiI_t = -IiI_t * (Vi - V_rev_E)

        # i --> X current
        IiX_t = 0
        for j in np.arange(Nx):
            if p_ij_eE[i,j] > 0:
                gate = synaptic_gate_iX(i, j, n)
                IiX_t += p_ij_iX[i, j] * g_ij_iX[i, j] * gate
        IiX_t = -IiX_t * (Vi - V_rev_E)

        return np.sum([IiE_t, IiI_t, IiX_t])

    else:
        return ValueError('No alpha cell specified')

# equations
# dVe_dt = lambda Ve_i, i, n: 1/tau * (Vl - Ve_i + 1/gL * synaptic_current(i, n, Ve_i, alpha_cell='E'))
# dVi_dt = lambda Vi_i, i, n: 1/tau * (Vl - Vi_i + 1/gL * synaptic_current(i, n, Vi_i, alpha_cell='I'))

# @njit
def dVe_dt(Ve_i, i, n):
    return (1/tau * (Vl - Ve_i + 1/gL * synaptic_current(i, n, Ve_i, alpha_cell='E')))

# @njit
def dVi_dt(Vi_i, i, n):
    return (1/tau * (Vl - Vi_i + 1/gL * synaptic_current(i, n, Vi_i, alpha_cell='I')))


#%% RUN SIMULATION
# @njit
def run_simulation(time, Ve, Vi, spikes_e, spikes_i, dt=dt, V_refrac=V_refrac):
    print('starting simulation...')
    for n in range(int(2 * (1 / dt)),len(time)-1):  # note that there is a 40ms gap to avoid having negative check-backs for the first 40ms in the first if statement
        if n % 100 == 0:
            print(n)

        for i in np.arange(Ve.shape[0]): ## select a neuron i to calculate dVe/dt for
            # print message about progress of simulation
            msg = ' -- Working on Exc cell %d out of %d, at dt step %d out of %d' % (i, Ve.shape[0], n, len(time))
            print(msg, end='\r')

            # if - else for sub-threshold vs spiking behaviours
            if np.sum(spikes_e[i, int(n - (2 * (1 / dt))):n]) > 0:
                Ve[i, n+1] = V_refrac  #mV -- refractory period
                # print('refrac', i, n)

            elif Ve[i, n] < V_t:
                # run voltage update equation - Euler's method using dVe_dt defined above
                Ve_change = dVe_dt(Ve[i,n], i, n)*(time[n+1] - time[n])
                Ve[i,n+1] = Ve[i,n] + Ve_change

            if Ve[i, n+1] >= V_t:
                spikes_e[i, n+1] = 1
                print('Spike! Voltage: ', Ve[i, n+1])
                # print('-- Neuron index', i, 'time index', n+1)
                # print(synaptic_current(i, n, Ve[i,n]))
                # print('-- Spike!')

        for i in np.arange(Vi.shape[0]): ## select a neuron i to calculate dVi/dt for
            # print message about progress of simulation
            # msg = ' -- Working on Inh cell %d out of %d, at dt step %d out of %d' % (i, Vi.shape[0], n, len(time))
            # print(msg, end='\r')

            # if - else for sub-threshold vs spiking behaviours
            if np.sum(spikes_i[i, int(n - (1 * (1 / dt))):n]) > 0:
                Vi[i, n+1] = -60.  #mV -- refractory period (should be -60mV but might have been changed it for trialling)
                # print('refrac', i, n)

            elif Vi[i, n] < V_t:
                # run voltage update equation - Euler's method using dVe_dt defined above
                Vi_change = dVi_dt(Vi[i,n], i, n)*(time[n+1] - time[n])
                Vi[i,n+1] = Vi[i,n] + Vi_change

            if Vi[i, n+1] >= V_t:
                spikes_i[i, n+1] = 1
                print('Spike! Voltage: ', Ve[i, n+1])
                # print('-- Neuron index', i, 'time index', n+1)
                # print(synaptic_current(i, n, Ve[i,n]))
                # print('-- Spike!')

    return Ve, Vi, spikes_e, spikes_i

run_sim = True

if run_sim:
    Ve, Vi, spikes_e, spikes_i = run_simulation(time=time[:10], Ve=Ve, Vi=Vi, spikes_e=spikes_e, spikes_i=spikes_i)
    Ve, Vi, spikes_e, spikes_i = run_simulation(time=time[10:], Ve=Ve, Vi=Vi, spikes_e=spikes_e, spikes_i=spikes_i)



#%% plotting
cell = 9
t_first = 0
t_last = len(time)
V = Ve
spk = spikes_e

fig, axs = plt.subplots(2,1)
axs[0].plot(V[cell, t_first:t_last])
axs[0].set_ylim(Vl-15,V_t+15)
axs[1].plot(spk[cell, t_first:t_last])
# fig.suptitle('Post simulation')
# plt.plot(spikes_f[cell, t_first:t_last])
plt.show()


#%% plot for population's spikes
spk = spikes_i

def collect_spikes(spikes):
    l = []
    for i in range(spikes.shape[0]):
        l.append(np.where(spikes[i]==1)[0])
    return l

plt.eventplot(collect_spikes(spk))
plt.show()


# cells = np.where(spikes_e == 1)[0]
# for cell in range(spikes_e.shape[0]):
#     spks = np.where(spikes_e[cell] == 1)
#     plt.plot(time[spks], y=cell, 'k.', ms=5, color='black', marker='|', markeredgewidth=1.5)
# plt.show()

#%%
########################################################################################################################
### ARCHIVED CODE BELOW ################################################################################################
########################################################################################################################

# time = np.linspace(0, time_simulation, time_simulation*(1/dt))
# # for loop for numerical simulation of e and i pop evolution
# for n in range(0,len(t)-1):
#     # step change for excitatory neuron population
#     if Ve[n] < V_t:
#         Ve[n+1] = [Ve[n] + f_e(Ve[n], t[n]) * (t[n + 1] - t[n])]
#         h = n
#     else:
#         while h < h+2*(1/dt):
#             spikes_e[n] = 1
#             Ve[n] = -60
#             h += 1
#
#     # step change for inhibitory neuron population
#     if Vi[n] < V_t:
#         Vi[n+1] = [Vi[n] + f_i(Vi[n], t[n]) * (t[n + 1] - t[n])]
#         k = n
#     else:
#         while k < k+1*(1/dt):
#             spikes_i[n] = 1
#             Ve[n] = -60
#             k += 1

