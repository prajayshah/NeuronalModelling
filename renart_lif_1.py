# replicating the Renart et al., 2010, Science balanced E-I network of spiking neurons

import numpy as np
import matplotlib.pyplot as plt

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

def rectify(input):
    return np.multiply(input, (input > 0))


#%%      ###### TRYING OUT AN EXCITATORY ONLY NETWORK FOR THE SYNAPTIC CURRENTS EQ'S ##### currently working on this
########################################################################################################################
Cm = 0.25e-6 #mF
gL = 16.7e-6 #mS
tau = 15  #ms -- membrane constant, defined by Cm/gL
time_simulation = 0.5*100 #ms
dt = 0.05  #ms
V_t = -40. # mV threshold
Vl = -70. #mV -- resting potential
V_rev_E = 0 #mV
Ne = 3
Ve = np.full([Ne, int(time_simulation*(1/dt))], Vl); spikes = np.zeros([Ne, int(time_simulation*(1/dt))])

# for the purpose of this network, just define some random spikes to add throughout the simulation. this should get some activity input going.
spikes_e = set_to_one(spikes, probability=0.001)
spikes_f = spikes_e  # copy of initialized spikes_e

# variables for connections/synapses
probability=1.0  # general probability of connectivity, same for all types of connections
p_ij_eE = np.zeros((Ne, Ne)); p_ij_eE = set_to_one(p_ij_eE, probability=probability)

# variables for synapse strength matrices
g_ij_eE = np.zeros((Ne, Ne)); g_ij_eE = set_from_gaus(g_ij_eE, mean=2.4e-3)  ## BEWARE THAT YOU'VE CHANGED THE MEAN SYNAPTIC CONDUCTANCE VALUE TO BE SIGNIFICANTLY HIGHER THAN 2.4nS

#%% initial plot
t_first = 0
t_last = int(time_simulation*(1/dt))
cell = 1
fig, axs = plt.subplots(2,1)
axs[0].plot(Ve[cell, t_first:t_last])
axs[0].set_ylim(Vl-10,V_t+10)
axs[1].plot(spikes_e[cell, t_first:t_last])
axs[1].plot(spikes_f[cell, t_first:t_last])
fig.suptitle('Pre simulation')
plt.show()

#%% set up equations and run simulation

# eqs and variables for synaptic gate and synaptic conductance delay
s_ij_eE = np.zeros((Ne, Ne, int(time_simulation*(1/dt))))
x_ij_eE = np.zeros((Ne, Ne, int(time_simulation*(1/dt))))
d_ij_eE = np.zeros((Ne, Ne))  # synaptic conductance delay - !!!! - this needs to be a specific uniform distribution (see Renart suppl)
tau_d = 5 #ms
tau_r = 1 #ms

#
def synaptic_gate(i, j, n):
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

    # spk = 0
    # for spike in spikes_e[j]:
    #     spk += delta(n - spike - d_ij_eE[i, j])

    dxdt = lambda x : 1/tau_r * spk - x
    dsdt = lambda x, s: 1 / tau_d * (x - s)

    x_ij_eE[i, j, n + 1] = x_ij_eE[i, j, n] + dxdt(x = x_ij_eE[i, j, n]) * (time[n + 1] - time[n])
    s_ij_eE[i, j, n + 1] = s_ij_eE[i, j, n] + dsdt(x = x_ij_eE[i, j, n+1], s = s_ij_eE[i, j, n]) * (time[n + 1] - time[n])

    # # different approach below - two coupled differential equations + odeEuler, where y = [s_ij_aB, x_ij_aB]
    # s = y[0]
    # # x = y[1]
    #
    # x = 1/tau_r*spk + np.exp(-t)
    # # dsdt = 1/tau_d * (s - x)
    # # dxdt = 1/tau_r * spk - x
    #
    # odeEuler(f=dsdt, y0=s, t=t)

    return s_ij_eE[i, j, n+1]

def synaptic_current(i, n, Ve):
    '''

    :param i: cell i from the network - post-synaptic cell
    :param n: current time step of the network
    :return: total synaptic current for post-synaptic cell i
    '''

    IeE_t = 0
    for j in np.arange(Ne):
        gate = synaptic_gate(i, j, n)
        IeE_t += p_ij_eE[i,j] * g_ij_eE[i,j] * gate

    IeE_t = -IeE_t * (Ve - V_rev_E)

    return IeE_t


# equations
# IeE = lambda  t: -1 * [p_ij_eE @ g_ij_eE @ s_ij_eE(t)] * (Ve - V_rev_E)
dVe_dt = lambda Ve_i, i, n: 1/tau * (Vl - Ve_i + 1/gL * synaptic_current(i, n, Ve_i))

''' trouble shooting with a constant current input source below - keep commented out until needed again.
def constant_current(n, n_start, n_end, amount):
    if n_start < n < n_end:
        return amount
    else:
        return 0
input_amps = gL * 2 #1 = 1mV 
dVe_dt_input = lambda Ve_i, i, n: 1/tau * ((Vl - Ve_i) + 1/gL * constant_current(n, 40, 200, input_amps))



# solution for a constant current input to the LIF dv/dt can be derived analytically
# V(t) = Vl + constant_current(0)*1/gL*(1 - np.exp(-t/(Cm/gL))); where t=0 is when the input is on
dV = (1/15 * (-70 - (-65)) + 5) * 0.05
# V = -70 + input_amps*1/gL
'''

## RUNNING SIMULATION -
# TODO gotta work out why the leak conductance is overpowering the dynamics of the system,
#  first step is to definitely make sure that the units of the synaptic currents line up with the units of the leak conductance
# - units look probably okay, fixed a couple more mistakes in the formulas as well. is the leak conductance still dominating the dynamics of the system?


time = np.linspace(0, time_simulation, int(time_simulation*(1/dt)))
for n in range(int(2 * (1 / dt)),len(time)-1):  # note that there is a 40ms gap to avoid having negative check-backs for the first 40ms in the first if statement
    for i in np.arange(Ve.shape[0]): ## select a neuron i to calculate dVe/dt for

        # print message about progress of simulation
        msg = ' -- Working on cell %d out of %d, at time step %d out of %d' % (i, Ve.shape[0], n, len(time))
        print(msg, end='\r')

        # if - else for sub-threshold vs spiking behaviours
        if sum(spikes_e[i, int(n - (0 * (1 / dt)))-1:n]) > 0:  # TODO IS THIS ACTUALLY WORKING NOW????? # just set refrac period to zero
            print('refrac', i, n)
            Ve[i, n] = -60  #mV -- refractory period (should be -60mV but might have been changed it for trialling)

        elif Ve[i, n] < 0:  # should be below 'V_t'
            # syn_input = synaptic_current(i, n)  # calculate synaptic input current
            # run voltage update equation - Euler's method using dVe_dt defined above
            Ve_change = dVe_dt(Ve[i,n], i, n)*(time[n+1] - time[n])
            if Ve_change < -5:
                # print('woooo +vee')
                print('wooooaaahhh negativeeee where from??', i, n, Ve_change)
            Ve[i,n+1] = Ve[i,n] + Ve_change  # where Ve_change = dVe_dt(Ve[i,n], i, n)*(time[n+1] - time[n])

            if Ve[i, n+1] > V_t:
                print('Spike! Voltage: ', Ve[i, n+1])
                print('-- Neuron index', i, 'time index', n+1)
                # print(synaptic_current(i, n, Ve[i,n]))
                # print('-- Spike!')
                spikes_f[i, n+1] = 1        # TODO IS THIS ACTUALLY WORKING NOW?????
            # h = n

        # else:
        #     if h < (h + 2 * (1 / dt)):
        #         spikes_e[i, n] = 1
        #         Ve[i, n] = -60
        #         h += 1

        # # print message about progress of simulation
        # msg = ' -- Completed cell %d out of %d, at time %d out of %d' % (i, Ve.shape[0], n, len(time))
        # print(msg, end='\r')


#%% plotting some voltage traces from the simulations above
t_first = 0
t_last = int(time_simulation*(1/dt))
cell = 1

fig, axs = plt.subplots(2,1)
axs[0].plot(Ve[cell, t_first:t_last])
axs[0].set_ylim(Vl-10,V_t+10)
axs[1].plot(spikes_e[cell, t_first:t_last])
fig.suptitle('Post simulation')
# plt.plot(spikes_f[cell, t_first:t_last])
plt.show()

sum(sum(spikes_e - spikes_f))
