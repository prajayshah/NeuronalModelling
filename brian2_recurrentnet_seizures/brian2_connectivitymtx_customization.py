## Using a custom weight matrix with CUBA example from Brian2 docs

import random
import numpy as np
import matplotlib.pyplot as plt
from brian2_recurrentnet_seizures.brian2_utils import *

# recurrent layer
Nn = 4000  # number of neurons in the recurrent layer
W = np.zeros([Nn, Nn])  # matrix of recurrent connection weights, should be a Nn x Nn size array
# W = np.diag(np.ones(Nn)) # -- self connections

def resetw(W=W):
    W = np.zeros([W.shape[0], W.shape[1]])
    return W

# modifying recurrent connection weights as you want to for the network


####################################################
# %% fully random connectivity matrix
####################################################
W = resetw()

ee_p = 0.02  # probability of e-e connections
i_p = 0.20  # probability of Inh. neurons

# set E -> all connectivity
for i in range(0, int(0.8 * Nn)):
    j = random.sample(range(len(W[i])), int(
        ee_p * len(W[i])))  # select random indexes to setup as connections based on selected probability
    W[i, j] = 1

# set I -> all connectivity
for i in range(int(0.8 * Nn), Nn):
    j = random.sample(range(len(W[i])), int(
        i_p * len(W[i])))  # select random indexes to setup as connections based on selected probability
    W[i, j] = 1


####################################################
# %% building a spatially localized connectivity matrix
####################################################

W = resetw()

p_e = 0.2  # probability of e-e connections
p_i = 0.2  # probability of Inh. neurons

Ne = random.sample(range(Nn), int(0.8*Nn))
Ni = [x for x in range(Nn) if x not in Ne]

# need to intersperse E and I neurons throughout the whole cell population
for i in range(Nn):
    # select random indexes to setup as connections based on selected probability
    # set E --> all connectivity
    if i in Ne:
        j = [int(target) for target in np.random.normal(i,200, int(p_e*len(W[i]))) if 0 < target < Nn]
        W[i, j] = 1

    # set I --> all connectivity
    elif i in Ni:
        j = [int(target) for target in np.random.normal(i, 200, int(p_i * len(W[i]))) if 0 < target < Nn]
        W[i, j] = 1

i, j = W[Ni].nonzero()
sources = [Ni[x] for x in i]
targets = [range(Nn)[x] for x in j]



# %% plot connectivity matrix
plot_connectivity_matrix(conn_matrix=W)

# %%
# Brian2 network

from brian2 import *

#%% original CUBA model from Brian2
start_scope()

runtime = 1 * second

taum = 20*ms
taue = 5*ms
taui = 10*ms
Vt = -50*mV
Vr = -60*mV
El = -49*mV

eqs = """
dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
dge/dt = -ge/taue : volt
dgi/dt = -gi/taui : volt
"""

P = NeuronGroup(4000, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms,
                method='exact')

# initialization
P.v = "Vr + rand() * (Vt - Vr)"
# P.v = 'Vr'
P.ge = 0*mV
P.gi = 0*mV

# make synaptic connections and specify the synaptic model
w_e = (60*0.27/10)*mV # excitatory synaptic weight (voltage)  (this is equal to [E_exc = 0mV - V_rest = -60mV] * g_exc = 0.27nS
w_i = (-20*4.5/10)*mV # inhibitory synaptic weight


# using custom synaptic connectivity matrix from above
Ce = Synapses(P, P, on_pre='ge += w_e')
i, j = W[Ne].nonzero()
sources = [Ne[x] for x in i]
targets = [range(Nn)[x] for x in j]
Ce.connect(i=sources, j=targets)

Ci = Synapses(P, P, on_pre='gi += w_i')
i, j = W[Ni].nonzero()
sources = [Ni[x] for x in i]
targets = [range(Nn)[x] for x in j]
Ci.connect(i=sources, j=targets)


trace = StateMonitor(P, 'v', record=[1, 10, 400, 600])
s_mon = SpikeMonitor(P)

run(runtime, report='text')

# %%
plot_raster(spike_monitor = s_mon)

# %%
figure(figsize=[20,3])
plot(s_mon.t/ms, s_mon.i, ',k')
#     plt.suptitle('Input rate (Hz): %s' % (input_rate))
plt.suptitle(title)
plt.show()
spike_counts = s_mon.count
spike_counts_Hz = array(spike_counts/runtime)
avg=mean(spike_counts_Hz); print('average spiking rate of population: ', np.round(avg, 2), 'Hz')


# %% 2.1) low rank connectivity - outer product of two vectors

def setrandomvector(min, max, N):
    return (max - min) * np.random.random(N) + min

N = 2000

v1 = setrandomvector(0.5, 1.5, N)
v2 = setrandomvector(0.5, 1.5, N)

mtx = np.outer(v1, v2)  # structured connectivity matrix

plot_connectivity_matrix(conn_matrix=mtx.T, color_lim=[0.5, 1.5], colorbar=True)