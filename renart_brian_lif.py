from brian2 import *
import matplotlib.pyplot as plt

#%%
# Parameters
Cm = 0.25*nfarad
gL = 16.7*nsiemens

Ne = 800  # number of E neurons
Ni = 200  # number of I neurons
Nx = 800  # number of X (external population) neurons

tau = 15*ms
Vl = -70*mV  # resting potential
V_t = -50*mV  # threshold
V_refrac = -60*mV  # refractory voltage

# Time constants
taue = 5*ms
taui = 10*ms
tau_r = 1*ms
tau_d = 5*ms
# Reversal potentials
V_rev_E = 0*mV
V_rev_X = 0*mV
V_rev_I = -80*mV
geE = 2.4*nS  # excitatory synaptic weight
geI = 40*nS  # inhibitory synaptic weight
giI = 40*nS  # inhibitory synaptic weight
giE = 4.8*nS  # inhibitory synaptic weight
geX = 5.4*nS  # inhibitory synaptic weight
giX = 5.4*nS  # inhibitory synaptic weight


# The model
eqs = Equations('''
dVe/dt = 1/tau * ((Vl - Ve) + IeE * (V_rev_E - Ve) + IeI * (V_rev_I - Ve) + IeX * (V_rev_X - Ve)): volt (unless refractory)
dVi/dt = 1/tau * ((Vl - Vi) + IiE * (V_rev_E - Vi) + IiI * (V_rev_I - Vi) + IiX * (V_rev_X - Vi)): volt (unless refractory)

dIeE/dt = (geE * s_e) : siemens 
dIeI/dt = (geI * s_e) : siemens
dIeX/dt = (geX * s_e) : siemens

dIiE/dt = (giE * s_i) : siemens 
dIiI/dt = (giI * s_i) : siemens
dIiX/dt = (giX * s_i) : siemens

s_e : 1
s_i : 1
''')


G = NeuronGroup(100, model=eqs, threshold='v > V_t', refractory=2*ms, reset= 'v = V_refrac',
                method='euler')
Ge = G[:80]
Gi = G[20:]



#%%
# defining synapses and synaptic connections
synaptic_eqs = '''ds_e_syn/dt = 1/tau_d*(x_e - s_e_syn) : 1 (clock-driven)
dx_e/dt = 1/tau_r*(spk_e - x_e) : 1 (clock-driven)
ds_i_syn/dt = 1/tau_d*(x_i - s_i_syn) : 1 (clock-driven)
dx_i/dt = 1/tau_r*(spk_i - x_i) : 1 (clock-driven)
spk_e : 1
spk_i : 1
'''
Ce = Synapses(Ge, G, model=synaptic_eqs, on_pre='spk_e+=1')
Ci = Synapses(Gi, G, model=synaptic_eqs, on_pre='spk_i+=1')
Ce.connect(p=0.02)
Ci.connect(p=0.02)


#%%
# defining Poisson input
P = PoissonGroup(20, rates=10*Hz)
CeX = Synapses(P, Ge, on_pre='spk_e += 1')
CiX = Synapses(P, Gi, on_pre='spk_i += 1')
CeX.connect(p=0.2)
CiX.connect(p=0.05)


# Initialization
G.Ve = 'Vl'
G.Vi = 'Vl'
G.IeE = 0
G.IeI = 0
G.IeX = 0
G.IiX = 0
G.IiE = 0
G.IiI = 0

#%% Record a few traces
trace = StateMonitor(G, 'Ve', record=[1, 10, 100])
s_mon = SpikeMonitor(G)
run(1 * second, report='text')

#%% plotting
plt.plot(trace.t/ms, trace[1].v/mV)
plt.plot(trace.t/ms, trace[10].v/mV)
plt.plot(trace.t/ms, trace[100].v/mV)
plt.xlabel('t (ms)')
plt.ylabel('v (mV)')
plt.show()

# plotting spikes
plt.plot(s_mon.t/ms, s_mon.i, ',k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.suptitle('Spiking behaviour')
plt.show()