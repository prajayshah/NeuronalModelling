#%% testing out different ways of implementing a time varying stimulus in Brian2
from brian2 import *

#%%
start_scope()
runtime = 5*second
stim_onset = 2*second
stim_off = 4*second
dt = 0.1*ms
Ntotal = 4000


# define stimulus as a TimedArray of time dependent values
stim = np.empty([int(runtime/dt),Ntotal])
stim[int(stim_onset/dt):int(stim_off/dt), 3200:4000] = 5  # constant current input into the specified  cells at the specified onset and offset times
stimulus = TimedArray(stim*mV, dt=0.1*ms)


#
taum = 20*ms
taue = 5*ms
taui = 10*ms
Vt = -50*mV
Vr = -60*mV
El = -49*mV

eqs = '''
dv/dt  = (stimulus(t,i) + ge+gi-(v-El))/taum : volt (unless refractory)
dge/dt = -ge/taue : volt
dgi/dt = -gi/taui : volt
'''

P = NeuronGroup(Ntotal, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms,
                method='euler', dt=0.1*ms)

# initialization
P.v = 'Vr + rand() * (Vt - Vr) * 0.1'
# P.v = 'Vr'
P.ge = 0*mV
P.gi = 0*mV
# P.gapp = 0*mV

# make synaptic connections and specify the synaptic model
we = (60*0.27/10)*mV # excitatory synaptic weight (voltage)  (this is equal to [E_exc = 0mV - V_rest = -60mV] * g_exc = 0.27nS
wi = (-20*4.5/10)*mV # inhibitory synaptic weight
Ce = Synapses(P, P, on_pre='ge += we')
Ci = Synapses(P, P, on_pre='gi += wi')
Ce.connect('i<3200', p=0.2)
Ci.connect('i>=3200', p=1.0)
trace = StateMonitor(P, 'v', record=[1, 10, 400, 600, 1150, 1100, 3300, 3500])
trace_gi = StateMonitor(P, 'gi', record=[1, 10, 400, 600, 1150, 1100, 3300, 3500])
# trace_app = StateMonitor(P, 'gapp', record=[1, 10, 400, 600])
# trace_active = StateMonitor(P, 'active', record=[1, 10, 400, 600])
s_mon = SpikeMonitor(P)



# @network_operation
# def f(t):
#     if 2 * second < t < 4 * second:
#         P.active_[399:605] = 1
#         # print('At time:', t, G.Iapp[0])

net = Network(P, trace, s_mon)

net.run(5 * second, report='text')



#%%
figure(figsize=[20,3])
# plot(trace.t/ms, trace[10].v/mV)
plot(trace_gi.t/ms, trace_gi[1150].gi/mV)
# plot(trace_app.t, trace_app[400].Iapp)
# plt.plot(trace.t/ms, trace[100].v/mV)
xlabel('t (ms)')
ylabel('mV')
show()

#%%
def plot_raster(spike_monitor=s_mon, neurons_to_plot=[0,Ntotal]):
    plt.figure(figsize=[20,3])
    plt.plot(spike_monitor.t/ms, spike_monitor.i, ',k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.ylim(neurons_to_plot[0],neurons_to_plot[1])
    plt.show()
plot_raster()

