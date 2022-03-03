# Brian2 Example: CUBA

from brian2 import *

#%% original CUBA model from Brian2
taum = 20*ms
taue = 5*ms
taui = 10*ms
Vt = -50*mV
Vr = -60*mV
El = -49*mV

eqs = '''
dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
dge/dt = -ge/taue : volt
dgi/dt = -gi/taui : volt
'''

P = NeuronGroup(4000, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms,
                method='exact')

# initialization
P.v = 'Vr + rand() * (Vt - Vr)'
# P.v = 'Vr'
P.ge = 0*mV
P.gi = 0*mV

# make synaptic connections and specify the synaptic model
we = (60*0.27/10)*mV # excitatory synaptic weight (voltage)  (this is equal to [E_exc = 0mV - V_rest = -60mV] * g_exc = 0.27nS
wi = (-20*4.5/10)*mV # inhibitory synaptic weight
Ce = Synapses(P, P, on_pre='ge += we')
Ci = Synapses(P, P, on_pre='gi += wi')
Ce.connect('i<3200', p=0.02)
Ci.connect('i>=3200', p=0.02)
trace = StateMonitor(P, 'v', record=[1, 10, 400, 600])
s_mon = SpikeMonitor(P)

run(1 * second)

plt.figure(figsize=[20,3])
# plot(trace.t/ms, trace[10].v/mV)
plot(trace.t/ms, trace[400].v/mV)
# plt.plot(trace.t/ms, trace[100].v/mV)
xlabel('t (ms)')
ylabel('v (mV)')
show()

plt.plot(s_mon.t/ms, s_mon.i, ',k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show()





