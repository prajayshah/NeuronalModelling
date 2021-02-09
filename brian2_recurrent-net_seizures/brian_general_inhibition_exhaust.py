#%% modification of CUBA model to include an exhaustible inhibition model
# - needed to change to Euler solution, the exact solution wasn't working because the eq.s were not linearizable
from brian2 import *

#%%

taum = 20*ms
taue = 5*ms
taui = 10*ms
tau_g = 25*ms  # need to find better justification for the proper timescale of this for the actual simulation
Vt = -50*mV
Vr = -60*mV
El = -60*mV
gi_t = 5*mV
factor=1*mV

we = 5.62*mV # excitatory synaptic weight (voltage)  (this is equal to [E_exc = 0mV - V_rest = -60mV] * g_exc = 0.27nS
wi = -10*mV # inhibitory synaptic weight
wx = 50.0*mV  # excitatory input neurons

runtime = 5*second
dt = 0.1*ms

# EXTERNAL STIMULUS (I KNOW THAT THE ORDER OF THIS IS REALLY WEIRD BUT THIS SECTION AND THE MODEL EQS CANNOT BE ADDED TO THE NETWORK BUILD FUNCTION DEFINITION SINCE IT KICKS UP ERRORS IN BRIAN
stim_external = True
# define external stimulus as a TimedArray of time dependent values
stim_onset = 2 * second
stim_off = 2.5 * second
stim = np.empty([int(runtime / dt), 20])  # where 20 = total number of neurons
stimulus = TimedArray(stim * mvolt, dt=0.1 * ms)  # constant current input into the specified  cells at the specified onset and offset times
if stim_external:
    neurons_to_stim = arange(0,5)
    stim[int(stim_onset / dt):int(stim_off / dt), neurons_to_stim] = 5
    stimulus = TimedArray(stim * mvolt,
                          dt=0.1 * ms)  # constant current input into the specified  cells at the specified onset and offset times


# NOTE THE Z VARIABLE SEEMS TO BE INVERSED -- IT'S GOING TO ZERO WITH NO
# INH. ACTIIVTY AND GOING TOWARDS 1 WITH HIGH INH. ACTIVITY

eqs = '''
dv/dt  = (stimulus(t,i) + ge + z * gi - (v-El))/taum : volt (unless refractory)

dge/dt = -ge/taue : volt
dgi/dt = -gi/taui : volt

#gi_z = z * gi : volt
dz/dt = 1/tau_g * (z_inf - z) : 1
z_inf = 1/(1 + exp(-2*10*gi_diff)) : 1
gi_diff = (gi_t + gi)/factor : 1  # note that this is addition because gi has negative weight
'''



# the goal is to try model a process whereby the failure of inhibition occurs
# taking inspiration from Liou et al., this can be achieved by relating the reversal potential for Inh. to the Exc. neuron's chloride gradient according to the Nernst equation for E_Cl (chloride)
# Liou also presents a generalized model of exhaustible inhibition where a factor z is modelled using first order kinetics and the Heaviside step function
# you're going to need to test this independently to make sure that it does work as expected -- i.e. more Inh --> lower Inh effectiveness

def build_network(record_id):
    start_scope()
    
    P = NeuronGroup(20, eqs, threshold='v>Vt', reset='v = Vr', refractory=1*ms,
                    method='euler')
    Pi = P[:10]
    Pe = P[10:]

    # initialization
    P.v = 'Vr + rand() * (Vt - Vr)'
    # P.v = 'Vr'
    P.ge = 0*mV
    P.gi = 0*mV
    P.z  = 1

    # make synaptic connections and specify the synaptic model

    Ce = Synapses(Pe, P, on_pre='ge+=we')
    Ci = Synapses(Pi, Pe, on_pre='gi+=wi')
    Ce.connect(p=0.2)
    Ci.connect(p = 0.8)

    # add a time varying input that increases in strength of firing for a short period of time


    # poisson inputs into I neurons
    X = PoissonGroup(1, rates=10. * Hz)
    X_2 = PoissonGroup(1, rates=10. * Hz)
    CiX = Synapses(X, Pi, on_pre='ge+=wx')
    CeX = Synapses(X_2, Pe, on_pre='ge+=wx')
    CiX.connect(p=1.0)
    CeX.connect(p=1.0)

    trace = StateMonitor(P, 'v', record=record_id)
    trace_z = StateMonitor(P, 'z', record=record_id)
    trace_gi = StateMonitor(P, 'gi', record=record_id)
    trace_gi_diff = StateMonitor(P, 'gi_diff', record=record_id)
    s_mon = SpikeMonitor(P)
    x_mon = SpikeMonitor(X)

    net = Network(collect())

    return net, trace, s_mon, trace_z, trace_gi, x_mon, trace_gi_diff


'''
Example usage of the script:

net, trace, s_mon, trace_z, trace_gi, x_mon, trace_gi_diff = build_network(record_id)

net.run(5 * second, report='text')


plt.figure(figsize=[20,3])
# plot(trace.t/ms, trace[10].v/mV)
plot(trace.t/ms, trace[11].v/mV)
# plt.plot(trace.t/ms, trace[100].v/mV)
xlabel('t (ms)')
ylabel('mV')
show()

plt.figure(figsize=[20,3])
plt.plot(s_mon.t/ms, s_mon.i, ',k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.ylim(0,20)
plt.show()

plt.figure(figsize=[20,3])
plot(trace_z.t/ms, trace_z[11].z)
xlabel('t (ms)')
ylabel('z')
show()

plt.figure(figsize=[20,3])
plot(trace_gi.t/ms, trace_gi[11].gi)
xlabel('t (ms)')
ylabel('gi')
show()

'''

