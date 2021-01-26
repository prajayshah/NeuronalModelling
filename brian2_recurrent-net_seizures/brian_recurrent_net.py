# import sys; sys.path.append('/Users/prajayshah/OneDrive - University of Toronto/PycharmProjects/utils_pj')
# import sys; sys.path.append('/Users/prajayshah/OneDrive - University of Toronto/PycharmProjects/NeuronalModelling/brian2_recurrent-net_seizures')

import sys; sys.path.append('/home/pshah/Documents/code/')
import sys; sys.path.append('/home/pshah/Documents/code/neuronal-modelling/brian2/')


from brian2_utils import *
from brian2 import *
import funcs_pj as pj
import matplotlib.pyplot as plt
import numpy as np

import pickle
import pandas as pd

#%% INITIALIZING THE MODEL
model_name = 'rec_sz1'
print('Welcome. This model is called ...', model_name)

#  experimenting with different synapse equations
#
# # difference of exponentials synapse
x = np.linspace(0, 40, 500)
y = (3 * (np.exp(-(x/2)) - np.exp(-(x/3))))
plt.plot(x, y)
plt.show()



# setup for network:
# Parameters
Cm = 0.25 * nfarad
gL = 16.7 * nsiemens

Ne = 4000  # number of E neurons
Ni = 1000 # number of I neurons
Nx = 4000  # number of X (external population) neurons
Ntotal  = Ne+Ni

tau = 15 * ms
Vl = -70 * mV  # resting potential
V_t = -50 * mV  # threshold
V_refrac = -60 * mV  # refractory voltage

# Time constants
tau = 5 * ms
tau_d = 5 * ms
# Reversal potentials
Ee = 0 * mV
V_rev_X = 0 * mV
Ei = -80 * mV
w_e = 2.4 * nsiemens  # excitatory synaptic weight
w_i = 40 * nsiemens  # inhibitory synaptic weight
w_x = 5.4 * nsiemens  # external input synaptic weight

runtime = 5*second
dt = 0.1*ms


# EXTERNAL STIMULUS (I KNOW THAT THE ORDER OF THIS IS REALLY WEIRD BUT THIS SECTION AND THE MODEL EQS CANNOT BE ADDED TO THE NETWORK BUILD FUNCTION DEFINITION SINCE IT KICKS UP ERRORS IN BRIAN
stim_external = False
# define external stimulus as a TimedArray of time dependent values
stim_onset = 1 * second
stim_off = 1.5 * second
stim = np.empty([int(runtime / dt), Ntotal])
stimulus = TimedArray(stim * amp, dt=0.1 * ms)  # constant current input into the specified  cells at the specified onset and offset times
if stim_external:
    neurons_to_stim = arange(200,210)
    stim[int(stim_onset / dt):int(stim_off / dt), neurons_to_stim] = 5
    stimulus = TimedArray(stim * amp,
                          dt=0.1 * ms)  # constant current input into the specified  cells at the specified onset and offset times

# The model
eqs = Equations('''
dV/dt = 1/Cm * (stimulus(t,i) + gL*(Vl - V) + ge*(Ee-V) + gi*(Ei-V)) : volt (unless refractory)
dge/dt = -ge/tau_d : siemens
dgi/dt = -gi/tau_d : siemens
''')

#%% NETWORK BUILD PART 2 - SETTING UP BRIAN STRUCTURE
def build_network(record_id, inh_conn=0.2):

    start_scope()

    G = NeuronGroup(Ntotal, model=eqs, threshold='V > V_t', refractory=2*ms, reset= 'V = V_refrac',
                    method='euler', dt=0.1*ms)
    Ge = G[Ni:Ntotal]
    Gi = G[:Ni]

    # defining synapses and synaptic connections
    Ce = Synapses(Ge, G, on_pre='ge+=w_e')
    Ci = Synapses(Gi, G, on_pre='gi+=w_i')
    Ce.connect(p=0.2)  # Excitatory connectivity
    Ci.connect(p=inh_conn)  # Inhibitory connectivity

    #
    # BACKGROUND Poisson input
    P = PoissonGroup(Nx, rates=2.5*Hz, dt=0.1*ms)
    CX = Synapses(P, G, on_pre='ge+=w_x')
    CX.connect(p=0.8)  # Excitatory external drive connectivity

    # CeX = Synapses(P, Ge, on_pre='ge+=w_x')
    # CiX = Synapses(P, Gi, on_pre='ge+=w_x')
    # CeX.connect(p=0.2)
    # CiX.connect(p=0.2)


    # Initialization
    G.V = 'Vl'
    G.ge = 0
    G.gi = 0

    # Setup a few monitors
    trace = StateMonitor(G, 'V', record=record_id)
    trace_ge = StateMonitor(G, 'ge', record=record_id)
    trace_gi = StateMonitor(G, 'gi', record=record_id)
    s_mon = SpikeMonitor(G)
    s_mon_p = SpikeMonitor(P)


    net = Network(collect())

    return net, trace, s_mon, trace_ge, trace_gi, s_mon_p


#%% BUILD AND RUN NETWORK
# build network
record_id=[100, 400, 3010, 4349, 4928, 29, 250, 283]
net, trace, s_mon, trace_ge, trace_gi, s_mon_p = build_network(record_id=record_id, inh_conn=0.5)

# run simulation
net.run(runtime, report='text')

# quick spike raster plot to initialize plotting
plot(s_mon.t/ms, s_mon.i, ',k'); show()
spike_counts = s_mon.count
spike_counts_Hz = array(spike_counts/runtime)
avg=mean(spike_counts_Hz); print('average spiking rate of population: ', avg, 'Hz')

#%% save output of neuronal simulation as arrays and pickles

# save in pkl file
pickle.dump(trace.V, open("/home/pshah/Documents/code/neuronal-modelling/brian2/trace_V.pkl", "wb"))

#%% convert trace array into pandas DataFrame
trace_df = pd.DataFrame(trace.V.T, columns=record_id)


#%% ANALYSIS OF NETWORK RESULTS - PLOTTING PLOTS

# plotting spikes
def plot_raster(spike_monitor, neurons_to_plot=None, xlimits=None):
    """
    Plot a raster plot using the spike monitor object.

    :param spike_monitor: spike monitor object recorded from a brian2 simulation run
    :param neurons_to_plot: range of neurons to plot for spike raster (y axis limits)
    :param xlimits: limit of x axis (time, in milliseconds)
    :return: raster plot
    """
    figure(figsize=[20,3])
    plot(spike_monitor.t/ms, spike_monitor.i, ',k')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    if xlim:
        xlim(xlimits)
    if neurons_to_plot:
        ylim(neurons_to_plot[0],neurons_to_plot[1])
    show()
# plot_raster(spike_monitor=s_mon, neurons_to_plot=[2000,2200], xlimits=[500,1500])
plot_raster(spike_monitor=s_mon)

# plot the average firing rate distribution
# spike_counts = np.zeros([Ntotal])  # spike counts ordered by neuron IDs
# for neuron in list(s_mon.all_values()['t'].keys()):
#     print(neuron, end='\r')
#     spike_counts[neuron] = s_mon.spike_trains()[neuron].__len__()


# brian has a built in method for this as well:
spike_counts = s_mon.count
spike_counts_Hz = array(spike_counts/runtime)
avg=mean(spike_counts_Hz); print('average spiking rate of population: ', avg, 'Hz')


# plot histogram of `spike_counts`
plt.hist(spike_counts_Hz, bins=100, color='gray')
plt.axvline(x=avg, color='black')
plt.title
plt.show()

#%% plotting voltage traces
colors = []
for i in range(0, len(record_id)):
    colors.append(pj.generate_new_color(colors, pastel_factor=0.2))

def plot_voltage(voltage_monitor, neurons_to_plot, alpha, xlimits=[]):
    plt.figure(figsize=[30,5])
    for neuron in neurons_to_plot:
        plt.plot(voltage_monitor.t/ms, voltage_monitor[neuron].V/mV, alpha=alpha, color=colors[neurons_to_plot.index(neuron)])
        # add vertical lines at the spike times for individual neurons (using the values from the spike monitor)
        plt.vlines(x=np.int_(array(np.round(s_mon.spike_trains()[neuron], 3))*1000), ymin=-50, ymax=0, alpha=alpha, color=colors[neurons_to_plot.index(neuron)])
    if xlim:
        xlim(xlimits)
    plt.xlabel('t (ms)')
    plt.ylabel('v (mV)')
    plt.show()
plot_voltage(voltage_monitor=trace, neurons_to_plot=[250, 283], alpha=0.7, xlimits=[500, 3500])


#%% plotting E and I inputs
def plot_inputs(e_monitor, i_monitor, neurons_to_plot, alpha, xlimits=None):
    plt.figure(figsize=[30,5])
    plt.plot(e_monitor.t/ms, e_monitor[neurons_to_plot].ge, alpha=alpha, color='green')
    plt.plot(i_monitor.t/ms, i_monitor[neurons_to_plot].gi, alpha=alpha, color='red')
    if xlimits:
        plt.xlim(xlimits)
    plt.xlabel('t (ms)')
    plt.ylabel('current (siemens)')
    plt.show()
plot_inputs(e_monitor=trace_ge, i_monitor=trace_gi, neurons_to_plot=3010, alpha=0.5, xlimits=[500,1500])


#%% plot X=external inputs events TODO



#%% plot histogram of cell-to-cell correlation values TODO add time shuffling control to the quantification
# first convert the spiking monitor into a numpy array
spike_raster = np.zeros([Ntotal, int(runtime/(10*ms))+1])
spike_train = np.zeros(int(runtime/(10*ms))+1)
for neuron in list(s_mon.spike_trains()):
    print(neuron, end='\r')
    spikes = array(np.round(s_mon.spike_trains()[neuron], 2))*100
    spike_train[np.int_(spikes)] = 1
    spike_raster[neuron, :] = spike_train

# calculate and plot histogram of correlation coefficients
corr_mtx = np.corrcoef(spike_raster)
corr_values = corr_mtx[np.triu_indices(corr_mtx.shape[0], k=1)]
plt.hist(corr_values, bins=5000, density=True, color='grey', edgecolor='green')
plt.axvline(np.mean(corr_values), color='black')
plt.show()

import seaborn as sns
sns.distplot(corr_values, kde=True, bins=5000, hist=True, hist_kws={'edgecolor': 'grey'}, color='black')
plt.axvline(np.mean(corr_values))
plt.show()

# depracated below
for i in exp_obj.spks_smooth:
    b = np.corrcoef(i)
    a = np.triu(b)
    np.fill_diagonal(a,0)
    c = list(a.flat)
    d = [i for i in c if i != 0]

    # plot histogram of correlation coefficient densities
    n, bins, patches = plt.hist(d, 400, density=True)
    plt.axvline(np.mean(d))
plt.show()


#%% delete monitors before re running network
del(trace, s_mon)

