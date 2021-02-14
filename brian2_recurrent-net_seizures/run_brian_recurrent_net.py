server = False

if server:
    location = '/home/pshah/Documents/code/'

    import sys; sys.path.append(location)
    import sys; sys.path.append('%sNeuronalModelling/brian2_recurrent-net_seizures/' % location)

else:
    location = '/Users/prajayshah/OneDrive - University of Toronto/PycharmProjects/'

    import sys; sys.path.append('%sutils_pj' % location)
    import sys; sys.path.append('%sNeuronalModelling/brian2_recurrent-net_seizures' % location)

# import statements
from brian2 import *
import funcs_pj as pj
import matplotlib.pyplot as plt
import numpy as np
from brian2_utils import *
from brian_recurrent_net import *

import pickle
import pandas as pd


#%% BUILD AND RUN NETWORK
runtime = 5*second
dt = 0.1*ms

# build network
record_id=[100, 4000, 2300, 3049, 494, 209, 250, 1505]
net, trace, s_mon, trace_ge, trace_gi, s_mon_p, Ce, Ci, Ge, Gi = build_network(record_id=record_id, inh_conn=0.2)

# run simulation
net.run(runtime, report='text')

# quick spike raster plot to initialize plotting
figure(figsize=[20,3])
plot(s_mon.t/ms, s_mon.i, ',k'); show()
spike_counts = s_mon.count
spike_counts_Hz = array(spike_counts/runtime)
avg=mean(spike_counts_Hz); print('average spiking rate of population: ', avg, 'Hz')

#%% save output of neuronal simulation as arrays and pickles

# save in pkl file
pickle.dump(trace.V, open("%sNeuronalModelling/sim-exports/trace_V.pkl" % location, "wb"))

# load pkl file
# trace = pickle.load(open("/home/pshah/Documents/code/neuronal-modelling/brian2/trace_V.pkl", "rb"))


#%% PROCESSING OF BRIAN OUTPUTS

# convert trace array into pandas DataFrame
trace_df = pd.DataFrame(trace.V.T, columns=record_id)


# create numpy array of spikes:
spike_array = np.empty([Ntotal, len(trace.t)])
for neuron in list(s_mon.all_values()['t'].keys()):
    print('processing neuron ', neuron+1, ' out of ', Ntotal, end='\r')
    spike_locs = [int(x) for x in list(s_mon.spike_trains()[neuron])/dt]
    spike_array[neuron, spike_locs] = 1

# collect 10ms spike bins  (with spike counts per bin as well)
spike_counts_binned = np.empty([Ntotal, int(runtime/dt/10)])
spike_raster_binned = np.empty([Ntotal, int(runtime/dt/10)])
for set in range(int(runtime/dt/10)):
    spike_counts_binned[:,set] = np.sum(spike_array[:,set*10:set*10+10], axis=1)  # count the number of spikes per neuron in the 10ms bin
    spike_raster_binned[np.where(spike_counts_binned[:,set] > 0), set] = 1  # set a positive number of spikes per neuron to 1

#%% calculate correlation coefficients
corr_mtx = np.corrcoef(spike_raster_binned[Ni:,:])
corr_values = corr_mtx[np.triu_indices(corr_mtx.shape[0], k=1)]
# not sure why but there are nan values coming up in the corr_values calculation
# remove nans from corr_values

corr_values = [value[~np.isnan(value)] for value in corr_values]
print(np.mean(corr_values))

#%% ANALYSIS OF NETWORK RESULTS - PLOTTING PLOTS

# plotting spikes as raster plots
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


# plot histogram of `spike_counts` per cell
plt.hist(spike_counts_Hz, bins=100, color='gray')
plt.axvline(x=avg, color='black')
plt.title
plt.show()


# plot of binned popn firing rate
popn_fr = np.sum([element for x, element in enumerate(spike_counts_binned) if x not in neurons_to_stim], axis=0)  # popn fr with stimulated neurons excluded
plt.figure(figsize=[30,5])
plt.plot(range(spike_counts_binned.shape[1]), popn_fr, linewidth=0.5)
plt.ylabel('total population spikes per 10ms bin')
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
plot_voltage(voltage_monitor=trace, neurons_to_plot=[100, 4000], alpha=0.7, xlimits=[500, 3500])


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
plot_inputs(e_monitor=trace_ge, i_monitor=trace_gi, neurons_to_plot=4000, alpha=0.5, xlimits=[500,1500])



#%% plot histogram of cell-to-cell correlation values TODO add time shuffling control to the quantification

# calculate and plot histogram of correlation coefficients
plt.figure(figsize=[5,5])
plt.hist(corr_values, bins=1000, density=True, color='grey', edgecolor='green')
plt.axvline(np.mean(corr_values), color='black')
plt.xlim(-0.1,0.1)
plt.show()

import seaborn as sns
sns.distplot(corr_values, kde=True, bins=1000, hist=True, hist_kws={'edgecolor': 'grey'}, color='black')
plt.axvline(np.mean(corr_values), color='green')
plt.xlim(-0.2,0.2)
plt.show()



#%% delete monitors before re running network
del(trace, s_mon)

