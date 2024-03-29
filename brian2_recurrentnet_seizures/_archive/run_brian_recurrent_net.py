server = True

if server:
    location = '/home/pshah/Documents/code/'
    import sys; sys.path.append('%sutils_praj' % location)
    sys.path.append('%sNeuronalModelling/' % location)

else:
    location = '/Users/prajayshah/OneDrive - University of Toronto/PycharmProjects/'
    import sys; sys.path.append('%sutils_praj' % location)

# import statements
from brian2 import *
from brian2_recurrentnet_seizures.brian2_utils import *
from brian2_recurrentnet_seizures.brian_recurrent_net_main import *

import pickle
import pandas as pd
import numpy as np
import random










#%% BUILD AND RUN NETWORK
spike_raster_binned_multi = None

runtime = 3*second
dt = 0.1*ms
inh_conn = 0.2
Ntotal = 5000
Ne = Ntotal * 0.8
Ni = Ntotal * 0.2
input_rate = 2.5

# build network
record_id=[100, 4000, 4300, 4400, 2300, 3049, 494, 209, 250, 1505]
net, W, trace, s_mon, trace_ge, s_mon_p, Ce, Ci, Ge, Gi, G, trace_z, trace_gi, trace_gi_diff = \
    b2_network(Ntotal=Ntotal, record_id=record_id, runtime = runtime, inh_conn=inh_conn, input_rate=input_rate,
               stim_external = False, neurons_to_stim = arange(208, 214))

# run simulation
net.run(runtime, report='text')

# quick spike raster plot to initialize plotting
figure(figsize=[20,3])
plot(s_mon.t/ms, s_mon.i, ',k'); plt.show()
spike_counts = s_mon.count
spike_counts_Hz = array(spike_counts/runtime)
avg=mean(spike_counts_Hz); print('average spiking rate of population: ', avg, 'Hz')

# plt.plot(s_mon.t/ms, array(s_mon.i), ',k')
# plt.show()

#%% save output of neuronal simulation as arrays and pickles

# save in pkl file
pickle.dump(trace.V, open("%sNeuronalModelling/sim-exports/trace_V.pkl" % location, "wb"))

# load pkl file
# trace = pickle.load(open("/home/pshah/Documents/code/neuronal-modelling/brian2/trace_V.pkl", "rb"))


#%% PROCESSING OF BRIAN OUTPUTS

# convert trace array into pandas DataFrame
trace_df = pd.DataFrame(trace.V.T, columns=record_id)

# make spike raster bins
spike_monitor_trains = [(array / dt).astype(int64) for array in list(s_mon.spike_trains().values())]

spike_monitor_trains_e = [spike_monitor_trains[i] for i in range(len(spike_monitor_trains)) if i in range(Ne)]

# create numpy array of spikes:
binsize = 100  # ms
spike_array, spike_counts_binned, spike_raster_binned = make_spike_array(spike_monitor_trains=spike_monitor_trains_e,
                                                                         ntotal=len(Ne), dt=dt, binsize=binsize,
                                                                         rectime=runtime)
plot_firing_rate(spike_raster_binned, binsize_sec=0.01, title='Population Firing rate')

## plot histogram of firing rates for all neurons in network
# plot_firing_rate_histogram(spike_array = spike_array, len_sec = runtime / second)

if spike_raster_binned_multi is None:
    spike_raster_binned_multi = spike_raster_binned
    print(spike_raster_binned_multi.shape)
else:
    spike_raster_binned_multi = np.dstack((spike_raster_binned_multi, spike_raster_binned))
    print(spike_raster_binned_multi.shape)

avg_corr = corr_coef(spike_raster_binned, binsize=binsize)

# create numpy array of spikes:
spike_array, spike_counts_binned, spike_raster_binned = make_spike_array(spike_monitor_trains=spike_counts, ntotal=Ntotal, rectime=runtime, rec_offset=0, dt=0.0001, binsize = 0.010)

# collect 10ms spike bins  (with spike counts per bin as well)
spike_counts_binned = np.empty([Ntotal, int(runtime/dt/10)])
spike_raster_binned = np.empty([Ntotal, int(runtime/dt/10)])
for set in range(int(runtime/dt/10)):
    spike_counts_binned[:,set] = np.sum(spike_array[:,set*10:set*10+10], axis=1)  # count the number of spikes per neuron in the 10ms bin
    spike_raster_binned[np.where(spike_counts_binned[:,set] > 0), set] = 1  # set a positive number of spikes per neuron to 1




#%% calculate correlation coefficients
corr_mtx = np.corrcoef(spike_raster_binned[:int(Ntotal*0.8),:])
x = corr_mtx[np.triu_indices(corr_mtx.shape[0], k=1)]
# not sure why but there are nan values coming up in the corr_values calculation
# remove nans from corr_values

corr_values = x[~np.isnan(x)]
avg_corr = np.mean(corr_values)
print(np.mean(corr_values))

plt.hist(corr_values, bins=100, color='gray')
plt.axvline(x=avg_corr, color='black')
plt.show()

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
plot_raster(spike_monitor=s_mon)


# plot histogram of `spike_counts` per cell
plt.hist(spike_counts_Hz, bins=100, color='gray')
plt.axvline(x=avg, color='black')
plt.show()





#%% plotting E and I inputs
def plot_inputs(e_monitor, i_monitor, neurons_to_plot, alpha, xlimits=None):
    fig, ax = plt.subplots(figsize=[20,3])
    ax.plot([e_monitor.t/ms]*len(neurons_to_plot), e_monitor[neurons_to_plot].ge, alpha=alpha, color='green')
    ax2 = ax.twinx()
    ax2.plot([i_monitor.t/ms]*len(neurons_to_plot), i_monitor[neurons_to_plot].gi, alpha=alpha, color='red')
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('e_monitor, current (siemens)', color='green')
    ax2.set_ylabel('i_monitor, current (siemens)', color='red')
    plt.show()
plot_inputs(e_monitor=trace_ge, i_monitor=trace_gi, neurons_to_plot=100, alpha=0.5)

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

