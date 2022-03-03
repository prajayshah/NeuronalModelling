from funcsforprajay.funcs import plot_hist_density

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

import numpy as np
import random

# %%
# create W matrix
inh_conn = 0.04
exc_conn = 0.04
E_prob = 0.80  # probability of excitatory cells (rest will be inhibitory)
Ntotal = 1000*5

# creating custom synaptic connectivity matrix for recurrent layer -- not quite a full synaptic weights matrix
Nn = Ntotal  # number of neurons in the recurrent layer
W = np.zeros([Nn, Nn])  # matrix of recurrent connection weights, should be a Nn x Nn size array
# W = np.diag(np.ones(Nn)) # -- self connections


###### spatially ordered connectivity
p_e = exc_conn  # probability of E->All connections
p_i = inh_conn  # probability of I->All connections neurons

Ne = random.sample(range(Nn), int(E_prob * Nn))
Ni = [x for x in range(Nn) if x not in Ne]

# need to intersperse E and I neurons throughout the whole cell population
for i in range(Nn):
    # select random indexes to setup as connections based on selected probability
    # set E --> all connectivity
    if i in Ne:
        j = [int(target) for target in np.random.normal(i, 200, int(p_e * len(W[i]))) if 0 < target < Nn]
        W[i, j] = 1

    # set I --> all connectivity
    elif i in Ni:
        j = [int(target) for target in np.random.normal(i, 200, int(p_i * len(W[i]))) if 0 < target < Nn]
        W[i, j] = 1



# %% FIRST BRIAN NETWORK - regular E/I (80/20 split)
def run_brian_network_and_analyses_0(Ne, Ni, W):

    #### set some network parameters
    runtime = 10 * second
    dt = 0.1 * ms
    neurons_to_stim = Ne[20:50]
    stim_external = False
    w_matrix = W

    #### build and run simulation
    spike_raster_binned_multi = None


    for input_rate in [0.5]:
        for tau_di in [8]:
            print('\n')
            print('\n')
            print('\n')
            print('\n')
            # build network
            record_id = [Ne[0], Ne[100], Ni[23], Ni[65]]
            net, W, trace, s_mon, trace_ge, s_mon_p, Ce, Ci, Ge, Gi, G, trace_z, trace_gi, trace_gi_diff = \
                b2_network(Ntotal=Ntotal, record_id=record_id, runtime=runtime, inh_conn=inh_conn, input_rate=input_rate,
                           stim_external=stim_external, neurons_to_stim=neurons_to_stim, Ne=Ne, Ni=Ni, w_matrix=w_matrix, tau_di=tau_di)

            if stim_external:
                print('--- STIMULATION ACTIVE: %s neurons stimulated ----' % len(neurons_to_stim))

            # quick spike raster plot to initialize plotting
            plt.figure(figsize=[20, 3])
            plt.plot(s_mon.t / ms, s_mon.i, ',k')
            plt.suptitle('Input rate (Hz): %s' % (input_rate))
            #     plt.suptitle(title)
            plt.show()
            spike_counts = s_mon.count
            spike_counts_Hz = np.array(spike_counts / runtime)
            avg = mean(spike_counts_Hz);
            print('average spiking rate of population: ', np.round(avg, 2), 'Hz')

            # make spike raster bins
            spike_monitor_trains = [(array / dt).astype(int64) for array in list(s_mon.spike_trains().values())]

            spike_monitor_trains_e = [spike_monitor_trains[i] for i in range(len(spike_monitor_trains)) if i in Ne]

            # create numpy array of spikes:
            binsize = 0.100  # sec
            spike_array, spike_counts_binned, spike_raster_binned = make_spike_array(
                spike_monitor_trains=spike_monitor_trains_e, ntotal=len(Ne), dt=dt, binsize=binsize, rectime=runtime)
            plot_firing_rate(spike_raster_binned, binsize_sec=0.01, title='Population Firing rate')

            ## plot histogram of firing rates for all neurons in network
            plot_firing_rate_histogram(spike_array = spike_array, rec_len_sec = runtime / second)

            if spike_raster_binned_multi is None:
                spike_raster_binned_multi = spike_raster_binned
                print(spike_raster_binned_multi.shape)
            else:
                spike_raster_binned_multi = np.dstack((spike_raster_binned_multi, spike_raster_binned))
                print(spike_raster_binned_multi.shape)

            avg_corr = corr_coef(spike_raster_binned, binsize=binsize)


# %% BRIAN NETWORK with mixed Inh timescales - regular E/I (80/20 split), Inh pop split with mixed timescales for rise and decay of synaptic current

def run_brian_network_and_analyses(Ne, Ni, W):
    #### set some network parameters
    runtime = 3 * second
    dt = 0.1 * ms
    neurons_to_stim = Ne[20:50]
    stim_external = False
    w_matrix = None

    #### build and run simulation
    spike_raster_binned_multi = None


    for input_rate in range(1, 10, 5):

        # build network
        record_id = [Ne[0], Ne[100], Ni[23], Ni[65]]
        net, W, trace, s_mon, trace_ge, s_mon_p, Ce, Ci_fast, Ci_slow, Ne, Ni, G = \
            b2_network_inhmix(Ntotal=Ntotal, record_id=record_id, runtime=runtime, inh_conn=0.2, exc_conn=0.2, input_rate=input_rate,
                              stim_external=stim_external, neurons_to_stim=neurons_to_stim, Ne=Ne, Ni=Ni, w_matrix=w_matrix,
                              tau_di_f=2, tau_di_s=5, ri_f=0.5, ri_s=1.5)

        if stim_external: print('--- STIMULATION ACTIVE: %s neurons stimulated ----' % len(neurons_to_stim))

        # quick spike raster plot to initialize plotting
        fig, ax = plt.subplots(figsize=[12,12])
        ax1 = plt.subplot2grid((4, 3), (0,0), rowspan=2, colspan=3)
        ax1.plot(s_mon.t / ms, s_mon.i, ',k')
        ax1.set_title('Input rate (Hz): %s' % (input_rate))
        # fig.suptitle('Input rate (Hz): %s' % (input_rate))
        # fig.show()
        spike_counts = s_mon.count
        spike_counts_Hz = np.array(spike_counts / runtime)
        avg = mean(spike_counts_Hz)
        print('average spiking rate of population: ', np.round(avg, 2), 'Hz')

        # make spike raster bins
        spike_monitor_trains = [(array / dt).astype(int64) for array in list(s_mon.spike_trains().values())]

        spike_monitor_trains_e = [spike_monitor_trains[i] for i in range(len(spike_monitor_trains)) if i in Ne]

        # create numpy array of spikes:
        binsize = 0.100  # sec
        spike_array, spike_counts_binned, spike_raster_binned = make_spike_array(
            spike_monitor_trains=spike_monitor_trains_e, ntotal=len(Ne), dt=dt, binsize=binsize, rectime=runtime)
        ax2 = plt.subplot2grid((4,3), (2,0), rowspan=1, colspan=3)
        plot_firing_rate(spike_raster_binned, binsize_sec=0.01, title='Population Firing rate', ax=ax2)

        ## plot histogram of firing rates for all neurons in network
        ax3 = plt.subplot2grid((4,3), (3,0), rowspan=1, colspan=1)
        plot_firing_rate_histogram(spike_array=spike_array, rec_len_sec=runtime / second, ax=ax3, fig=fig)

        if spike_raster_binned_multi is None:
            spike_raster_binned_multi = spike_raster_binned
            print(spike_raster_binned_multi.shape)
        else:
            spike_raster_binned_multi = np.dstack((spike_raster_binned_multi, spike_raster_binned))
            print(spike_raster_binned_multi.shape)

        ax4 = plt.subplot2grid((4,3), (3,1), rowspan=1, colspan=1)
        avg_corr = corr_coef(spike_raster_binned, binsize=binsize, ax=ax4, fig=fig)

        ax5 = plt.subplot2grid((4,3), (3,2), rowspan=1, colspan=1)
        plot_connectivity_matrix(W, color_lim=[0.0, 0.51], fig=fig, ax=ax5)


        fig.tight_layout()
        fig.show()

    # return avg_corr

# %%
if __name__ == '__main__':

    run_brian_network_and_analyses(Ne=Ne, Ni=Ni, W=W)
    # run_brian_network_and_analyses_0(Ne=Ne, Ni=Ni, W=W)


