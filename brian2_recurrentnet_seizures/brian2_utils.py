import sys;

import numpy as np
from matplotlib import pyplot as plt

sys.path.append('/home/pshah/Documents/code/utils_praj')

from funcsforprajay.funcs import _generate_new_color, plot_hist_density, dataplot_frame_options
from brian2 import *
from sklearn.decomposition import PCA
import pandas as pd
from scipy import stats


######## UTILITIES FOR ANLAYSIS OF DATA ################################################################################
# not sure that this actually works with very sparsely firing networks where some neurons have no firing
def make_spike_array(spike_monitor_trains: np.array, ntotal, rectime, rec_offset=0, dt=0.0001, binsize = 0.010):
    """
    make a binned spike array from an array of neurons x spike_times.

    TODO consider doing a sliding window for spike counting. Huang et al., do 200ms sliding window at 1ms timesteps.

    Parameters
    ----------
    spike_monitor_trains: an array or list of neurons x spike_times
    ntotal: # of total neurons (should be same as len of spike_monitor_trains)
    rectime: total recording time of simulation (seconds)
    rec_offset: how to offset the spike locations (if they are relative to the total simulation time and NOT the total recording time)
    dt: time resolution of simulation (seconds)
    binsize: size of bin to use for making spike array (sec)

    Returns
    -------

    """
    assert ntotal == len(spike_monitor_trains)
    spike_array = np.zeros([ntotal, int(rectime / dt)])

    for neuron in range(ntotal):
        #         if neuron % 100 == 0:  # print the progress once every 100 cell iterations
        #             print(neuron, " out of ", ntotal, " cells done")
        spike_locs = (spike_monitor_trains[neuron] - rec_offset)
        if type(spike_locs) is not list:
            spike_locs = list(spike_locs)
        spike_locs = [int(x) for x in spike_locs]
        spike_array[neuron, spike_locs] = 1

    print('collecting binned spikes rasters... %s ms bin size' % (binsize * 10 ** 3))

    # collect spike bins  (with spike counts per bin as well)
    spike_counts_binned = np.zeros([ntotal, int(rectime / dt / (binsize / dt))])
    spike_raster_binned = np.zeros([ntotal, int(rectime / dt / (binsize / dt))])
    for set in range(int(rectime / dt / (binsize / dt))):
        spike_counts_binned[:, set] = np.sum(
            spike_array[:, int(set * binsize / dt): int(set * binsize / dt + binsize / dt)],
            axis=1)  # count the number of spikes per neuron in the 10ms bin
        spike_raster_binned[
            np.where(spike_counts_binned[:, set] > 0), set] = 1  # set a positive number of spikes per neuron to 1

    return spike_array, spike_raster_binned, spike_counts_binned

# calculate correlation coefficients
def corr_coef(spike_raster_binned, binsize, **kwargs):
    corr_mtx = np.corrcoef(spike_raster_binned)
    x = corr_mtx[np.triu_indices(corr_mtx.shape[0], k=1)]
    # not sure why but there are nan values coming up in the corr_values calculation
    # remove nans from corr_values

    corr_values = x[~np.isnan(x)]
    avg_corr = np.mean(corr_values)
    print('Avg. corr coef value: ', avg_corr)

    dataplot_frame_options()
    if not 'ax' in [*kwargs]:
        fig, ax = plt.subplots(figsize=[3, 3])
    else:
        ax = kwargs['ax']
        fig = kwargs['fig']

    ax.hist(corr_values, bins=100, color='gray')
    ax.axvline(x=avg_corr, color='black')
    ax.set_xlabel('correlation coefficient')
    ax.set_ylabel('density')
    ax.set_title(f'avg corr coef distribution, binsize = {(binsize * 1000)} ms, avg = {round(avg_corr,6)}', wrap=True)
    fig.show() if not 'fig' in [*kwargs] else None

    return avg_corr


def plot_firing_rate_histogram(spike_array, rec_len_sec, **kwargs):
    firing_rates = []
    for neuron in spike_array:
        n_spikes = sum(neuron)
        firing_rates.append(n_spikes / rec_len_sec)

    dataplot_frame_options()
    if not 'ax' in [*kwargs]:
        fig, ax = plt.subplots(figsize=[3, 3])
    else:
        fig = kwargs['fig']
        ax = kwargs['ax']


    # make histogram
    plot_hist_density(data=[firing_rates], fill_color=['limegreen'], alpha=0.0, num_bins=20, best_fit_line='powerlaw',
                      colors=['cornflowerblue'], x_label='firing rate (Hz)', fig=fig, ax=ax, show=False, mean_line=True)


# PCA on dataset
def pca_spike_raster(spike_raster_binned):
    pca = PCA(n_components=min(spike_raster_binned.shape))
    pca.fit(spike_raster_binned)
    pca_result = pd.DataFrame(pca.transform(spike_raster_binned))  # , columns=['PCA%i' % i for i in range(275)])
    sv = pca.singular_values_
    su = (sv / sum(sv))
    return pca.explained_variance_ratio_


# fit and plot variance per PC
def powerlawfit(data: np.array, subset: list = None):
    x = range(len(data))[subset[0]:subset[1]]
    y = data[subset[0]:subset[1]]


    # to perform powerlaw fit, first take data into log-log space and then do linear regression in that space
    res = stats.linregress(np.log2(x), np.log2(y))
    print(res)

    x_fit = range(len(data))[int(0.2*len(data)): -int(0.2*len(data))]
    y_fit = res.intercept + res.slope * np.log2(x_fit)

    fig, ax = plt.subplots(figsize=[6,6])
    ax.plot(np.log2(range(len(data))), np.log2(data), label='original data', color='darkblue')
    ax.plot(np.log2(x_fit), y_fit, c = 'darkgreen', label='fitted line')
    ax.set_xlabel('PC Dimension')
    ax.set_ylabel('Variance (|eigenvalue|)')
    plt.show()

    return res.slope

# #%%
# import matplotlib.pyplot as plt
# import numpy as np
# x = np.linspace(1, 10 ** 2, 10 ** 2)
# y = 1 / (x ** 1.1)
# plt.scatter(x, y); plt.show()
#
# def func(x, a, b):
#     return x**a + b
#
# fit = np.polyfit(np.log(x), np.log(y), 1)
# plt.scatter(x, y)
# plt.plot(x, func(x, *fit), color='orange'); plt.show()





#%%


######## PLOTTING FUNCTIONS ############################################################################################
def plot_raster(spike_monitor, title='Raster plot', neurons_to_plot=None, xlimits=None, color='black'):
    """
    Plot a raster plot using the spike monitor object.

    :param spike_monitor: spike monitor object recorded from a brian2 simulation run
    :param neurons_to_plot: range of neurons to plot for spike raster (y axis limits)
    :param xlimits: limit of x axis (time, in milliseconds)
    """
    figure(figsize=[20,3])
    plot(spike_monitor.t/ms, spike_monitor.i, ',k', color=color)
    xlabel('Time (ms)')
    ylabel('Neuron index')
    if xlimits:
        xlim(xlimits)
    if neurons_to_plot:
        ylim(neurons_to_plot[0],neurons_to_plot[1])
    if title:
        suptitle(title)
    show()


def plot_voltage(voltage_monitor, spike_monitor, alpha, ylimits, xlimits, neuron_id=[],
                 title='Voltage trace - Neuron: '):
    # make random color choices
    colors = []
    for i in range(len(neuron_id)):
        colors.append(_generate_new_color(colors, pastel_factor=0.2))

    # make plot
    plt.figure(figsize=[20, 3])
    for neuron in neuron_id:
        plt.plot(voltage_monitor.t / ms, voltage_monitor[neuron].V / mV, alpha=alpha,
                 color=colors[neuron_id.index(neuron)])
        # add vertical lines at the spike times for individual neurons (using the values from the spike monitor)
        plt.vlines(x=np.int_(array(np.round(spike_monitor.spike_trains()[neuron], 3)) * 1000), ymin=-50, ymax=0,
                   alpha=alpha, color=colors[neuron_id.index(neuron)])
        plt.suptitle(title + str(neuron))
    if xlimits:
        plt.xlim(xlimits)
    if ylimits:
        plt.ylim(ylimits)
    plt.xlabel('t (ms)')
    plt.ylabel('v (mV)')
    plt.show()


def plot_firing_rate(spike_raster_binned, binsize_sec=0.01, title= 'Neuronal Population Firing rate', **kwargs):
    """calculate and plot firing rate across 10ms timebins"""

    firing_rate_binned = np.sum(spike_raster_binned, axis=0)
    firing_rate_binned_norm = firing_rate_binned / binsize_sec / spike_raster_binned.shape[0]

    dataplot_frame_options()
    if not 'ax' in [*kwargs]:
        fig, ax = plt.subplots(figsize=[20, 3])
    else:
        ax = kwargs['ax']
    ax.plot(firing_rate_binned_norm, c='black', linewidth=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Population Firing rate (Hz)')
    # title = 'avg firing rate %s' % (np.sum(firing_rate_binned_norm[params['rec_start']/1000/binsize_sec:params['rec_stop']/1000/binsize_sec])/len(firing_rate_binned_norm[params['rec_start']/1000/binsize_sec:params['rec_stop']/1000/binsize_sec]))
    ax.set_title(title + ', binsize = %s ms' % (binsize_sec * 1000))
    fig.show() if not 'ax' in [*kwargs] else None


def make_plots_inh_exhaust_mech(s_mon, s_mon_p, trace, trace_z, trace_gi_diff, trace_gi, trace_ge, neuron,
                                xlimits=False):
    "bunch of plots for looking at the Inh. exhaust mech"

    # plt.style.use('dark_background')

    # figure(figsize=[20, 3])
    # plot(s_mon.t / ms, s_mon.i, ',k', color='black')
    # xlabel('t (ms)')
    # ylabel('Neuron index')
    # show()
    # spike_counts = s_mon.count
    # spike_counts_Hz = array(spike_counts / runtime)
    # avg = mean(spike_counts_Hz);
    # print('average spiking rate of population: ', avg, 'Hz')

    figure(figsize=[20, 3])
    plot(s_mon.t / ms, s_mon.i, ',k', color='black')
    if xlimits:
        xlim(xlimits)
    xlabel('t (ms)')
    ylabel('Neuron index - main group')
    show()

    figure(figsize=[20, 3])
    plot(s_mon_p.t / ms, s_mon_p.i, ',k', color='black')
    if xlimits:
        xlim(xlimits)
    xlabel('t (ms)')
    ylabel('Neuron index - Poisson input group')
    show()

    for i in neuron:
        plot_voltage(voltage_monitor=trace, spike_monitor=s_mon,
                     neuron_id=[i], alpha=0.7, ylimits=[-95, 20], xlimits=xlimits)

    plt.figure(figsize=[20, 3])
    plot(trace_ge.t / ms, trace_ge[neuron[0]].ge / nS, color='salmon', linewidth=0.5)
    suptitle('ge - Neuron %s' % neuron[0])
    if xlimits:
        xlim(xlimits)
    xlabel('t (ms)')
    ylabel('ge')
    show()

    plt.figure(figsize=[20, 3])
    plot(trace_gi.t / ms, trace_gi[neuron[0]].gi / nS, color='deepskyblue', linewidth=0.5)
    suptitle('gi - Neuron %s' % neuron[0])
    if xlimits:
        xlim(xlimits)
        ylim([0, 500])
    xlabel('t (ms)')
    ylabel('gi')
    show()

    plt.figure(figsize=[20, 3])
    plot(trace_gi_diff.t / ms, trace_gi_diff[neuron[0]].gi_diff, color='black', linewidth=0.5)
    suptitle('gi_diff - Neuron %s' % neuron[0])
    if xlimits:
        xlim(xlimits)
        ylim([-700, 700])
    xlabel('t (ms)')
    ylabel('gi_diff')
    show()

    plt.figure(figsize=[20, 3])
    plot(trace_z.t / ms, trace_z[neuron[0]].z, color='purple')
    suptitle('z variable - Neuron %s' % neuron[0])
    if xlimits:
        xlim(xlimits)
    #         ylim([0.0, 1.0])
    xlabel('t (ms)')
    ylabel('z')
    show()


def plot_inputs(e_monitor, i_monitor, neurons_to_plot, alpha, xlimits=None):
    fig, ax = plt.subplots(figsize=[20, 3])
    ax.plot([i_monitor.t / ms] * len(neurons_to_plot), i_monitor[neurons_to_plot].gi, alpha=alpha, color='deepskyblue')
    ax2 = ax.twinx()
    ax2.plot([e_monitor.t / ms] * len(neurons_to_plot), e_monitor[neurons_to_plot].ge, alpha=alpha, color='salmon')
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('e_monitor, current (siemens)', color='salmon')
    ax2.set_ylabel('i_monitor, current (siemens)', color='deepskyblue')
    plt.suptitle('Synaptic inputs across neurons #: %s' % neurons_to_plot)
    plt.show()


def plot_e_inputs(e_monitor, neurons_to_plot, alpha):
    fig, ax = plt.subplots(figsize=[40, 3])
    ax.plot([e_monitor.t / ms] * len(neurons_to_plot), e_monitor[neurons_to_plot].ge, alpha=alpha, color='salmon')
    ax.plot(e_monitor.t / ms, np.mean(e_monitor[neurons_to_plot].ge, axis=0), linewidth=0.5,
            color='black')  # plot average of
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('e_monitor, current (siemens)', color='green')
    plt.suptitle('Exc. synaptic inputs across neurons #: %s' % neurons_to_plot)
    plt.show()


def plot_i_inputs(i_monitor, neurons_to_plot, alpha):
    fig, ax = plt.subplots(figsize=[40, 3])
    ax.plot([i_monitor.t / ms] * len(neurons_to_plot), i_monitor[neurons_to_plot].gi, alpha=alpha, color='deepskyblue')
    ax.plot(i_monitor.t / ms, np.mean(i_monitor[neurons_to_plot].gi, axis=0), linewidth=0.5,
            color='black')  # plot average of
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('i_monitor, current (siemens)', color='red')
    plt.suptitle('Inh. synaptic inputs across neurons #: %s' % neurons_to_plot)
    plt.show()


def plot_connectivity_matrix(conn_matrix, cmap='Purples', color_lim=[0.05, 0.1], colorbar=False, **kwargs):
    """plot heatmap of synaptic connectivity matrix given as numpy array where 1 denotes connection between i and j index"""
    # plt.figure(figsize=[10, 10])

    if not 'ax' in [*kwargs]:
        fig, ax = plt.subplots(figsize=[5, 5])
    else:
        ax = kwargs['ax']
        fig = kwargs['fig']

    hmap = ax.imshow(conn_matrix, cmap=cmap, vmin=color_lim[0], vmax=color_lim[1])
#     plt.clim(color_lim[0], color_lim[1])
#     if title is not None:
#         ax.set_title(title)
    ax.set_title('Binary synaptic connectivity matrix (source neurons on left axis, target neurons on bottom axis)', wrap=True)
    if colorbar:
        color_bar = fig.colorbar(hmap, ax=ax)
        color_bar.minorticks_on()
    ax.set_xlabel('Neurons (source)')
    ax.set_ylabel('Neurons (target)')
    fig.show() if 'fig' not in [*kwargs] else None
