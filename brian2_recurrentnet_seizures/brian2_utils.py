import sys; sys.path.append('/Users/prajayshah/OneDrive - University of Toronto/PycharmProjects/utils_praj')

from funcs_pj import generate_new_color
from brian2 import *


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

def make_spike_array(spike_monitor, ntotal, runtime, dt, binsize=10):
    spike_array = np.empty([ntotal, int(runtime / dt)])

    for neuron in range(ntotal):
        if neuron % 100 == 0:  # print the progress once every 100 cell iterations
            print(neuron, " out of ", ntotal, " cells done", end='\r')
        spike_locs = (list(spike_monitor.spike_trains()[neuron])/dt).astype(int64)
        spike_array[neuron, spike_locs] = 1

    print('done collecting spikes for all cells')

    # collect 10ms spike bins  (with spike counts per bin as well)
    spike_counts_binned = np.empty([ntotal, int(runtime / dt / binsize)])
    spike_raster_binned = np.empty([ntotal, int(runtime / dt / binsize)])
    for set in range(int(runtime / dt / binsize)):
        spike_counts_binned[:, set] = np.sum(spike_array[:, set * binsize:set * binsize + binsize],
                                             axis=1)  # count the number of spikes per neuron in the 10ms bin
        spike_raster_binned[
            np.where(spike_counts_binned[:, set] > 0), set] = 1  # set a positive number of spikes per neuron to 1

    return spike_array, spike_raster_binned, spike_counts_binned


def plot_voltage(voltage_monitor, spike_monitor, alpha, ylimits, xlimits, neuron_id=[],
                 title='Voltage trace - Neuron: '):
    # make random color choices
    colors = []
    for i in range(len(neuron_id)):
        colors.append(generate_new_color(colors, pastel_factor=0.2))

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


def plot_firing_rate(spike_raster_binned, binsize_s=0.01, title='Population Firing rate'):
    """calculate and plot firing rate across 10ms timebins"""

    firing_rate_binned = np.sum(spike_raster_binned, axis=0)
    firing_rate_binned_norm = firing_rate_binned / binsize_s
    plt.figure(figsize=[20, 3])
    plt.plot(firing_rate_binned_norm, c='black', linewidth=1)
    plt.suptitle(title)
    plt.ylabel('Firing rate value (Hz)')
    plt.show()


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
    plot(trace_z.t / ms, trace_z[neuron[0]].z, color='black')
    if xlimits:
        xlim(xlimits)
    #         ylim([0.0, 1.0])
    xlabel('t (ms)')
    ylabel('z')
    show()


def plot_inputs(e_monitor, i_monitor, neurons_to_plot, alpha, xlimits=None):
    fig, ax = plt.subplots(figsize=[20, 3])
    ax.plot([e_monitor.t / ms] * len(neurons_to_plot), e_monitor[neurons_to_plot].ge, alpha=alpha, color='salmon')
    ax2 = ax.twinx()
    ax2.plot([i_monitor.t / ms] * len(neurons_to_plot), i_monitor[neurons_to_plot].gi, alpha=alpha, color='deepskyblue')
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