import sys; sys.path.append('/Users/prajayshah/OneDrive - University of Toronto/PycharmProjects/utils_praj')

from funcs_pj import generate_new_color
from brian2 import *


def plot_voltage(voltage_monitor, spike_monitor, alpha, ylimits, xlimits, neuron_id=[], title=None):
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
    if xlimits:
        plt.xlim(xlimits)
    if ylimits:
        plt.ylim(ylimits)
    if title:
        suptitle(title)
    plt.xlabel('t (ms)')
    plt.ylabel('v (mV)')
    plt.show()

def plot_raster(spike_monitor, title=None, neurons_to_plot=None, xlimits=None, color='black'):
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



