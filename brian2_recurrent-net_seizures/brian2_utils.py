import sys; sys.path.append('/Users/prajayshah/OneDrive - University of Toronto/PycharmProjects/utils_pj')

import funcs_pj as pj
from brian2 import *


def plot_voltage(voltage_monitor, spike_monitor, alpha, xlimits, neuron_id=[]):
    # make random color choices
    colors = []
    for i in range(len(neuron_id)):
        colors.append(pj.generate_new_color(colors, pastel_factor=0.2))

    # make plot
    plt.figure(figsize=[20, 3])
    for neuron in neuron_id:
        plt.plot(voltage_monitor.t / ms, voltage_monitor[neuron].V / mV, alpha=alpha,
                 color=colors[neuron_id.index(neuron)])
        # add vertical lines at the spike times for individual neurons (using the values from the spike monitor)
        plt.vlines(x=np.int_(array(np.round(spike_monitor.spike_trains()[neuron], 3)) * 1000), ymin=-50, ymax=0,
                   alpha=alpha, color=colors[neuron_id.index(neuron)])
    plt.xlim(xlimits)
    plt.xlabel('t (ms)')
    plt.ylabel('v (mV)')
    plt.show()

def plot_raster(spike_monitor, neurons_to_plot):
    plt.figure(figsize=[20,3])
    plt.plot(spike_monitor.t/ms, spike_monitor.i, ',k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.ylim(neurons_to_plot[0],neurons_to_plot[1])
    plt.show()


