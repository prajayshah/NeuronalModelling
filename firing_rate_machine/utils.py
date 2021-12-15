import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from functools import partial
import pickle

### Functions for building CONNECTIVITY MATRIX

def GetBulk (N):
    chi = np.random.normal( 0, np.sqrt(1./(N)), (N,N) )
    return chi

def GetGaussianVector (mean, std, N):
    if std>0:
        return np.random.normal (mean, std, N )
    else:
        return mean*np.ones(N)


### Functions for INTEGRATING

def Integrate (X, t, J, I):
    dXdT = -X + np.dot( J, np.tanh(X) ) + I
    return dXdT

def SimulateActivity (t, x0, J, I):
    # print(' ** Simulating... **')
    return scipy.integrate.odeint( partial(Integrate, J=J, I=I), x0, t )

# plotting
def pop_firing_rate(F, plot=True):
    F_avg = np.mean(F, axis=1)
    F_avg_total = np.mean(F_avg)

    if plot:
        f, ax = plt.subplots(figsize = [4,2])
        ax.axhline(y=F_avg_total, color='gray', alpha=0.2)
        ax.plot(F_avg)
        ax.set_xlabel('Time (norm.)')
        ax.set_ylabel('Avg. Pop firing rate')
        f.tight_layout(pad=1.3)
        f.show()

    return F_avg_total

def plot_network_simulation(data, J, title=None):
    ## plot connectivity matrix
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=[7.5, 5])
    im = axs[0].imshow(J, cmap='Purples', vmin=np.mean(J) - np.std(J), vmax=np.mean(J) + np.std(J))
    axs[0].set_title('Sources (left) x Targets (bottom) connectivity matrix', wrap=True)

    # plotting firing rate as heatmap of Neurons x Time
    im = axs[1].imshow(data.T, cmap='RdBu_r', aspect=27.5, vmin=-10, vmax=10)
    f.colorbar(im, orientation='vertical', fraction=0.05)
    axs[1].set_title('Firing Rate: Neurons x Time')
    # ax.set_colorbar(fraction=0.008)
    f.suptitle(title) if title else None
    f.tight_layout()
    f.show()


def plot_heatmap(data, xlabel: str, ylabel: str, colorlabel: str, xlabels: list = None, ylabels: list = None,
                 vmin=None, vmax=None, invert_y=True, minor_ticks=False, title='high quality heatmap',
                 **kwargs):

    if 'fig' in kwargs.keys():
        fig = kwargs['fig']
        ax = kwargs['ax']
    else:
        if 'figsize' in kwargs.keys():
            fig, ax = plt.subplots(figsize=kwargs['figsize'])
        else:
            fig, ax = plt.subplots(figsize=[data.shape[0] * 2 / 3, data.shape[1] * 2 / 3])

    if vmin is None:
        vmin_ = np.min(data)
    else:
        vmin_ = vmin
    if vmax is None:
        vmax_ = np.max(data)
    else:
        vmax_ = vmax
    hmap = ax.imshow(data, cmap='Reds', vmin=vmin_, vmax=vmax_)
    color_bar = fig.colorbar(hmap, ax=ax, fraction=0.046, pad=0.04)
    if minor_ticks:
        color_bar.minorticks_on()
    color_bar.set_label(colorlabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlabels:
        ax.set_xticks(range(data.shape[1])[::2])
        ax.set_xticklabels(xlabels[::2])
    if ylabels:
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels(ylabels)
    if invert_y:
        ax.invert_yaxis()


    # show or return axes objects
    if 'show' in kwargs.keys():
        fig.show() if kwargs['show'] else None
    else:
        fig.show()

    if 'fig' in kwargs.keys():
        return fig, ax


def load_(pkl_path: str):
    f = pickle.load(open(pkl_path, 'rb'))
    return f