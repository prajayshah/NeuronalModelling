import pyabf
import numpy as np
# import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


import matplotlib as mpl
# import matplotlib.pyplot as plt
mpl.use('macosx')  # or can use 'TkAgg', whatever you have/prefer
import plotly.graph_objects as go

import pandas as pd

# ----------------------------------------------------------------------------------------------------------------------
# Load up the abf file into python
# ----------------------------------------------------------------------------------------------------------------------

#%% open a selection dialog to retrieve EEG file names to analyse
root = tk.Tk()
root.withdraw()

# filez = filedialog.askopenfilenames(initialdir='/Volumes/Extreme SSD/Exp 2020.1T_KainicAcidOptogenetics/247EEG', parent=root, title='Choose files')
filez = filedialog.askopenfilenames(initialdir='/Users/prajayshah/OneDrive - University of Toronto/UTPhD/2020/Exp 2020.1T/2020-10-19_OptoEEG', parent=root, title='Choose files')
fpath = list(root.tk.splitlist(filez))
print(fpath)

#


# file_path = '/Users/prajayshah/OneDrive - University of Toronto/UTPhD/2018-2019/Epilepsy model project/EEG recordings/2019_09_13_0002.abf'
# fpath = file_path

# Load up abf file with pyABF

def load_abf(fpath=fpath):
    print('Loading %s' % fpath)

    a = pyabf.ABF(fpath)
    fs = a.dataRate # sampling frequency
    V = a.data
    t = np.arange(0, len(V[0])) * (1.0 / fs) # length of the data, converted to seconds by dividing by the sampling frequency
    print('ABF File Comment: ', a.abfFileComment)
    print('Sampling rate: ', fs)
    print('length of recording (seconds): ', t[-1])
    print('number of datapoints: ', len(V[0]))

    return a, fs, t, V

if len(fpath)==1:
    a, fs, t, V = load_abf(fpath[0])





#%%
# Create figure

# set layout
layout = go.Layout(
    title="EEG - Voltage series - %s - %s" % (a.abfFileComment, a.abfFilePath[-12:]), # set title
    plot_bgcolor="#FFF",  # Sets background color to white
    hovermode='x',
    hoverdistance=10,
    spikedistance=1000,
    xaxis=dict(
        title="time",
        linecolor="#BCCCDC",  # Sets color of X-axis line
        showgrid=False,  # Removes X-axis grid lines
        # rangeslider=list(),

    # format spikes
        showspikes=True,
        spikethickness=2,
        spikedash = 'dot',
        spikecolor="#999999",
        spikemode='across'
    ),
    yaxis=dict(
        title="price",
        linecolor="#BCCCDC",  # Sets color of Y-axis line
        showgrid=False,  # Removes Y-axis grid lines
        fixedrange = False,
        rangemode='normal'
    )
)

fig = go.Figure(data= go.Scatter(x=list(t[::10]), y=list(V[0][::10]), line=dict(width=0.95)),  # downsampling data by 10,
                layout=layout)

# fig.update_traces(hovertemplate=None)

# fig.add_trace(
#     go.Scatter(x=list(t[::10]), y=list(V[0][::10]), line=dict(width=0.75)))  # downsampling data by 10



# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type="linear"
    )
)

fig.show()

# go.FigureWidget(fig)