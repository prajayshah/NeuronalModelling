import numpy as np


def gen_param(mu, sigma, m, flag=0):
    if sigma == 0:
        output = mu * np.ones(m)
    elif sigma > 0:
        if flag == 0:
            output = mu + np.multiply(sigma, np.random.normal(size=m))
        if flag == 1:
            ouput = None ##

class Projection:
    def __init__(self):
        None
    ##

class ExternalInput:
    None
    ##

class Recorder:
    None
    ##


class NeuralNetwork(Projection):
    # set up a SUPER-CLASS called 'NeuralNetwork'
    # - this super-class will contain the information that is shared by all sub-types of networks

    def __init__(self):
        # define basic properties of the neural network
        self.n = None # num of cells in network
        self.t = 0 # current time

        # association with instances of other classes
        self.Proj = {'In': None,
                     'Out': None}

    def update(self,o,dt):
        if dt <= 0:
            pass
        else:
            for i in range(len(o)):
                if o.__class__ == MeanFieldModel:
                    MeanFieldModel.IndividualModelUpdate(o[i],dt)
            for i in range(len(o)):
                for j in range(o.Proj['Out']):
                    





class MeanFieldModel(NeuralNetwork):
    def __init__(self):
        super().__init__()  # figure out the use of this call
        self.param = None ##
        self.V = None ##
        self.phi = None ##
        self.Cl_in = None ##
        self.g_K = None ##

        self.VarName = ['V', 'phi', 'Cl_in', 'g_K']

    def IndividualModelUpdate(self, o, dt):

