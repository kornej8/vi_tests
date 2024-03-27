import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import seaborn as sns
import matplotlib.pyplot as plt
import pytensor
import pytensor.tensor as pt
from pycalphad import Database, equilibrium

# define a pytensor Op for our likelihood function
class LogLike(pt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, func, df, tdb_object, params = None):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.data = df
        self.tdb = tdb_object
        self.params = params
        self.func = func

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        self.params = {'L0_FCC' : inputs[0][0]}

        # call the log-likelihood function
        logl = self.func(self.data, self.tdb_object, self.params)

        outputs[0][0] = logl  # output the log-likelihood



