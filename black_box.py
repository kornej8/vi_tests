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
from scipy import optimize

import numpy as np
from scipy import optimize


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

    def __init__(self, func, df, tdb_object):
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
        self.func = func
        self.logpgrad = LogLikeGrad(self.func, self.data, self.tdb)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        p_l0_0 = inputs[0][0]
        p_l0_1 = inputs[0][1]
        # params, df, tdb_object
        # call the log-likelihood function
        logl = self.func([p_l0_0,p_l0_1], self.data, self.tdb)
        outputs[0][0] = logl  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]

class LogLikeGrad(pt.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, func, data, tdb):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.func = func
        self.data = data
        self.tdb = tdb
    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        # calculate gradients
        eps = np.sqrt(np.finfo(float).eps)
        grads = optimize.approx_fprime(theta, self.func, [eps, eps], self.data, self.tdb)
        outputs[0][0] = grads

# [-2.30795729e+04,  7.64607207e+00]

# theta = [-2.30795729e+04,  7.64607207e+00]
# eps = np.sqrt(np.finfo(float).eps)



    # print((func([1,1 + eps], 1, 200) - func([1,1], 1, 200)) / eps)


# def normal_gradients(theta, x, data, sigma):
#     """
#     Calculate the partial derivatives of a function at a set of values. The
#     derivatives are calculated using the central difference, using an iterative
#     method to check that the values converge as step size decreases.
#
#     Parameters
#     ----------
#     theta: array_like
#         A set of values, that are passed to a function, at which to calculate
#         the gradient of that function
#     x, data, sigma:
#         Observed variables as we have been using so far
#
#
#     Returns
#     -------
#     grads: array_like
#         An array of gradients for each non-fixed value.
#     """
#
#     grads = np.empty(2)
#     aux_vect = data - my_model(theta, x)  # /(2*sigma**2)
#     grads[0] = np.sum(aux_vect * x)
#     grads[1] = np.sum(aux_vect)
#
#     return grads