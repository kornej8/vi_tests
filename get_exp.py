import pandas as pd
import numpy as np
from pycalphad import Database
from pycalphad.core.utils import extract_parameters, instantiate_models,unpack_components
from pycalphad.model import Model
from black_box import LogLike
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import seaborn as sns
import matplotlib.pyplot as plt
import pytensor
import pytensor.tensor as pt
from pymc.variational.callbacks import CheckParametersConvergence
from pycalphad import Database, equilibrium
import pycalphad.variables as v
from scipy.optimize import minimize


ELEM = 'CR'
tdb = 'CoCr-01Oik.tdb'
tdb_object = f"./test_data/{tdb}"
# file_name = sys.argv[-1]
file_name = 'sigma_fcc_allibert.xls'
path = f'./test_data/{file_name}'
tdb_object_path = f"./test_data/{tdb}"
# dbf = Database(tdb_object_path)
# dbf_elem = list(dbf.elements)
# dbf_phase = list(dbf.phases.keys())

def get_max_concentration_2(x ,*args):
    temp = args[1]
    params = args[2]
    if params is None:
        params_dict = None
    else:
        params_dict = {'L0_0_FCC': params[0], 'L0_1_FCC': params[1]}

    def get_maximum_np(x, *argss):
        cond = {v.X('CR'): (x[0]), v.T: (argss[0]), v.P: 101325, v.N: 1}
        res = equilibrium(dbf, dbf.elements, list(dbf.phases.keys()), conditions=cond, parameters=params_dict)
        ans = 1-np.squeeze(res.NP.values)[0]
        if ans <= 0.001:
            return 100
        return ans

    x0 = np.array([x])
    res = minimize(get_maximum_np, x0, method='nelder-mead', args = [temp],
        options={'maxiter' : 5})
    return res.x[0]

def get_x_based_on_experement_2(params, df, dbf):
    df_conc = df[0]
    df_T = df[1]

    res = [(get_max_concentration_2(conc, dbf, df_T[i], params) - conc )**2 for i, conc in enumerate(df_conc)]

    return np.array(sum(res))

def main2(df, dbf):
    pytensor.config.exception_verbosity = 'high'
    logl = LogLike(get_x_based_on_experement_2, df, dbf)
    observed_data = df[0]

    with pm.Model() as model:
        vL0_0 = pm.Normal("vL0_0", mu=-23080, sigma=1)
        vL0_1 = pm.Normal("vL0_1", mu=8.34, sigma=1)

        theta = pt.as_tensor_variable([vL0_0, vL0_1])

        cr_observed = pm.ConstantData(name='cr_observed', value=observed_data)
        log = pm.Normal('y', logl(theta),  sigma=1, observed=cr_observed)

    with model:
        advi = pm.ADVI()

        tracker = pm.callbacks.Tracker(
            mean=advi.approx.mean.eval,  # callable that returns mean
            std=advi.approx.std.eval,  # callable that returns std
        )

        mean_field = pm.fit(n=6000, method=advi, callbacks=[tracker])

    return mean_field, tracker, advi

if __name__ == '__main__':
    df = pd.read_excel(path)
    df = df.loc[df.phase == 'fcc_a1']
    dbf = Database(tdb_object_path)
    df = [df['cr_conc'].values, df['T'].values]

    result, tracker_res, advi_res = main2(df, dbf)
