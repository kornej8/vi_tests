import pandas as pd
import numpy as np
from pycalphad import Database
from pycalphad.core.utils import extract_parameters, instantiate_models, unpack_components
from pycalphad.model import Model
from black_box import LogLike
import arviz as az
import pickle
import cloudpickle
import numpy as np
import pymc as pm
import pytensor
import seaborn as sns
from numba import jit, cuda
import pytensor
import pytensor.tensor as pt
from pymc.variational.callbacks import CheckParametersConvergence
from pycalphad import Database, equilibrium, calculate
import pycalphad.variables as v
from scipy.optimize import minimize, approx_fprime
from memory_profiler import profile
from numba import jit, cuda


ELEM = 'CR'
tdb = 'CoCr-18Cac.tdb'
tdb_object = f"./test_data/{tdb}"
# file_name = sys.argv[-1]
file_name = 'sigma_fcc_allibert.xls'
path = f'./test_data/{file_name}'
tdb_object_path = f"./test_data/{tdb}"
tdb_path = f"./test_data/CoCr-01Oik.tdb"
element = 'CR'
dbf = Database(tdb_object)
conditions = {v.X('CR'): (0.5), v.T: (1321.34831460674), v.P: 101325, v.N: 1}
print(equilibrium(dbf, dbf.elements, ['FCC_A1', 'SIGMA_D8B'], conditions=conditions, to_xarray=False).Phase)
lst_of_x = [0.5]


def check_cond(lst):
    return lst[0] == 'FCC_A1' and lst[1] == 'SIGMA_OLD'


def tt_func(params, x, t):
    remove_phase = True
    conditions = {v.X('CR'): (0, 1, 0.05), v.T: (t), v.P: 101325, v.N: 1}
    res = equilibrium(dbf, dbf.elements, list(dbf.phases.keys()), conditions=conditions, to_xarray=False,
                      parameters=params)
    r = [[list(phase), max(np.squeeze(res.MU)[i])] for i, phase in enumerate(np.squeeze(res.Phase))]
    for i in r:
        if check_cond(i[0]):
            return i[1]

    res = equilibrium(dbf, dbf.elements, ['FCC_A1', 'SIGMA_OLD'], conditions=conditions, to_xarray=False,
                      parameters=params)
    r = [[list(phase), max(np.squeeze(res.MU)[i])] for i, phase in enumerate(np.squeeze(res.Phase))]
    for i in r:
        if check_cond(i[0]):
            return i[1]

    conditions = {v.X('CR'): (0.5), v.T: (t), v.P: 101325, v.N: 1}
    return max(
        np.squeeze(equilibrium(dbf, dbf.elements, ['FCC_A1', 'SIGMA_OLD'], conditions=conditions, to_xarray=False,
                               parameters=params).MU))


def mainsss(params, dfs, tdb):

    def get_mu_of_point(x, t, params):
        conditions = {v.X('CR'): (x), v.T: (t), v.P: 101325, v.N: 1}
        res = equilibrium(tdb, tdb.elements, ['FCC_A1'], conditions=conditions, to_xarray=False, parameters=params)
        return max(np.squeeze(res.MU))

    def get_mu_of_point_3(x, t, params):
        conditions = {v.X('CR'): lst_of_x, v.T: (t), v.P: 101325, v.N: 1}
        res = equilibrium(tdb, tdb.elements, list(tdb.phases.keys()), conditions=conditions, to_xarray=False,
                          parameters=params)
        if check_cond(np.squeeze(res.Phase)):
            return max(np.squeeze(res.MU))
        else:
            return tt_func(params, x, t)

    params = {'L0_0_FCC': params[0], 'L0_1_FCC': params[1]}

    df_conc = df[0]
    df_T = df[1]
    res = sum(
        [(get_mu_of_point(conc, df_T[i], params) - get_mu_of_point_3(conc, df_T[i], params)) ** 2 for i, conc in
         enumerate(df_conc)])

    return np.array(res)

    # df[f'mu_with_tdb'] = df.apply(lambda table: get_mu_of_point(table['cr_conc'], table['T'], params), axis=1)
    # # df[f'mu_with_tdb_2'] = df.apply(lambda table: get_mu_of_point_2(table['cr_conc'], table['T'], params), axis=1)
    # df[f'mu_with_tdb_3'] = df.apply(lambda table: get_mu_of_point_3(table['cr_conc'], table['T'], params), axis=1)
    #
    # df['dif'] = df.apply(lambda t: (t['mu_with_tdb'] - t['mu_with_tdb_3'])**2, axis = 1)
    # return df.dif.sum()




def main2(df):
    pytensor.config.exception_verbosity = 'high'
    logl = LogLike(mainsss, df, dbf)
    observed_data = df[0]

    with pm.Model() as model:
        vL0_0 = pm.Normal("vL0_0", mu=-23080, sigma=1)
        vL0_1 = pm.Normal("vL0_1", mu=8.34, sigma=1)

        theta = pt.as_tensor_variable([vL0_0, vL0_1])

        cr_observed = pm.ConstantData(name='cr_observed', value=observed_data)
        log = pm.Normal('y', logl(theta), sigma=1, observed=cr_observed)

    with model:
        advi = pm.ADVI()

        tracker = pm.callbacks.Tracker(
            mean=advi.approx.mean.eval,  # callable that returns mean
            std=advi.approx.std.eval,  # callable that returns std
        )

        mean_field = pm.fit(n=8000, method=advi, callbacks=[tracker])

        dict_to_save = {'model': model,
                        'tracker': tracker,
                        'advi': advi,
                        'mean_field': mean_field
                        }

        with open('my_model.pkl', 'wb') as buff:
            cloudpickle.dump(dict_to_save, buff)

    return mean_field, tracker, advi


# pickle_filepath = f'path/to/pickle.pkl'
# with open(pickle_filepath , 'rb') as buff:
#     model_dict = cloudpickle.load(buff)
#
# idata = model_dict['idata']
# model = model_dict['model']
#
# with model:
#     ppc_logit = pm.sample_posterior_predictive(idata )


if __name__ == '__main__':
    df = pd.read_excel(path)
    df = df.loc[df.phase == 'fcc_a1']

    df = [df['cr_conc'].values, df['T'].values]

    # result, tracker_res, advi_res = main2(df)
