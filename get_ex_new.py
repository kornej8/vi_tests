import sys
import pandas as pd
import numpy as np
from vi_project import TDBPoints, TDB_LIST
from pycalphad import Database
from pycalphad.core.utils import extract_parameters, instantiate_models, unpack_components
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

tdb = Database(tdb_object)
#
# condition = {v.X('CR'): (0.01, 1, 0.01),
#                           v.T: 1400,
#                           v.P: 101325}
#
# res = equilibrium(tdb, tdb.elements, list(tdb.phases.keys()), conditions=condition )
# # print(res)

"""
Запуск скрипта в терминале: python get_exp.py sigma_fcc_allibert.xls 

PS Пока что тестил только на sigma_fcc_allibert 
PSS Пока работает только с 'CoCr-01Oik.tdb' потому что во второй базе название фаз отличается =)

"""


def get_x_based_on_experement(params, df, tdb_object):
    return_square_of_diff = lambda first_var, second_var: (first_var - second_var) ** 2

    if params is None:
        tdb_object.model_params = None
    else:
        tdb_object.model_params = {'L0_0_FCC': params[0], 'L0_1_FCC': params[1]}

    df[f'conc_from_{tdb}'] = df['T'].astype(str) + ';' + df['phase']
    df[f'conc_from_{tdb}'] = df[f'conc_from_{tdb}'].apply(lambda x: tdb_object.get_params(
        t=float(x.split(';')[0]),
        checked_phase=x.split(';')[1]).get_max_concentration()[0])
    df['diffs_values'] = df.apply(lambda table: return_square_of_diff(table.cr_conc, table[f'conc_from_{tdb}']), axis=1)
    print(df)
    return np.array(df['diffs_values'].sum())


def get_max_concentration_2(x, *args):
    tdb = args[0]
    temp = args[1]
    params_dict = args[2]

    def get_maximum_np(x, *argss):
        cond = {v.X('CR'): (x[0]), v.T: (argss[0]), v.P: 101325, v.N: 1}
        res = equilibrium(tdb, list(tdb.elements), list(tdb.phases.keys()), conditions=cond, parameters=params_dict)
        ans = 1 - np.squeeze(res.NP.values)[0]
        if ans <= 0.001:
            return 100
        return ans

    x0 = np.array([x])
    res = minimize(get_maximum_np, x0, method='nelder-mead', args=[temp],
                   options={'maxiter': 20})
    return res.x[0]


def return_eq(tdb_object_2, x, t):
    tdb = tdb_object_2
    cond = {v.X('CR'): (x), v.T: (t), v.P: 101325, v.N: 1}
    res = equilibrium(tdb, list(tdb.elements), list(tdb.phases.keys()), conditions=cond)
    return np.squeeze(res.NP.values)[0]


def debug():
    tdb_object = TDBPoints(tdb_object_path, element=ELEM)
    df = pd.read_excel(path)
    df = df.loc[df.phase == 'fcc_a1']
    tdb_path = f"./test_data/CoCr-01Oik.tdb"
    element = 'CR'
    tdb_object = TDBPoints(tdb_path, element=element, t=1375)
    eps = np.sqrt(np.finfo(float).eps)
    tdb_object_2 = Database(tdb_object_path)
    # tdb_object.model_params = {'L0_0_FCC': -23030, 'L0_1_FCC': 7.9}
    params = {'L0_0_FCC': 8.34 + 0.0001}
    df = pd.read_excel(path)
    df = df.loc[df.phase == 'fcc_a1']

    # df[f'conc_from_tdb_2'] = df['T'].astype(str) + ';' + df['phase']
    # df[f'conc_from_tdb_2'] = df[f'conc_from_tdb_2'].apply(lambda x: tdb_object.get_params(
    #     t=float(x.split(';')[0]),
    #     checked_phase=x.split(';')[1]).get_max_concentration()[0])

    df[f'conc_with_CoCr-01Oik.tdb'] = df.apply(lambda table: get_max_concentration_2(table['cr_conc'], tdb_object_2, table['T'], params), axis=1)

    # df['eq_res1'] = df.apply(lambda table: return_eq(tdb_object_2, table['conc_from_CoCr-01Oik.tdb'], table['T']), axis=1)
    # df['eq_res2'] = df.apply(lambda table: return_eq(tdb_object_2, table['conc_with_optimzie'], table['T']), axis = 1)

    return df[df.columns[2:]]


def main():
    pytensor.config.exception_verbosity = 'high'
    tdb_object = TDBPoints(tdb_object_path, element=ELEM)
    df = pd.read_excel(path)
    df = df.loc[df.phase == 'fcc_a1']
    logl = LogLike(get_x_based_on_experement, df, tdb_object)
    # observed_data = df.cr_conc
    #
    # with pm.Model() as model:
    #     vL0_0 = pm.Normal("vL0_0", mu=-23080, sigma=1)
    #     vL0_1 = pm.Normal("vL0_1", mu=8.34, sigma=1)
    #
    #     theta = pt.as_tensor_variable([vL0_0, vL0_1])
    #
    #     cr_observed = pm.ConstantData(name='cr_observed', value=observed_data)
    #     log = pm.Normal('y', logl(theta),  sigma=1, observed=cr_observed)
    #
    # with model:
    #     advi = pm.ADVI()
    #
    #     tracker = pm.callbacks.Tracker(
    #         mean=advi.approx.mean.eval,  # callable that returns mean
    #         std=advi.approx.std.eval,  # callable that returns std
    #     )
    #
    #     mean_field = pm.fit(n=6000, method=advi, callbacks=[tracker])
    #
    # return mean_field, tracker, advi


if __name__ == '__main__':
    print(debug())
    # result, tracker_res, advi_res = main()
