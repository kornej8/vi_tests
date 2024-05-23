import pandas as pd
import numpy as np
from pycalphad import Database
from pycalphad.core.utils import extract_parameters, instantiate_models, unpack_components
from pycalphad.model import Model
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
from black_box import LogLike

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


def mainsss(params, dfs, tdb):
    def get_mu_of_point(x, t, params):
        conditions = {v.X('CR'): (x), v.T: (t), v.P: 101325, v.N: 1}
        res = equilibrium(tdb, tdb.elements, ['FCC_A1'], conditions=conditions, to_xarray=False, parameters=params)
        return np.squeeze(res.MU)[0]

    def get_mu_of_point_3(x, t, params):
        conditions = {v.X('CR'): (0.5), v.T: (t), v.P: 101325, v.N: 1}
        res = equilibrium(tdb, tdb.elements, ['FCC_A1', 'SIGMA_D8B'], conditions=conditions, to_xarray=False,
                          parameters=params)
        return np.squeeze(res.MU)[0]

    params = {'L0_0_FCC': params[0], 'L0_1_FCC': params[1]}

    df_conc = df[0]
    df_T = df[1]
    res = sum([(get_mu_of_point(conc, df_T[i], params) - get_mu_of_point_3(conc, df_T[i], params)) ** 2 for i, conc in
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
        vL0_0 = pm.Normal("vL0_0", mu=-24052.09, sigma=1)
        vL0_1 = pm.Normal("vL0_1", mu=8.1884, sigma=1)

        theta = pt.as_tensor_variable([vL0_0, vL0_1])

        cr_observed = pm.ConstantData(name='cr_observed', value=observed_data)
        log = pm.Normal('y', logl(theta), sigma=1, observed=cr_observed)

    return model


if __name__ == '__main__':
    df = pd.read_excel(path)
    df = df.loc[df.phase == 'fcc_a1']

    df = [df['cr_conc'].values, df['T'].values]

    model = main2(df)
    with model:
        mean_field = pm.sample_smc(1000, chains=1, progressbar=True)
