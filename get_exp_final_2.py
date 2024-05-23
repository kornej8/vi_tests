from setting_app import ConfigurateApp
import pandas as pd
from black_box import LogLike
import cloudpickle
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pycalphad import Database, equilibrium
import pycalphad.variables as v


def sub_main(params, dfs, tdb):
    lst_of_x = [0.5]

    def get_mu_of_point(x, t, params):
        conditions = {v.X('CR'): (x), v.T: (t), v.P: 101325, v.N: 1}
        res = equilibrium(tdb, tdb.elements, ['FCC_A1'], conditions=conditions, to_xarray=False, parameters=params)
        return np.squeeze(res.MU)[0]

    def get_mu_of_point_3(x, t, params):
        conditions = {v.X('CR'): lst_of_x, v.T: (t), v.P: 101325, v.N: 1}
        res = equilibrium(tdb, tdb.elements, ['FCC_A1', 'SIGMA_OLD'], conditions=conditions, to_xarray=False,
                          parameters=params)
        return np.squeeze(res.MU)[0]

    params = {'L0_0_FCC': params[0], 'L0_1_FCC': params[1]}

    df_conc = df[0]
    df_T = df[1]
    res = sum(
        [(get_mu_of_point(conc, df_T[i], params) - get_mu_of_point_3(conc, df_T[i], params)) ** 2 for i, conc in
         enumerate(df_conc)])

    return np.array(res)


def main(samples_num, path, dbf, ):
    df = pd.read_excel(path)
    df = df.loc[df.phase == 'fcc_a1']

    df = [df['cr_conc'].values, df['T'].values]
    pytensor.config.exception_verbosity = 'high'
    logl = LogLike(sub_main, df, dbf)
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

        mean_field = advi.fit(samples_num, callbacks=[tracker])

        dict_to_save = {'model': model,
                        'tracker': tracker,
                        'advi': advi,
                        'mean_field': mean_field}

        with open('D:/Files/test_my_01model7000.pkl', 'wb') as buff:
            cloudpickle.dump(dict_to_save, buff)



if __name__ == '__main__':
    samples_num = 14000
    data_path = './test_data/sigma_bcc_allibert.xls'
    main(samples_num, data_path)
