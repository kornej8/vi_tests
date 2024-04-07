import sys
import pandas as pd
import numpy as np
from vi_project import TDBPoints, TDB_LIST
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


ELEM = 'CR'
tdb = 'CoCr-01Oik.tdb'
tdb_object = f"./test_data/{tdb}"
# file_name = sys.argv[-1]
file_name = 'sigma_fcc_allibert.xls'
path = f'./test_data/{file_name}'
tdb_object_path = f"./test_data/{tdb}"


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
        tdb_object.model_params = {'L0_FCC': params}

    df[f'conc_from_{tdb}'] = df['T'].astype(str) + ';' + df['phase']
    df[f'conc_from_{tdb}'] = df[f'conc_from_{tdb}'].apply(lambda x: tdb_object.get_params(
        t=float(x.split(';')[0]),
        checked_phase=x.split(';')[1]).get_max_concentration()[0])
    df['diffs_values'] = df.apply(lambda table: return_square_of_diff(table.cr_conc, table[f'conc_from_{tdb}']), axis=1)
    return np.array(df['diffs_values'].sum())



def main():
    pytensor.config.exception_verbosity = 'high'
    tdb_object = TDBPoints(tdb_object_path, element=ELEM)
    df = pd.read_excel(path)
    logl = LogLike(get_x_based_on_experement, df, tdb_object)
    observed_data = df.cr_conc
    print(df)
    exit()
    with pm.Model() as model:
        vL0_FCC = pm.Normal("vL0_FCC", mu=-12000, sigma=1)

        theta = pt.as_tensor_variable([vL0_FCC])

        cr_observed = pm.ConstantData(name='cr_observed', value=observed_data)
        log = pm.Normal('y', logl(theta),  sigma=1, observed=cr_observed)

    # with model:
    #     print('tut')
    #     mean_field = pm.fit(method = 'advi', obj_optimizer=pm.adagrad_window(learning_rate=1e-2))

        # advi = pm.ADVI()

        # tracker = pm.callbacks.Tracker(
        #     mean=advi.approx.mean.eval,  # callable that returns mean
        #     std=advi.approx.std.eval,  # callable that returns std
        # )
        #
        # approx = advi.fit(20000, callbacks=[tracker])
        #
        # fig = plt.figure(figsize=(16, 9))
        # mu_ax = fig.add_subplot(221)
        # std_ax = fig.add_subplot(222)
        # hist_ax = fig.add_subplot(212)
        # mu_ax.plot(tracker["mean"])
        # mu_ax.set_title("Mean track")
        # std_ax.plot(tracker["std"])
        # std_ax.set_title("Std track")
        # hist_ax.plot(advi.hist)
        # hist_ax.set_title("Negative ELBO track");

        # Use custom number of draws to replace the HMC based defaults
        # # use a Potential to "call" the Op and include it in the logp computation
        # pm.Potential("likelihood", logl(theta))
        #
        # likelihood = pm.Normal('y', mu=logl(theta), sigma=0.001, observed=cr_observed)
        #
        with model:
            mean_field = pm.fit(method="advi", callbacks=[CheckParametersConvergence()])
    #
    plt.plot(mean_field.hist);
if __name__ == '__main__':
    main()