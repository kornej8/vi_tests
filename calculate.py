from loglike import LogLike
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor
from pycalphad import equilibrium
import pycalphad.variables as v
import cloudpickle


def main_calculations(params, dfs, tdb, sub_params):
    def get_mu_of_single_phase(x, t, params):
        conditions = {v.X(sub_params['element']): (x), v.T: (t), v.P: 101325, v.N: 1}

        res = equilibrium(tdb, tdb.elements,
                          [sub_params['research_phase']],
                          conditions=conditions,
                          to_xarray=False,
                          parameters=params)

        return np.squeeze(res.MU)[0]

    def get_mu_of_some_phases(x, t, params):
        conditions = {v.X(sub_params['element']): (0.5), v.T: (t), v.P: 101325, v.N: 1}

        res = equilibrium(tdb,
                          tdb.elements,
                          [sub_params['research_phase'], sub_params['boundary_phase']],
                          conditions=conditions,
                          to_xarray=False,
                          parameters=params)

        return np.squeeze(res.MU)[0]

    params = {function: params[num] for num, function in enumerate(sub_params['functions_to_alg'].keys())}

    df_conc, df_T = dfs[0], dfs[1]

    res = sum([(get_mu_of_single_phase(conc, df_T[i], params) -
                get_mu_of_some_phases(conc, df_T[i], params)) ** 2 for i, conc in
               enumerate(df_conc)])

    return np.array(res)


def main(data_path, samples_num, db, sub_params):
    df = pd.read_excel(data_path)
    data = [df['cr_conc'].values, df['T'].values]
    pytensor.config.exception_verbosity = 'high'

    logl = LogLike(main_calculations, data, db, sub_params)

    observed_data = data[0]

    with pm.Model() as model:
        tensor_variable = [pm.Normal(function, mu=mu, sigma=1) for function, mu in
                           sub_params['functions_to_alg'].items()]
        theta = pt.as_tensor_variable(tensor_variable)

        cr_observed = pm.ConstantData(name='cr_observed', value=observed_data)
        log = pm.Normal('y', logl(theta), sigma=1, observed=cr_observed)

    with model:
        advi = pm.ADVI()

        tracker = pm.callbacks.Tracker(
            mean=advi.approx.mean.eval,  # Для подсчета среднего
            std=advi.approx.std.eval,  # Для подсчета стандартного отклонения
        )

        mean_field = advi.fit(samples_num, callbacks=[tracker])

        # Для удобства и дальнейшего анализа принято решение сохранять результат расчета на диск
        # Дальнейший анализ проводить посредством сохраненных расчетов
        dict_to_save = {'model': model,
                        'tracker': tracker,
                        'advi': advi,
                        'mean_field': mean_field}

        with open(f'./models/{sub_params["tdb_name"]}_model_{samples_num}.pkl', 'wb') as buff:
            cloudpickle.dump(dict_to_save, buff)

        print(f'Расчет модели для файла {sub_params["tdb_name"]} успешно завершен. \n')
        print(f'Для доступа к модели обратитесь по пути: ./models/{sub_params["tdb_name"]}_model_{samples_num}.pkl')