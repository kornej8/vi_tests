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

print('start')
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

def get_x_based_on_experement(df, tdb_object, params = None):
    print('123')
    tdb_object.model_params = params
    df[f'conc_from_{tdb}'] = df['T'].astype(str) + ';' + df['phase']
    df[f'conc_from_{tdb}'] = df[f'conc_from_{tdb}'].apply(lambda x: tdb_object.get_params(
        t=float(x.split(';')[0]),
        checked_phase=x.split(';')[1]).get_max_concentration()[0])
    return np.array(df['conc_from_CoCr-01Oik.tdb'].astype(np.float32).values)


def main():
    tdb_object = TDBPoints(tdb_object_path, element=ELEM)
    df = pd.read_excel(path)
    print(df)
    logl = LogLike(get_x_based_on_experement, df, tdb_object)

    with pm.Model() as model:
        L0_FCC = pm.Normal("L0_FCC", mu=-12000, sigma=1)

        theta = pt.as_tensor_variable([L0_FCC])

        # use a Potential to "call" the Op and include it in the logp computation
        pm.Potential("likelihood", logl(theta))


    with model:
        mean_field = pm.fit(method="advi")

    az.plot_posterior(mean_field.sample(1000), color="LightSeaGreen")
    plt.show()

if __name__ == '__main__':
    main()