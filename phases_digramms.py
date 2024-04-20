from pycalphad import Database, equilibrium
import pycalphad.variables as v
import numpy as np
import matplotlib.pyplot as plt
from pycalphad import Database, equilibrium, calculate, binplot
import pycalphad.variables as v
from pycalphad.plot.utils import phase_legend
import pandas as pd
import sympy

# ELEM = 'CR'
# tdb = 'CoCr-01Oik.tdb'
# tdb_object = f"./test_data/{tdb}"
# # file_name = sys.argv[-1]
# file_name = 'sigma_fcc_allibert.xls'
# path = f'./test_data/{file_name}'
# tdb_object_path = f"./test_data/{tdb}"
#
# # dict_of_elems, dict_of_phases = {} , {}
# #
# # df = pd.read_excel(path)
#
#
# tdb = Database(tdb_object)

# dict_of_elems[tdb] = list(tdb.elements)
# dict_of_phases[tdb] = list(tdb.phases.keys())
#
#
#
#
# from scipy.optimize import minimize
#
# def make_condition2(x, *args):
#     cond = {v.X('CR'): (x[0]), v.T: (args[0]), v.P: 101325, v.N: 1}
#     res = equilibrium(tdb, list(tdb.elements), list(tdb.phases.keys()), conditions=cond)
#     ans = 1-np.squeeze(res.NP.values)[0]
#     if ans <= 0.001:
#         return 100
#     return ans
#
# x0 = np.array([0.382368193604148])
# res = minimize(make_condition2, x0, options={'maxiter' : 5}, method='nelder-mead', args = [1321.34831460674]),
#
# print(res.x[0])
#
#
# # 0.3806 0.3691
# def make_condition3(x):
#     cond = {v.X('CR'): (x[0]), v.T: (1321.34831460674), v.P: 101325, v.N: 1}
#     res = equilibrium(tdb, list(tdb.elements), ['FCC_A1', 'SIGMA_OLD'], conditions=cond)
#     return np.squeeze(res.NP.values)
#
# print(make_condition3([0.3806505239844419])[0])



dict_of_elems, dict_of_phases = {} , {}
dict_of_elems2, dict_of_phases2 = {} , {}

ELEM = 'CR'
tdb = 'CoCr-01Oik.tdb'
tdb2 = 'CoCr-18Cac.tdb'
tdb_object = f"./test_data/{tdb}"
tdb_object2 = f"./test_data/{tdb2}"
# file_name = sys.argv[-1]
file_name = 'sigma_fcc_allibert.xls'
path = f'./test_data/{file_name}'
tdb_object_path = f"./test_data/{tdb}"
df = pd.read_excel(path)


dict_of_elems[tdb] = list(Database(tdb_object).elements)
# dict_of_phases[tdb] = list(map(lambda x: f"{x}_18", list(Database(tdb_object).phases.keys())))
dict_of_phases[tdb] = list(Database(tdb_object).phases.keys())
dict_of_elems2[tdb2] = list(Database(tdb_object2).elements)
# dict_of_phases2[tdb2] = list(map(lambda x: f"{x}_18", list(Database(tdb_object2).phases.keys())))
dict_of_phases2[tdb2] = list(Database(tdb_object2).phases.keys())



def make_condition(el, T):
    return {v.X(el):(0.3, 0.6, 0.01), v.T: (1150, T, 2), v.P:101325, v.N: 1}

cond = make_condition('CR', 1550)
tdb_path = Database(tdb_object)
tdb_path2 = Database(tdb_object2)
fig = plt.figure(figsize=(13,8))
axes = fig.gca()

binplot(tdb_path, dict_of_elems[tdb] , dict_of_phases[tdb],  cond, plot_kwargs={'ax': axes, 'tielines': False})
binplot(tdb_path2, dict_of_elems2[tdb2], dict_of_phases2[tdb2], cond, plot_kwargs={'ax': axes, 'tielines': False})

for phase in df.phase.unique().tolist():
    df_f = df.loc[df.phase == phase]
    plt.scatter(df_f['cr_conc'], df_f['T'], label=phase)


axes.set_xlim(0.3, 0.6)
plt.legend()
plt.show()

