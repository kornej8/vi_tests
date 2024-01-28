from pycalphad import Database, equilibrium, calculate
import pycalphad.variables as v
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


model1 = 'F:/Python/vi_git/test_data/CoCr-01Oik.tdb'
model2 = 'F:/Python/vi_git/test_data/CoCr-18Cac.tdb'

tdb = Database(model1)
tdb2 = Database(model2)
print(tdb.elements)

cr_conc = 0.382368193604148 #fcc_a1
t = 1321.34831460674

cr_conc1 = 0.559283413215997
t1 = 1422.47191011235

phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'HCP_A3', 'SIGMA_OLD']

# {'CR', 'VA', 'CO'}

def make_conditions(temp, conc, N=1, P=101325):
    return {v.X('CR') : conc, v.T: (temp), v.P: P}

cond = make_conditions(t1, cr_conc1)

res = equilibrium(tdb, tdb.elements, phases, cond)
res2 = equilibrium(tdb2, tdb.elements, phases, cond)

# res2 = calculate(tdb, tdb.elements, phases, P=101325, T=t)

print(f'CoCr-01Oik.tdb: {res.Phase.values}')
print(f'CoCr-18Cac.tdb: {res2.Phase.values}')



#
# def res(el, phases, cond):
#     return equilibrium(tdb, el, phases, cond)
#
#


# gamma_data = np.random.gamma(2, 0.5, size=200)
# print(sns.histplot(gamma_data))


# with open(model1, mode = 'r') as f:
#     for line in f.readlines():
#         print(line)

# Гистограмма
# dist = plt.hist(vals, bins = 'auto')