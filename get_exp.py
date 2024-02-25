import sys
import pandas as pd
import numpy as np
from vi_project import TDBPoints, TDB_LIST

ELEM = 'CR'
tdb = 'CoCr-01Oik.tdb'

"""
Запуск скрипта в терминале: python get_exp.py sigma_fcc_allibert.xls 

PS Пока что тестил только на sigma_fcc_allibert 
PSS Пока работает только с 'CoCr-01Oik.tdb' потому что во второй базе название фаз отличается =)

"""

file_name = sys.argv[-1]
path = f'./test_data/{file_name}'
df = pd.read_excel(path)



df[f'conc_from_{tdb}'] = df['T'].astype(str) + ';' + df['phase']
tdb_object = f"./test_data/{tdb}"
tdb_object = TDBPoints(tdb_object, element=ELEM)

df[f'conc_from_{tdb}'] = df[f'conc_from_{tdb}'].apply(lambda x: tdb_object.get_params(
    t=float(x.split(';')[0]),
    checked_phase=x.split(';')[1]).get_max_concentration()[0])


print(df)