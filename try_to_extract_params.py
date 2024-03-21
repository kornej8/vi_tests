import sys
import pandas as pd
import numpy as np
from vi_project import TDBPoints, TDB_LIST
from pycalphad import Database
from pycalphad.core.utils import extract_parameters, instantiate_models,unpack_components
from pycalphad.model import Model
from symengine import exp, log, Abs, Add, And, Float, Mul, Piecewise, Pow, S, sin, StrictGreaterThan, Symbol, zoo, oo


ELEM = 'CR'
tdb = 'CoCr-01Oik.tdb'

"""
Запуск скрипта в терминале: python get_exp.py sigma_fcc_allibert.xls 

PS Пока что тестил только на sigma_fcc_allibert 
PSS Пока работает только с 'CoCr-01Oik.tdb' потому что во второй базе название фаз отличается =)

"""

tdb_object = f"./test_data/{tdb}"

# df = pd.read_excel(path)
tdb = Database(tdb_object)
# print(tdb.symbols.items())

print(tdb.symbols)

symbols = {Symbol(s): val for s, val in tdb.symbols.items()}

print(extract_parameters(tdb.symbols)) #не работает

print(extract_parameters(symbols)) #не работает


# print(Model(Database(tdb_object), ['CO', 'CR', 'VA'], "fcc_a1"))
