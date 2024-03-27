from pycalphad import Database, equilibrium, calculate
from pycalphad.core.utils import extract_parameters, instantiate_models, unpack_components
import pycalphad.variables as v
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from decimal import Decimal
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')

TDB_LIST = ['CoCr-01Oik.tdb', 'CoCr-18Cac.tdb']


class TDBPoints:
    """
    Экземплярами класса являются точки представляемой в
    иницализаторе термодинамической базы данных
    """
    _TOLERANCE = Decimal(0.00001)  # Точность (Пока не используется)
    __P = 101325  # Давление (Па)

    def __init__(self, tdb_object, checked_phase=None, element=None):
        self.tdb_object = Database(tdb_object)
        self.element = element
        self.checked_phase = checked_phase
        self.all_phases = list(self.tdb_object.phases.keys())
        self.t = None
        self.condition = {v.X(self.element): (Decimal(0.01), Decimal(1), Decimal(0.001)),
                          v.T: (self.t),
                          v.P: 101325}
        self.vector = None
        self._params = None

    def get_params(self, t, checked_phase, element=None):
        """
        Вызов метода изменяет параметры, подставляемые в pycalphad.equilibrium
        (на текущий момент только температура)
        """
        if element is not None:
            self.element = element
        self.t = t
        self.checked_phase = checked_phase
        self.condition = {v.X(self.element): (Decimal(0.01), Decimal(1), Decimal(0.001)),
                          v.T: (self.t),
                          v.P: __class__.__P}

        return self

    @property
    def model_params(self):
        return self._params
    @model_params.setter
    def model_params(self, params: dict):
        self._params = params

    def get_max_concentration(self):
        """
        Метод возвращает кортеж с расситаными параметрами ТДБ,
        соответствующими приближению из эксперементальных данных
        ('Концентрация вещества (x)', 'Список из фаз', 'Значения растворимостей фаз NP')
        """
        result = self.do_mapping()
        return max(result, key=lambda x: x[2][x[1].index(self.checked_phase.upper())])

    @staticmethod
    def check_cond(vector, x, phase):
        """
        Проверка условия в do_mapping
        """
        return len(np.squeeze(vector.Phase.values)[x]) >= 2 and np.squeeze(vector.Phase.values)[x].tolist().count(
            '') <= 1 \
               and phase.upper() in np.squeeze(vector.Phase.values)[x]

    def do_equilibrium(self):
        """
        Метод запускает pycalphad.equilibrium и присваивает результат атрибуту vector
        """
        self.vector = equilibrium(self.tdb_object, self.tdb_object.elements, self.all_phases, self.condition,
                                  parameters=self.model_params)

    def do_mapping(self):
        """
        Мапимся по значениям из pycalphad.equilibrium и проверяем на условия:
        1. Количество фаз >= 2
        2. В фазах должна присутствовать интересущая нас фаза
        """
        self.do_equilibrium()
        res = self.vector
        return [(res.X_CR.values[x_cr],
                 list(np.squeeze(res.Phase.values)[x_cr]),
                 np.squeeze(res.NP.values)[x_cr].tolist()) for x_cr in range(len(res[f'X_{self.element}'].values))
                if self.check_cond(res, x_cr, self.checked_phase)]

