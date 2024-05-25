import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy import optimize


class LogLike(pt.Op):
    """
    В нашем случае мы будем передавать ему вектор значений (параметры
    термодинамической модели) и возвращать одно "скалярное" значение (
    логарифмическую вероятность).
    """

    itypes = [pt.dvector]  # Ожидает вектор значений параметров при вызове
    otypes = [pt.dscalar]  # Выводит одно скалярное значение (логарифмическую вероятность).

    def __init__(self, func, df, tdb_object, sub_params):
        """
        Расчет функции правдоподобия через интерфейс PyTensor.Op

        """

        self.data = df
        self.tdb = tdb_object
        self.func = func
        self.sub_params = sub_params
        self.logpgrad = LogLikeGrad(self.func, self.data, self.tdb, self.sub_params)

    def perform(self, node, inputs, outputs):
        (theta,) = inputs  # Вектор входящих параметров термодинамической модели

        p_l0_0 = inputs[0][0]
        p_l0_1 = inputs[0][1]

        # РАсчет правдоподобия
        logl = self.func([p_l0_0, p_l0_1], self.data, self.tdb, self.sub_params)
        outputs[0][0] = logl  # Вывод правдоподобия

    def grad(self, inputs, g):
        # Метод, вызывающий расчет градиентов
        (theta,) = inputs  # Вектор входящих параметров
        return [g[0] * self.logpgrad(theta)]


class LogLikeGrad(pt.Op):
    """
    Этот класс будет вызван с вектором значений и также вернет вектор
    значений.
    """

    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, func, data, tdb, sub_params):
        """
        Инициализация с помощью различных параметров, которые требуются
        для функции main_calculations.

        """

        # add inputs as class attributes
        self.func = func
        self.data = data
        self.tdb = tdb
        self.sub_params = sub_params

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        # Рассчет градиентов
        eps = np.sqrt(np.finfo(float).eps)
        grads = optimize.approx_fprime(theta, self.func, [eps, eps], self.data, self.tdb, self.sub_params)
        outputs[0][0] = grads
