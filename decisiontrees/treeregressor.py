import numpy as np
from abstract.base_tree import DecisionTree


class DecisionTreeRegressor(DecisionTree):
    # Inicializácia regresného rozhodovacieho stromu
    def __init__(self, max_depth=None, min_samples_split=2, loss='mse'):
        # Volanie konštruktora nadradenej triedy
        super().__init__(max_depth, min_samples_split)
        self.loss = loss  # Funkcia straty: 'mse' alebo 'mae'

    # Súkromná metóda na výpočet strednej kvadratickej chyby (MSE)
    def __mse(self, data):
        # Priemerná hodnota cieľovej premennej
        mean_target = np.mean(data[:, -1])
        mse_value = np.sum((data[:, -1] - mean_target)
                           ** 2) / data.shape[0]  # Výpočet MSE
        return mse_value

    # Súkromná metóda na výpočet strednej absolútnej chyby (MAE)
    def __mae(self, data):
        # Priemerná hodnota cieľovej premennej
        mean_target = np.mean(data[:, -1])
        mae_value = np.sum(
            np.abs(data[:, -1] - mean_target)) / data.shape[0]  # Výpočet MAE
        return mae_value

    # Chránená metóda na výpočet nečistoty uzla
    def _impurity(self, data):
        impurity_value = None
        if self.loss == 'mse':
            impurity_value = self.__mse(data)  # Výpočet nečistoty pomocou MSE
        elif self.loss == 'mae':
            impurity_value = self.__mae(data)  # Výpočet nečistoty pomocou MAE
        return impurity_value

    # Chránená metóda na výpočet hodnoty listového uzla
    def _leaf_value(self, data):
        # Návrat priemernej hodnoty cieľovej premennej
        return np.mean(data[:, -1])

    # Verejná metóda na získanie parametrov modelu
    def get_params(self, deep=False):
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'loss': self.loss
        }
