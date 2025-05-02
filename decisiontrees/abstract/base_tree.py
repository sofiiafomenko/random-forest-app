# Importy
from abc import ABC, abstractmethod
import numpy as np

# Trieda na reprezentáciu uzla stromu


class Node:
    def __init__(self):
        # Hodnota rozdelenia
        self._split_value = None
        # Index vlastnosti, podľa ktorej sa rozdeľuje
        self._feature_index = None
        # Ľavý podstrom
        self._left_child = None
        # Pravý podstrom
        self._right_child = None
        # Hodnota listu (ak je uzol list)
        self.leaf_value = None

    def set_params(self, split_value, feature_index):
        # Nastavenie parametrov uzla
        self._split_value = split_value
        self._feature_index = feature_index

    def get_params(self):
        # Získanie parametrov uzla
        return self._split_value, self._feature_index

    def set_children(self, left, right):
        # Nastavenie detí uzla
        self._left_child = left
        self._right_child = right

    def get_left(self):
        # Získanie ľavého podstromu
        return self._left_child

    def get_right(self):
        # Získanie pravého podstromu
        return self._right_child

# Abstraktná základná trieda pre rozhodovacie stromy


class DecisionTree(ABC):
    def __init__(self, max_depth=None, min_samples_split=2):
        # Koreň stromu
        self.root = None
        # Maximálna hĺbka stromu
        self.max_depth = max_depth
        # Minimálny počet vzoriek na rozdelenie
        self.min_samples_split = min_samples_split

    @abstractmethod
    def _impurity(self, data):
        # Abstraktná metóda na výpočet nečistoty
        pass

    @abstractmethod
    def _leaf_value(self, data):
        # Abstraktná metóda na výpočet hodnoty listu
        pass

    def _grow_tree(self, node, data, current_depth):
        # Podmienky na rast stromu
        can_grow = (self.max_depth is None or current_depth < self.max_depth) and \
                   (data.shape[0] >= self.min_samples_split) and \
                   (np.unique(data[:, -1]).shape[0] > 1)

        if can_grow:
            # Inicializácia najlepších parametrov
            best_impurity = None
            best_feature = None
            best_split = None
            best_left = None
            best_right = None

            # Výber kandidátnych vlastností
            candidate_features = np.random.choice(
                range(data.shape[1] - 1),
                size=int(np.sqrt(data.shape[1] - 1)),
                replace=False
            )

            for feature_index in candidate_features:
                for split_value in np.unique(data[:, feature_index]):
                    # Rozdelenie dát na základe hodnoty rozdelenia
                    left_data = data[data[:, feature_index] <= split_value]
                    right_data = data[data[:, feature_index] > split_value]

                    if left_data.size and right_data.size:
                        # Výpočet váženej nečistoty
                        weighted_impurity = (left_data.shape[0] / data.shape[0]) * self._impurity(left_data) + \
                                            (right_data.shape[0] / data.shape[0]
                                             ) * self._impurity(right_data)
                        if best_impurity is None or weighted_impurity < best_impurity:
                            # Aktualizácia najlepších parametrov
                            best_impurity = weighted_impurity
                            best_feature = feature_index
                            best_split = split_value
                            best_left = left_data
                            best_right = right_data

            if best_split is None or best_feature is None or best_left is None or best_right is None:
                # Ak nie je možné rozdeliť, nastaví sa hodnota listu
                node.leaf_value = self._leaf_value(data)
                return

            # Nastavenie parametrov uzla a detí
            node.set_params(best_split, best_feature)
            left_node = Node()
            right_node = Node()
            node.set_children(left_node, right_node)

            # Rekurzívny rast stromu
            self._grow_tree(node.get_left(), best_left, current_depth + 1)
            self._grow_tree(node.get_right(), best_right, current_depth + 1)
        else:
            # Ak nie je možné rozdeliť, nastaví sa hodnota listu
            node.leaf_value = self._leaf_value(data)
            return

    def _traverse_tree(self, node, sample):
        # Prechod stromom na základe vzorky
        if node.leaf_value is None:
            split_value, feature_index = node.get_params()
            if sample[feature_index] <= split_value:
                return self._traverse_tree(node.get_left(), sample)
            else:
                return self._traverse_tree(node.get_right(), sample)
        else:
            return node.leaf_value

    def fit(self, X, Y):
        # Trénovanie stromu na dátach
        data = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
        self.root = Node()
        self._grow_tree(self.root, data, 1)

    def predict(self, X):
        # Predikcia na základe vstupných dát
        predictions = []
        for i in range(X.shape[0]):
            prediction = self._traverse_tree(self.root, X[i, :])
            predictions.append(prediction)
        return np.array(predictions).flatten()
