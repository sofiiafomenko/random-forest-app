from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.datasets import load_wine, load_diabetes
from sklearn.model_selection import StratifiedKFold, cross_validate
# Import vlastných metrík z externého súboru metrics.py
from metrics import (
    accuracy_score_manual,
    precision_score_manual,
    recall_score_manual,
    mean_absolute_error_manual,
    mean_squared_error_manual,
    r2_score_manual
)
from abc import ABC, abstractmethod
from decisiontrees import DecisionTreeClassifier, DecisionTreeRegressor

# Základná trieda pre algoritmus náhodného lesa


class RandomForest(ABC):
    # Inicializátor
    def __init__(self, n_trees=100):
        self.n_trees = n_trees
        self.trees = []

    # Súkromná funkcia na vytvorenie bootstrap vzoriek
    def __make_bootstraps(self, data):
        dc = {}
        unip = 0
        b_size = data.shape[0]
        idx = [i for i in range(b_size)]
        for b in range(self.n_trees):
            sidx = np.random.choice(idx, replace=True, size=b_size)
            b_samp = data[sidx, :]
            unip += len(set(sidx))
            oidx = list(set(idx) - set(sidx))
            o_samp = np.array([])
            if oidx:
                o_samp = data[oidx, :]
            dc['boot_' + str(b)] = {'boot': b_samp, 'test': o_samp}
        return dc

    # Verejná funkcia na získanie parametrov modelu
    def get_params(self, deep=False):
        return {'n_trees': self.n_trees}

    # Abstraktná metóda na vytvorenie modelu rozhodovacieho stromu
    @abstractmethod
    def _make_tree_model(self):
        pass

    # Chránená funkcia na trénovanie ansámblu
    def _train(self, X_train, y_train):
        training_data = np.concatenate(
            (X_train, y_train.reshape(-1, 1)), axis=1)
        dcBoot = self.__make_bootstraps(training_data)
        tree_m = self._make_tree_model()
        dcOob = {}
        for b in dcBoot:
            model = clone(tree_m)
            model.fit(dcBoot[b]['boot'][:, :-1], dcBoot[b]
                      ['boot'][:, -1].reshape(-1, 1))
            self.trees.append(model)
            if dcBoot[b]['test'].size:
                dcOob[b] = dcBoot[b]['test']
            else:
                dcOob[b] = np.array([])
        return dcOob

    # Chránená funkcia na predikciu s ansámblom
    def _predict(self, X):
        if not self.trees:
            print('Musíte najskôr natrénovať ansámbl pred predikciou!')
            return None
        predictions = []
        for m in self.trees:
            yp = m.predict(X)
            predictions.append(yp.reshape(-1, 1))
        ypred = np.mean(np.concatenate(predictions, axis=1), axis=1)
        return ypred


# Trieda pre náhodný les - klasifikátor
class RandomForestClassifier(RandomForest):
    # Inicializátor
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, loss='gini', balance_class_weights=False):
        super().__init__(n_trees)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.balance_class_weights = balance_class_weights

    # Chránená funkcia na vytvorenie modelu rozhodovacieho stromu
    def _make_tree_model(self):
        return DecisionTreeClassifier(max_depth=self.max_depth,
                                      min_samples_split=self.min_samples_split,
                                      loss=self.loss,
                                      balance_class_weights=self.balance_class_weights)

    # Verejná funkcia na získanie parametrov modelu
    def get_params(self, deep=False):
        return {'n_trees': self.n_trees,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'loss': self.loss,
                'balance_class_weights': self.balance_class_weights}

    # Trénovanie ansámblu
    def fit(self, X_train, y_train, print_metrics=False):
        dcOob = self._train(X_train, y_train)
        if print_metrics:
            accs = np.array([])
            pres = np.array([])
            recs = np.array([])
            for b, m in zip(dcOob, self.trees):
                if dcOob[b].size:
                    yp = m.predict(dcOob[b][:, :-1])
                    acc = accuracy_score_manual(dcOob[b][:, -1], yp)
                    pre = precision_score_manual(
                        dcOob[b][:, -1], yp, average='weighted')
                    rec = recall_score_manual(
                        dcOob[b][:, -1], yp, average='weighted')
                    accs = np.append(accs, acc)
                    pres = np.append(pres, pre)
                    recs = np.append(recs, rec)
            print("Štandardná chyba presnosti: %.2f" % np.std(accs))
            print("Štandardná chyba presnosti: %.2f" % np.std(pres))
            print("Štandardná chyba recall: %.2f" % np.std(recs))
            print("Priemerná presnosť: %.2f" % np.mean(accs))
            print("Priemerná presnosť: %.2f" % np.mean(pres))
            print("Priemerný recall: %.2f" % np.mean(recs))

    # Predikcia s ansámblom

    def predict(self, X):
        ypred = self._predict(X)
        return np.round(ypred).astype(int)


# Trieda pre náhodný les - regresor
class RandomForestRegressor(RandomForest):
    # Inicializátor
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, loss='mse'):
        super().__init__(n_trees)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss

    # Chránená funkcia na vytvorenie modelu rozhodovacieho stromu
    def _make_tree_model(self):
        return DecisionTreeRegressor(self.max_depth, self.min_samples_split, self.loss)

    # Verejná funkcia na získanie parametrov modelu
    def get_params(self, deep=False):
        return {'n_trees': self.n_trees,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'loss': self.loss}

    # Trénovanie ansámblu
    def fit(self, X_train, y_train, print_metrics=False):
        dcOob = self._train(X_train, y_train)
        if print_metrics:
            maes = np.array([])
            mses = np.array([])
            r2ss = np.array([])
            for b, m in zip(dcOob, self.trees):
                if dcOob[b].size:
                    yp = m.predict(dcOob[b][:, :-1])
                    mae = mean_absolute_error_manual(dcOob[b][:, -1], yp)
                    mse = mean_squared_error_manual(dcOob[b][:, -1], yp)
                    r2s = r2_score_manual(dcOob[b][:, -1], yp)
                    maes = np.append(maes, mae)
                    mses = np.append(mses, mse)
                    r2ss = np.append(r2ss, r2s)
            print("Štandardná chyba MAE: %.2f" % np.std(maes))
            print("Štandardná chyba MSE: %.2f" % np.std(mses))
            print("Štandardná chyba R2: %.2f" % np.std(r2ss))

    # Predikcia s ansámblom
    def predict(self, X):
        return self._predict(X)


# # ============================
# # Príklad použitia kódu:
# # ============================

# # Načítanie dát
# data = pd.read_csv('ds/c/heart.csv')

# # Uloženie pôvodného DataFrame pre ďalšie použitie
# original_df = data.copy()

# # Určenie cieľového stĺpca (posledný stĺpec)
# target_col = data.columns[-1]

# # Konverzia kategórií (okrem cieľového stĺpca) na číselné kódy
# feature_columns = data.columns[:-1]

# for col in feature_columns:
#     if data[col].dtype == 'object' or str(data[col].dtype) == 'category':
#         data[col] = data[col].astype('category').cat.codes

# # Ak ide o klasifikáciu, spracovať aj cieľový stĺpec
# selected_task = 'classification'  # Zadajte 'classification' alebo 'regression'
# if selected_task == 'classification':
#     if data[target_col].dtype == 'object' or str(data[target_col].dtype) == 'category':
#         data[target_col] = data[target_col].astype('category')
#         target_categories = data[target_col].cat.categories
#         data[target_col] = data[target_col].cat.codes
#     else:
#         target_categories = None
# else:
#     target_categories = None

# # Rozdelenie na X (vstupy) a y (cieľové hodnoty)
# X = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values

# # Inicializácia a trénovanie modelu
# if selected_task == 'classification':
#     model = RandomForestClassifier(
#         n_trees=100, max_depth=None, min_samples_split=2, balance_class_weights=False)
# else:
#     model = RandomForestRegressor(
#         n_trees=100, max_depth=None, min_samples_split=2)


# model.fit(X, y, print_metrics=True)
