from abstract.base_tree import DecisionTree
import numpy as np
from scipy import stats


class DecisionTreeClassifier(DecisionTree):
    # Inicializácia klasifikátora rozhodovacích stromov
    def __init__(self, max_depth=None, min_samples_split=2, loss='gini', balance_class_weights=False):
        # Volanie konštruktora nadradenej triedy
        super().__init__(max_depth, min_samples_split)
        self.loss = loss  # Funkcia straty: 'gini' alebo 'entropy'
        self.balance_class_weights = balance_class_weights  # Vyváženie váh tried
        self.class_weights = None  # Váhy tried, inicializované neskôr

    # Súkromná metóda na výpočet Giniho indexu
    def __gini(self, data):
        gini_value = 0
        total_count = data.shape[0]  # Celkový počet vzoriek
        unique_labels = np.unique(data[:, -1])  # Unikátne triedy
        for label, weight in zip(unique_labels, self.class_weights):
            # Počet vzoriek pre danú triedu
            label_count = data[data[:, -1] == label].shape[0]
            probability = weight * label_count / total_count  # Pravdepodobnosť triedy
            gini_value += probability * \
                (1 - probability)  # Výpočet Giniho indexu
        return gini_value

    # Súkromná metóda na výpočet entropie
    def __entropy(self, data):
        entropy_value = 0
        total_count = data.shape[0]  # Celkový počet vzoriek
        unique_labels = np.unique(data[:, -1])  # Unikátne triedy
        for label, weight in zip(unique_labels, self.class_weights):
            # Počet vzoriek pre danú triedu
            label_count = data[data[:, -1] == label].shape[0]
            probability = weight * label_count / total_count  # Pravdepodobnosť triedy
            entropy_value -= probability * \
                np.log2(probability)  # Výpočet entropie
        return entropy_value

    # Chránená metóda na výpočet nečistoty uzla
    def _impurity(self, dataset):
        # Výber funkcie straty na výpočet nečistoty
        impurity_value = None
        if self.loss == 'gini':
            impurity_value = self.__gini(dataset)
        elif self.loss == 'entropy':
            impurity_value = self.__entropy(dataset)
        # Vrátenie vypočítanej nečistoty
        return impurity_value

    # Chránená metóda na výpočet hodnoty listového uzla
    def _leaf_value(self, dataset):
        # Návrat najčastejšej hodnoty v listovom uzle
        return stats.mode(dataset[:, -1])[0]

    # Verejná metóda na získanie parametrov modelu
    def get_params(self, deep=False):
        return {'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'loss': self.loss,
                'balance_class_weights': self.balance_class_weights}

    # Trénovanie modelu rozhodovacieho stromu
    def fit(self, Xin, Yin):
        # Výpočet váh tried, ak je povolené vyváženie
        if self.balance_class_weights:
            eps = 1e-8  # Malá hodnota na zabránenie deleniu nulou
            total_samples = Yin.shape[0]  # Celkový počet vzoriek
            unique_count = np.unique(Yin).shape[0]  # Počet unikátnych tried
            # Počet vzoriek pre každú triedu
            counts = np.bincount(Yin.flatten().astype(int))
            self.class_weights = total_samples / \
                (unique_count * (counts + eps))  # Výpočet váh tried
        else:
            # Jednotkové váhy, ak nie je vyváženie povolené
            self.class_weights = np.ones(np.unique(Yin).shape[0])
        # Trénovanie modelu pomocou implementácie nadradenej triedy
        super().fit(Xin, Yin)
