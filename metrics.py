import numpy as np

# Metriky pre klasifikáciu


def accuracy_score_manual(y_true, y_pred):
    """Presnosť: podiel správnych predpovedí."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)


def precision_score_manual(y_true, y_pred, average='weighted', zero_division=0):
    """
    Presnosť (precision) pre multiklasifikačnú úlohu.

    average:
        'weighted' - vážený priemer podľa tried,
        'macro' - jednoduchý priemer podľa tried.
    zero_division: hodnota vrátená pri delení nulou.
    """

    classes = np.unique(y_true)
    precisions = []
    supports = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        if (tp + fp) > 0:
            prec = tp / (tp + fp)
        else:
            prec = zero_division
        precisions.append(prec)
        supports.append(np.sum(y_true == c))

    precisions = np.array(precisions)
    supports = np.array(supports)

    if average == 'weighted':
        return np.sum(precisions * supports) / np.sum(supports)
    elif average == 'macro':
        return np.mean(precisions)
    else:
        raise ValueError(
            "Unsupported average parameter. Use 'weighted' or 'macro'.")


def recall_score_manual(y_true, y_pred, average='weighted', zero_division=0):
    """
    Úplnosť (recall) pre multiklasifikačnú úlohu.

    average:
        'weighted' - vážený priemer podľa tried,
        'macro' - jednoduchý priemer podľa tried.
    zero_division: hodnota vrátená pri delení nulou.
    """
    classes = np.unique(y_true)
    recalls = []
    supports = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if (tp + fn) > 0:
            rec = tp / (tp + fn)
        else:
            rec = zero_division
        recalls.append(rec)
        supports.append(np.sum(y_true == c))

    recalls = np.array(recalls)
    supports = np.array(supports)

    if average == 'weighted':
        return np.sum(recalls * supports) / np.sum(supports)
    elif average == 'macro':
        return np.mean(recalls)
    else:
        raise ValueError(
            "Unsupported average parameter. Use 'weighted' or 'macro'.")


def confusion_matrix_manual(y_true, y_pred):
    """
    Matica chýb: prienik skutočných a predpovedaných tried.
    Vracia maticu, kde riadky zodpovedajú skutočným triedam,
    a stĺpce – predpovedaným.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            matrix[i, j] = np.sum((y_true == c1) & (y_pred == c2))
    return matrix

# Metriky pre regresiu


def mean_absolute_error_manual(y_true, y_pred):
    """Priemerná absolútna chyba (MAE)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error_manual(y_true, y_pred):
    """Priemerná kvadratická chyba (MSE)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred)**2)


def r2_score_manual(y_true, y_pred):
    """Koeficient determinácie (R2)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total_variance = np.sum((y_true - np.mean(y_true))**2)
    residual_variance = np.sum((y_true - y_pred)**2)
    return 1 - residual_variance / total_variance
