"""
Implementacoes concretas de metricas de classificacao.

Todas sao "multiclass-aware": quando n_classes > 2 (tarefa de severidade,
0-4), usam average='macro' (nao ponderado por suporte, para nao mascarar
desempenho ruim nas classes minoritarias) e, no caso de roc_auc,
multi_class='ovr'.
"""

from typing import Optional

import numpy as np
from sklearn import metrics as skm


def accuracy(y_true, y_pred, y_proba, n_classes) -> Optional[float]:
    return float(skm.accuracy_score(y_true, y_pred))


def balanced_accuracy(y_true, y_pred, y_proba, n_classes) -> Optional[float]:
    return float(skm.balanced_accuracy_score(y_true, y_pred))


def precision(y_true, y_pred, y_proba, n_classes) -> Optional[float]:
    average = "binary" if n_classes == 2 else "macro"
    return float(skm.precision_score(y_true, y_pred, average=average, zero_division=0))


def recall(y_true, y_pred, y_proba, n_classes) -> Optional[float]:
    average = "binary" if n_classes == 2 else "macro"
    return float(skm.recall_score(y_true, y_pred, average=average, zero_division=0))


def f1(y_true, y_pred, y_proba, n_classes) -> Optional[float]:
    average = "binary" if n_classes == 2 else "macro"
    return float(skm.f1_score(y_true, y_pred, average=average, zero_division=0))


def f2(y_true, y_pred, y_proba, n_classes) -> Optional[float]:
    average = "binary" if n_classes == 2 else "macro"
    return float(skm.fbeta_score(y_true, y_pred, beta=2.0, average=average, zero_division=0))


def roc_auc(y_true, y_pred, y_proba, n_classes) -> Optional[float]:
    if y_proba is None:
        return None
    try:
        if n_classes == 2:
            # y_proba tem 2 colunas; usa a coluna da classe positiva (1)
            return float(skm.roc_auc_score(y_true, y_proba[:, 1]))
        return float(
            skm.roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        )
    except ValueError:
        # pode falhar se algum fold nao contiver todas as classes
        return None


def mcc(y_true, y_pred, y_proba, n_classes) -> Optional[float]:
    return float(skm.matthews_corrcoef(y_true, y_pred))
