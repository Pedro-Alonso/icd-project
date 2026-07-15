"""
Registro de metricas.

Para adicionar uma nova metrica: escreva uma funcao em metrics/classification.py
(ou em um novo arquivo) seguindo o contrato de metrics/contracts.py, e
registre-a aqui com um nome. Nenhum outro modulo precisa ser alterado.
"""

from metrics import classification as clf

_METRICS = {
    "accuracy": clf.accuracy,
    "balanced_accuracy": clf.balanced_accuracy,
    "precision": clf.precision,
    "recall": clf.recall,
    "f1": clf.f1,
    "f2": clf.f2,
    "roc_auc": clf.roc_auc,
    "mcc": clf.mcc,
}


def get_metric_fn(metric_name: str):
    if metric_name not in _METRICS:
        raise ValueError(
            f"Metrica '{metric_name}' desconhecida. Disponiveis: {list(_METRICS.keys())}"
        )
    return _METRICS[metric_name]


def compute_all(metric_names, y_true, y_pred, y_proba, n_classes) -> dict:
    """Calcula todas as metricas solicitadas, retornando um dict
    {nome: valor|None}. Uma metrica que falhar (ex: excecao inesperada)
    grava None e nao interrompe as demais."""
    results = {}
    for name in metric_names:
        fn = get_metric_fn(name)
        try:
            results[name] = fn(y_true, y_pred, y_proba, n_classes)
        except Exception:
            results[name] = None
    return results


def available_metrics():
    return list(_METRICS.keys())
