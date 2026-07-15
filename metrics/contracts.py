"""
Contrato de metrica.

Toda metrica registrada em metrics/registry.py deve ser uma funcao com a
assinatura:

    metric(y_true, y_pred, y_proba, n_classes) -> float | None

- y_true: array de labels verdadeiros
- y_pred: array de labels preditos (classe, nao probabilidade)
- y_proba: matriz (n_amostras, n_classes) de probabilidades, ou None se o
  modelo nao suportar predict_proba
- n_classes: numero de classes distintas no problema (2 para binario, 5
  para a tarefa de severidade)

Retornar None quando a metrica nao puder ser calculada (ex: roc_auc sem
y_proba disponivel) - o pipeline trata isso registrando o valor como null
no JSON, sem quebrar a execucao.
"""

from typing import Callable, Optional

import numpy as np

MetricFn = Callable[[np.ndarray, np.ndarray, Optional[np.ndarray], int], Optional[float]]
