"""
Treinamento generico.

Responsabilidade unica: dado um plugin de modelo, um conjunto de
hiperparametros e dados de treino, instanciar e treinar o estimator. Nao
sabe nada sobre CV, grid search, metricas ou persistencia.
"""

from typing import Any, Dict

from models.base import ModelPlugin


def train_estimator(plugin: ModelPlugin, params: Dict[str, Any], X_train, y_train, random_state: int):
    estimator = plugin.create_model(params, random_state=random_state)
    estimator = plugin.fit(estimator, X_train, y_train)
    return estimator
