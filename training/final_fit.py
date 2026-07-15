"""
Retreino final.

Depois que o Grid Search + CV determina os melhores hiperparametros
(ver persistence/summary_builder.py), este modulo treina um modelo NOVO
(do zero, nao reaproveita nenhum modelo de fold) usando 100% do conjunto
de treino. Esse e o unico modelo que sera avaliado no conjunto de teste.
"""

from typing import Any, Dict

from models.base import ModelPlugin
from training.trainer import train_estimator
from utils.seeding import derive_seed


def fit_final_model(
    plugin: ModelPlugin,
    best_params: Dict[str, Any],
    X_train,
    y_train,
    master_seed: int,
):
    final_seed = derive_seed(master_seed, "final_fit", plugin.name)
    estimator = train_estimator(plugin, best_params, X_train, y_train, random_state=final_seed)
    return estimator
