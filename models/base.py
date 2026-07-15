"""
Contrato de plugin de modelo.

Cada arquivo em models/ (logistic.py, random_forest.py, xgboost_model.py...)
deve definir uma classe que herda de ModelPlugin e implementa os metodos
abaixo. O pipeline NUNCA importa um algoritmo de ML diretamente - ele so
conhece essa interface. Isso e o que garante que trocar de algoritmo seja
so uma mudanca de config (model_name).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class ModelPlugin(ABC):
    #: nome usado em configs e no registry (ex: "logistic", "random_forest")
    name: str = "base"

    #: dataset que este modelo espera: "linear" (para modelos lineares/distancia,
    #: exige scaling/one-hot) ou "tree" (para modelos baseados em arvore).
    expected_dataset: str = "tree"

    #: se este modelo suporta predict_proba de forma nativa
    supports_proba: bool = True

    @abstractmethod
    def create_model(self, params: Dict[str, Any], random_state: int):
        """Instancia e retorna um estimator (nao treinado) com os
        hiperparametros fornecidos. `random_state` deve SEMPRE ser repassado
        ao estimator quando ele suportar, para reprodutibilidade."""
        raise NotImplementedError

    @abstractmethod
    def parameter_grid(self) -> Dict[str, list]:
        """Retorna o espaco de busca de hiperparametros deste modelo, no
        formato {nome_do_param: [lista de candidatos]}."""
        raise NotImplementedError

    def fit(self, estimator, X, y):
        """Treina o estimator. Comportamento padrao: chama .fit(). Plugins
        podem sobrescrever se precisarem de logica especial (ex: early
        stopping com conjunto de validacao interno)."""
        estimator.fit(X, y)
        return estimator

    def predict(self, estimator, X):
        return estimator.predict(X)

    def predict_proba(self, estimator, X):
        """Retorna a matriz de probabilidades (n_amostras, n_classes) ou
        None se o modelo nao suportar. O pipeline de metricas deve tratar
        None graciosamente (pulando metricas que dependem de proba, como
        roc_auc)."""
        if not self.supports_proba or not hasattr(estimator, "predict_proba"):
            return None
        return estimator.predict_proba(X)
