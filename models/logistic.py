from typing import Any, Dict

from sklearn.linear_model import LogisticRegression

from models.base import ModelPlugin


class LogisticPlugin(ModelPlugin):
    name = "logistic"
    expected_dataset = "linear"
    supports_proba = True

    def create_model(self, params: Dict[str, Any], random_state: int):
        return LogisticRegression(
            random_state=random_state,
            max_iter=2000,
            **params,
        )

    def parameter_grid(self) -> Dict[str, list]:
        # penalty/solver combinados manualmente pois nem toda combinacao e valida:
        # 'l1' exige solver 'liblinear' ou 'saga'; 'l2' aceita varios solvers.
        # OBS: a partir do sklearn 1.8, o parametro 'penalty' esta sendo
        # depreciado em favor de 'l1_ratio' (l1_ratio=0 equivale a penalty='l2').
        # Usamos l1_ratio diretamente para evitar o FutureWarning e continuar
        # compativel com versoes futuras do sklearn.
        return {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs"],
            "l1_ratio": [0.0],  # 0.0 == equivalente a penalty='l2' (default)
            "class_weight": [None, "balanced"],
        }
