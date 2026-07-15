from typing import Any, Dict

from models.base import ModelPlugin

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class CatBoostPlugin(ModelPlugin):
    name = "catboost"
    expected_dataset = "tree"
    supports_proba = True

    def create_model(self, params: Dict[str, Any], random_state: int):
        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "catboost nao esta instalado. Rode: pip install catboost"
            )
        return CatBoostClassifier(
            random_state=random_state,
            verbose=False,
            allow_writing_files=False,
            **params,
        )

    def parameter_grid(self) -> Dict[str, list]:
        return {
            "iterations": [200, 400],
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5],
        }
