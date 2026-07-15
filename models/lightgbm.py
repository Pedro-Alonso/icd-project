from typing import Any, Dict

from models.base import ModelPlugin

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class LightGBMPlugin(ModelPlugin):
    name = "lightgbm"
    expected_dataset = "tree"
    supports_proba = True

    def create_model(self, params: Dict[str, Any], random_state: int):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "lightgbm nao esta instalado. Rode: pip install lightgbm"
            )
        return LGBMClassifier(
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )

    def parameter_grid(self) -> Dict[str, list]:
        return {
            "n_estimators": [200, 400],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.05, 0.1],
            "min_child_samples": [5, 10, 20],
        }
