from typing import Any, Dict

from models.base import ModelPlugin

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostPlugin(ModelPlugin):
    name = "xgboost"
    expected_dataset = "tree"
    supports_proba = True

    def create_model(self, params: Dict[str, Any], random_state: int):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost nao esta instalado. Rode: pip install xgboost"
            )
        return XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
            **params,
        )

    def parameter_grid(self) -> Dict[str, list]:
        return {
            "n_estimators": [200, 400],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
