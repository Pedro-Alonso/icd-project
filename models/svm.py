from typing import Any, Dict

from sklearn.svm import SVC

from models.base import ModelPlugin


class SVMPlugin(ModelPlugin):
    name = "svm"
    expected_dataset = "linear"
    supports_proba = True

    def create_model(self, params: Dict[str, Any], random_state: int):
        return SVC(
            random_state=random_state,
            probability=True,  # necessario para predict_proba / roc_auc
            **params,
        )

    def parameter_grid(self) -> Dict[str, list]:
        return {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
            "class_weight": [None, "balanced"],
        }
