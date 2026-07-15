from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier

from models.base import ModelPlugin


class RandomForestPlugin(ModelPlugin):
    name = "random_forest"
    expected_dataset = "tree"
    supports_proba = True

    def create_model(self, params: Dict[str, Any], random_state: int):
        return RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            **params,
        )

    def parameter_grid(self) -> Dict[str, list]:
        return {
            "n_estimators": [200, 400, 800],
            "max_depth": [None, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "class_weight": [None, "balanced"],
        }
