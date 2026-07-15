"""
Registro de plugins de modelo.

Mapeia nome (string, usado em config) -> instancia de ModelPlugin.
Se a biblioteca de um modelo nao estiver instalada (ex: catboost), o
plugin simplesmente nao aparece no registry, em vez de quebrar o import
de todo o projeto.
"""

from models.logistic import LogisticPlugin
from models.random_forest import RandomForestPlugin
from models.svm import SVMPlugin

_PLUGINS = {
    "logistic": LogisticPlugin(),
    "random_forest": RandomForestPlugin(),
    "svm": SVMPlugin(),
}

try:
    from models.xgboost import XGBoostPlugin, XGBOOST_AVAILABLE

    if XGBOOST_AVAILABLE:
        _PLUGINS["xgboost"] = XGBoostPlugin()
except ImportError:
    pass

try:
    from models.lightgbm import LightGBMPlugin, LIGHTGBM_AVAILABLE

    if LIGHTGBM_AVAILABLE:
        _PLUGINS["lightgbm"] = LightGBMPlugin()
except ImportError:
    pass

try:
    from models.catboost import CatBoostPlugin, CATBOOST_AVAILABLE

    if CATBOOST_AVAILABLE:
        _PLUGINS["catboost"] = CatBoostPlugin()
except ImportError:
    pass


def get_model_plugin(model_name: str):
    if model_name not in _PLUGINS:
        raise ValueError(
            f"Modelo '{model_name}' desconhecido ou biblioteca nao instalada. "
            f"Disponiveis: {list(_PLUGINS.keys())}"
        )
    return _PLUGINS[model_name]


def available_models():
    return list(_PLUGINS.keys())
