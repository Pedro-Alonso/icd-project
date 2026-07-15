"""
Registro de datasets.

Mapeia um nome logico de dataset ("linear" ou "tree") para os caminhos
fisicos dos arquivos gerados pela etapa de preparacao de dados
(features_engineer/). Este projeto NUNCA modifica esses arquivos -
apenas os le.

Tambem centraliza metadados dependentes do dataset:
- nome da coluna de sexo (difere entre "linear" e "tree" por causa do
  one-hot encoding aplicado na matriz linear)
- colunas derivadas de sexo, que devem ser removidas quando o experimento
  usa um subgrupo (male/female), pois se tornam degeneradas.
"""

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Colunas de identificacao/target que nunca sao features
TARGET_COLUMNS = {"doenca", "num"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    train_path: Path
    test_path: Path
    sex_column: str
    sex_derived_columns: tuple  # colunas a remover quando subgroup != "all"


_REGISTRY = {
    "linear": DatasetSpec(
        name="linear",
        train_path=DATA_DIR / "train_matrix_linear.csv",
        test_path=DATA_DIR / "test_matrix_linear.csv",
        sex_column="sex_Male",
        sex_derived_columns=("sex_Male", "sex_x_age", "sex_x_thalch", "sex_x_oldpeak"),
    ),
    "tree": DatasetSpec(
        name="tree",
        train_path=DATA_DIR / "train_matrix_tree.csv",
        test_path=DATA_DIR / "test_matrix_tree.csv",
        sex_column="sex",
        sex_derived_columns=("sex", "sex_x_age", "sex_x_thalch", "sex_x_oldpeak"),
    ),
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    if dataset_name not in _REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' desconhecido. Opcoes: {list(_REGISTRY.keys())}"
        )
    spec = _REGISTRY[dataset_name]
    if not spec.train_path.exists() or not spec.test_path.exists():
        raise FileNotFoundError(
            f"Arquivos do dataset '{dataset_name}' nao encontrados em {DATA_DIR}. "
            "Verifique se a etapa de preparacao de dados foi executada."
        )
    return spec


def available_datasets():
    return list(_REGISTRY.keys())
