"""
Camada de carregamento de dados.

Responsabilidade UNICA: ler as matrizes ja prontas (train/test, linear/tree),
selecionar a coluna de target correta para a tarefa (binaria ou severidade),
e opcionalmente filtrar por subgrupo de sexo. Nao faz nenhuma transformacao
estatistica (isso ja foi feito na etapa de preparacao de dados, que este
projeto nao modifica).
"""

from dataclasses import dataclass

import pandas as pd

from datasets.registry import TARGET_COLUMNS, get_dataset_spec

VALID_TASKS = {
    "binary": "doenca",
    "severity": "num",
}

VALID_SUBGROUPS = {"all", "male", "female"}


@dataclass
class LoadedData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list
    dataset_name: str
    task: str
    subgroup: str
    target_column: str


def _select_target_column(task: str) -> str:
    if task not in VALID_TASKS:
        raise ValueError(f"Task '{task}' invalida. Opcoes: {list(VALID_TASKS.keys())}")
    return VALID_TASKS[task]


def _apply_subgroup_filter(df: pd.DataFrame, sex_column: str, subgroup: str) -> pd.DataFrame:
    if subgroup == "all":
        return df
    if subgroup == "male":
        return df[df[sex_column] == 1.0]
    if subgroup == "female":
        return df[df[sex_column] == 0.0]
    raise ValueError(f"Subgroup '{subgroup}' invalido. Opcoes: {VALID_SUBGROUPS}")


def load_dataset(dataset_name: str, task: str = "binary", subgroup: str = "all") -> LoadedData:
    """Carrega o dataset logico (`linear` ou `tree`), aplica o filtro de
    subgrupo (se houver) e separa X/y para a tarefa escolhida.

    Parameters
    ----------
    dataset_name: "linear" ou "tree"
    task: "binary" (target = doenca, 0/1) ou "severity" (target = num, 0-4)
    subgroup: "all", "male" ou "female" - filtra as amostras por sexo.
        Quando != "all", as colunas derivadas de sexo (sex, sex_Male,
        sex_x_age, sex_x_thalch, sex_x_oldpeak) sao removidas das features,
        pois se tornam constantes/degeneradas dentro do subgrupo.
    """
    if subgroup not in VALID_SUBGROUPS:
        raise ValueError(f"Subgroup '{subgroup}' invalido. Opcoes: {VALID_SUBGROUPS}")

    spec = get_dataset_spec(dataset_name)
    target_col = _select_target_column(task)

    train_df = pd.read_csv(spec.train_path)
    test_df = pd.read_csv(spec.test_path)

    train_df = _apply_subgroup_filter(train_df, spec.sex_column, subgroup)
    test_df = _apply_subgroup_filter(test_df, spec.sex_column, subgroup)

    drop_cols = set(TARGET_COLUMNS)
    if subgroup != "all":
        drop_cols.update(spec.sex_derived_columns)

    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].reset_index(drop=True)
    y_train = train_df[target_col].astype(int).reset_index(drop=True)
    X_test = test_df[feature_cols].reset_index(drop=True)
    y_test = test_df[target_col].astype(int).reset_index(drop=True)

    return LoadedData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_cols,
        dataset_name=dataset_name,
        task=task,
        subgroup=subgroup,
        target_column=target_col,
    )
