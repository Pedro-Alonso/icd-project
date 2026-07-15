"""
Estrategia de validacao cruzada.

Usa RepeatedStratifiedKFold: estratifica pela classe (essencial com
desbalanceamento, especialmente na tarefa de severidade onde a classe 4
tem poucas amostras) e repete com seeds diferentes para reduzir a
variancia da estimativa em um dataset pequeno (~700 amostras de treino).

Ver a justificativa completa desta escolha na conversa de definicao da
arquitetura: com ~920 amostras totais, K=10 deixa folds de validacao
pequenos demais para metricas estaveis (ex: ROC-AUC); K=5 com varias
repeticoes e o ponto de equilibrio entre vies e variancia.
"""

from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from utils.seeding import derive_seed


def build_cv_splits(
    X, y, n_splits: int, n_repeats: int, master_seed: int
) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray]]:
    """Gera splits (repeat_idx, fold_idx, train_idx, val_idx).

    A seed do RepeatedStratifiedKFold e derivada da seed mestre, entao o
    MESMO conjunto de splits e gerado sempre que o pipeline for reexecutado
    com a mesma config - condicao necessaria para o checkpoint funcionar
    corretamente (senao "fold=2, repeat=1" apontaria para amostras
    diferentes em cada execucao).
    """
    cv_seed = derive_seed(master_seed, "cv_strategy")
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=cv_seed
    )

    for split_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        repeat_idx = split_idx // n_splits
        fold_idx = split_idx % n_splits
        yield repeat_idx, fold_idx, train_idx, val_idx
