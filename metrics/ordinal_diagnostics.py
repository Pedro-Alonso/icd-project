"""
metrics/ordinal_diagnostics.py
================================

Diagnóstico de erro para tarefas de classificação ORDINAL — como `severity`,
onde as classes 0..n-1 representam níveis crescentes de gravidade (ex.:
graus de uma cardiopatia).

Por que este módulo existe
---------------------------
As métricas em `metrics/classification.py` (accuracy, precision, recall, F1,
ROC AUC, MCC) tratam **todo erro de classe como igualmente ruim**. Para um
problema ordinal isso é uma simplificação perigosa: confundir nível 4 com
nível 3 é um erro pequeno; confundir nível 4 com nível 0 é um erro grave — e,
na prática clínica, **subestimar** a gravidade (prever um nível mais baixo
que o real) tende a ser mais perigoso que superestimar (prever mais alto).

Este módulo calcula:
  - matriz de confusão completa (pra ver pra ONDE cada classe erra);
  - precision/recall/F1 por classe (quais níveis o modelo realmente
    reconhece, e quais ele ignora);
  - MAE (erro absoluto médio em "número de níveis de distância");
  - QWK — Quadratic Weighted Kappa (a métrica padrão da literatura pra
    classificação ordinal: concorda com o acaso = 0, concordância perfeita
    = 1, e penaliza erros grandes proporcionalmente ao QUADRADO da
    distância entre a classe prevista e a real);
  - taxa de acerto "dentro de 1 nível" (within_1_rate);
  - viés direcional: com que frequência o modelo subestima vs. superestima,
    e um "clinical_risk_score" que pune subestimar mais que superestimar
    (fator ajustável — o valor real do trade-off é uma decisão clínica, não
    estatística, então o padrão é só um ponto de partida);
  - a mesma decomposição (MAE, viés direcional) quebrada POR CLASSE
    VERDADEIRA, para responder exatamente perguntas do tipo "quando a
    gravidade real é 4, pra onde o modelo erra, e com que frequência?".

Não é uma "métrica" no sentido do contrato em `metrics/contracts.py` (que
exige um único float) — retorna um dict rico. Por isso não é registrado em
`metrics/registry.py`; é chamado diretamente na avaliação do conjunto de
TESTE (ver instruções de integração no final deste arquivo), não a cada fold
da CV — isso manteria o grid search rápido e mesmo assim dá o retrato mais
importante, que é o do modelo final.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support


def compute_ordinal_diagnostics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_classes: int,
    class_names: Optional[List[str]] = None,
    underestimate_penalty: float = 1.5,
) -> Dict:
    """
    Parâmetros
    ----------
    y_true, y_pred : arrays de inteiros 0..n_classes-1 (níveis de severidade,
        NA ORDEM CRESCENTE DE GRAVIDADE — isso é assumido, não verificado).
    n_classes : número total de níveis possíveis (5 para severity 0-4).
    class_names : rótulos legíveis para cada nível (default: "0", "1", ...).
    underestimate_penalty : quanto mais caro é subestimar a gravidade do que
        superestimar, na mesma distância. 1.5 = subestimar custa 50% mais.
        Ajuste ao custo clínico real do seu problema (1.0 = custo simétrico).

    Retorna um dict serializável em JSON, pronto para ser salvo dentro de
    test_results.json (ver rodapé deste arquivo).
    """
    # .ravel() "achata" arrays de formato (n, 1) para (n,) — alguns modelos
    # (ex.: CatBoostClassifier.predict()) devolvem uma coluna em vez de um
    # vetor simples, mesmo prevendo uma única classe por amostra.
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true e y_pred têm tamanhos diferentes depois de achatados "
            f"({y_true.shape} vs {y_pred.shape}) — isso não é um problema de "
            f"formato, e sim de contagem de amostras; confira se X_test e "
            f"y_test estão alinhados."
        )

    labels = list(range(n_classes))
    class_names = class_names or [str(i) for i in labels]
    if len(class_names) != n_classes:
        raise ValueError("class_names precisa ter exatamente n_classes elementos.")

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
            "support": int(supports[i]),
        }
        for i in labels
    }

    diff = y_pred - y_true  # > 0: superestimou a gravidade; < 0: subestimou
    abs_diff = np.abs(diff)

    mae = float(np.mean(abs_diff))
    exact_match_rate = float(np.mean(diff == 0))
    within_1_rate = float(np.mean(abs_diff <= 1))
    try:
        qwk = float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic"))
    except Exception:
        qwk = None
    mean_signed_error = float(np.mean(diff))
    underestimate_rate = float(np.mean(diff < 0))
    overestimate_rate = float(np.mean(diff > 0))

    # custo assimétrico: subestimar custa `underestimate_penalty`x mais do
    # que superestimar, à mesma distância. Menor = melhor.
    cost = np.where(diff < 0, underestimate_penalty * (diff.astype(float) ** 2), diff.astype(float) ** 2)
    clinical_risk_score = float(np.mean(cost))

    # por classe verdadeira: pra onde o modelo erra quando a classe real é X
    error_by_true_class = {}
    for i, cname in enumerate(class_names):
        mask = y_true == i
        n = int(mask.sum())
        if n == 0:
            error_by_true_class[cname] = None
            continue
        d = diff[mask]
        pred_dist = {class_names[k]: int(np.sum(y_pred[mask] == k)) for k in labels}
        error_by_true_class[cname] = {
            "n": n,
            "mae": float(np.mean(np.abs(d))),
            "mean_signed_error": float(np.mean(d)),  # >0 => tende a superestimar essa classe
            "underestimate_rate": float(np.mean(d < 0)),
            "overestimate_rate": float(np.mean(d > 0)),
            "exact_rate": float(np.mean(d == 0)),
            "predicted_class_distribution": pred_dist,
        }

    return {
        "n_classes": n_classes,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),  # linhas = classe verdadeira, colunas = classe prevista
        "per_class": per_class,
        "mae": mae,
        "qwk": qwk,
        "exact_match_rate": exact_match_rate,
        "within_1_rate": within_1_rate,
        "mean_signed_error": mean_signed_error,
        "underestimate_rate": underestimate_rate,
        "overestimate_rate": overestimate_rate,
        "underestimate_penalty_used": underestimate_penalty,
        "clinical_risk_score": clinical_risk_score,
        "error_by_true_class": error_by_true_class,
    }

