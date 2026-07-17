"""
reporting/compare_models.py
============================

Compara, entre si, os modelos já treinados pelo pipeline (`pipeline.run_model`),
usando os `test_results.json` gravados em `results/{model}__{dataset}__{task}__{subgroup}/`.

Formato esperado de test_results.json (o campo "diagnostics" é OPCIONAL —
ver metrics/ordinal_diagnostics.py; sem ele, o script funciona normalmente,
só sem os gráficos de matriz de confusão / viés direcional):

    {
      "model": "logistic", "dataset": "linear", "task": "severity", "subgroup": "all",
      "best_hyperparameters": {...}, "n_train_total": 736, "n_test": 184,
      "metrics": {"accuracy": ..., "balanced_accuracy": ..., ..., "roc_auc": ..., "mcc": ...},
      "diagnostics": {                              <- opcional, ver metrics/ordinal_diagnostics.py
        "n_classes": 5, "class_names": ["0","1","2","3","4"],
        "confusion_matrix": [[...], ...],
        "per_class": {"0": {"precision":..,"recall":..,"f1":..,"support":..}, ...},
        "mae": .., "qwk": .., "exact_match_rate": .., "within_1_rate": ..,
        "mean_signed_error": .., "underestimate_rate": .., "overestimate_rate": ..,
        "clinical_risk_score": ..,
        "error_by_true_class": {"0": {...}, ...}
      },
      "timestamp": "..."
    }

Por que a versão anterior deste script não era boa o suficiente
-------------------------------------------------------------------
A primeira versão comparava modelos só pelas métricas agregadas (roc_auc,
balanced_accuracy, f1, mcc...), que tratam QUALQUER erro de classe como
igualmente ruim. Numa tarefa ORDINAL como `severity` (níveis 0-4 de
gravidade), isso é enganoso: confundir nível 4 com nível 3 é um erro
pequeno; confundir nível 4 com nível 0 é grave — e subestimar a gravidade
tende a ser clinicamente mais perigoso que superestimar. Esta versão:

  1. Lê o bloco "diagnostics" (quando presente) de cada test_results.json;
  2. Para grupos (task/subgroup) com diagnósticos disponíveis, desenha:
       - matriz de confusão por modelo (contagens + normalizada por classe
         verdadeira, ou seja, recall visual);
       - heatmap de recall por classe x modelo (quais níveis cada modelo
         realmente reconhece);
       - heatmap de erro médio COM SINAL por classe verdadeira x modelo
         (azul = tende a subestimar aquela classe, vermelho = tende a
         superestimar) — é a resposta direta para "4 virar 3 é diferente de
         4 virar 0";
       - barras de QWK, MAE, within-1-rate e clinical_risk_score;
       - barras de viés direcional (taxa de subestimação vs. superestimação);
  3. Troca o critério de "score robusto" para QWK + within_1_rate quando o
     grupo é ordinal (em vez de balanced_accuracy/mcc/f1, que ignoram
     distância e direção do erro);
  4. Detecta e avisa quando um modelo que aparece em outros grupos está
     ausente do grupo atual (ex.: svm sumiu de severity/all — pode ter
     falhado silenciosamente);
  5. Trata grupos com um único modelo como o que são — não finge uma
     "comparação" que não existe (pula gráficos de trade-off/ranking,
     mostra só a tabela e um aviso);
  6. Gera um veredito em texto por grupo, não só tabelas e gráficos.

Onde colocar este arquivo
---------------------------
    reporting/
        __init__.py             <- crie vazio, se ainda não existir
        compare_models.py       <- este arquivo

No mesmo nível de models/, metrics/, pipeline/, persistence/, training/.
Só lê arquivos em results/ — não precisa importar nada do resto do projeto,
exceto se usar --rerun-missing (importa pipeline.run_model).

Uso
---
    python -m reporting.compare_models
    python -m reporting.compare_models --task severity --open-browser
    python -m reporting.compare_models --rerun-missing
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuração de métricas
# ---------------------------------------------------------------------------

ALL_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "f2", "roc_auc", "mcc"]

# métricas robustas "genéricas" (usadas quando o grupo NÃO tem diagnósticos
# ordinais) — accuracy fica de fora por ser enganosa sob desbalanceamento.
DEFAULT_ROBUST_METRICS: List[Tuple[str, str]] = [
    ("balanced_accuracy", "max"), ("mcc", "max"), ("f1", "max")
]

# quando o grupo TEM diagnósticos ordinais, o score robusto passa a usar
# estas colunas (extraídas de "diagnostics") em vez das de cima.
ORDINAL_ROBUST_METRICS: List[Tuple[str, str]] = [
    ("qwk", "max"), ("within_1_rate", "max"), ("clinical_risk_score", "min")
]

DIAGNOSTIC_SCALAR_FIELDS = [
    "mae", "qwk", "exact_match_rate", "within_1_rate", "mean_signed_error",
    "underestimate_rate", "overestimate_rate", "clinical_risk_score",
]


def _rescale_mcc(value: Optional[float]) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return (value + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Carregamento dos dados
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    model: str
    dataset: str
    task: str
    subgroup: str
    best_hyperparameters: Dict[str, Any]
    n_train_total: Optional[int]
    n_test: Optional[int]
    metrics: Dict[str, Optional[float]]
    timestamp: Optional[str]
    source_dir: Path
    cv_std: Dict[str, Optional[float]] = field(default_factory=dict)
    cv_n_runs: Optional[int] = None
    diagnostics: Optional[Dict[str, Any]] = None

    @property
    def group_key(self) -> str:
        return f"{self.task} / {self.subgroup}"


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[aviso] não consegui ler {path}: {exc}", file=sys.stderr)
        return None


def _hyperparams_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return {k: a.get(k) for k in a} == {k: b.get(k) for k in a}


def load_experiment(dir_path: Path) -> Optional[ExperimentResult]:
    test_path = dir_path / "test_results.json"
    if not test_path.exists():
        return None
    test = _load_json(test_path)
    if test is None:
        return None

    required = {"model", "dataset", "task", "subgroup", "metrics"}
    missing = required - test.keys()
    if missing:
        print(f"[aviso] {test_path} não tem os campos {missing}; ignorando.", file=sys.stderr)
        return None

    cv_std: Dict[str, Optional[float]] = {}
    cv_n_runs = None
    summary_path = dir_path / "summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        if summary and summary.get("best_configuration"):
            best_cfg = summary["best_configuration"]
            if _hyperparams_match(test.get("best_hyperparameters", {}), best_cfg.get("hyperparameters", {})):
                cv_n_runs = best_cfg.get("n_runs")
                for m, stats in (best_cfg.get("metrics") or {}).items():
                    if isinstance(stats, dict):
                        cv_std[m] = stats.get("std")

    diagnostics = test.get("diagnostics") or None

    return ExperimentResult(
        model=test["model"], dataset=test["dataset"], task=test["task"], subgroup=test["subgroup"],
        best_hyperparameters=test.get("best_hyperparameters", {}),
        n_train_total=test.get("n_train_total"), n_test=test.get("n_test"),
        metrics={m: test["metrics"].get(m) for m in ALL_METRICS},
        timestamp=test.get("timestamp"), source_dir=dir_path,
        cv_std=cv_std, cv_n_runs=cv_n_runs, diagnostics=diagnostics,
    )


def discover_experiments(results_dir: Path) -> List[ExperimentResult]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Diretório de resultados não encontrado: {results_dir}")
    experiments = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        exp = load_experiment(child)
        if exp is not None:
            experiments.append(exp)
    return experiments


def find_incomplete_experiments(results_dir: Path) -> List[Path]:
    incomplete = []
    if not results_dir.exists():
        return incomplete
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "summary.json").exists() and not (child / "test_results.json").exists():
            incomplete.append(child)
    return incomplete


def rerun_missing(results_dir: Path) -> None:
    incomplete = find_incomplete_experiments(results_dir)
    if not incomplete:
        print("Nenhum experimento incompleto encontrado (todos têm test_results.json).")
        return
    try:
        from pipeline.run_model import run_model
    except ImportError:
        print(
            "[erro] não consegui importar pipeline.run_model. Rode a partir da raiz do "
            "projeto: `python -m reporting.compare_models --rerun-missing`.",
            file=sys.stderr,
        )
        return
    for dir_path in incomplete:
        parts = dir_path.name.split("__")
        if len(parts) != 4:
            print(f"[aviso] não consegui interpretar o nome da pasta {dir_path.name}; pulando.")
            continue
        model_name, dataset, task, subgroup = parts
        print(f"Completando experimento: {dir_path.name} ...")
        run_model(model_name=model_name, dataset=dataset, task=task, subgroup=subgroup)
    print("Concluído. Rode de novo sem --rerun-missing para gerar a comparação.")


# ---------------------------------------------------------------------------
# Montagem da tabela de comparação
# ---------------------------------------------------------------------------

def to_dataframe(experiments: List[ExperimentResult]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        row = {
            "model": exp.model, "dataset": exp.dataset, "task": exp.task, "subgroup": exp.subgroup,
            "group": exp.group_key, "n_train_total": exp.n_train_total, "n_test": exp.n_test,
            "timestamp": exp.timestamp, "hyperparameters": json.dumps(exp.best_hyperparameters, sort_keys=True),
            "cv_n_runs": exp.cv_n_runs, "source_dir": str(exp.source_dir),
            "has_diagnostics": exp.diagnostics is not None,
        }
        for m in ALL_METRICS:
            row[m] = exp.metrics.get(m)
            row[f"{m}_cv_std"] = exp.cv_std.get(m)
        if exp.diagnostics:
            for f_ in DIAGNOSTIC_SCALAR_FIELDS:
                row[f_] = exp.diagnostics.get(f_)
        else:
            for f_ in DIAGNOSTIC_SCALAR_FIELDS:
                row[f_] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["mcc_scaled"] = df["mcc"].apply(_rescale_mcc)
    return df.sort_values(["group", "model"]).reset_index(drop=True)


def compute_rankings(df: pd.DataFrame, primary_metric: str, robust_metrics_by_group: Dict[str, List[Tuple[str, str]]]) -> pd.DataFrame:
    """Adiciona rank pela métrica primária e um score robusto (que pode
    variar de definição por grupo, dependendo se há diagnósticos ordinais)."""
    df = df.copy()
    df["primary_rank"] = np.nan
    df["robust_score"] = np.nan
    df["robust_rank"] = np.nan
    df["robust_metrics_used"] = ""

    for group, idx in df.groupby("group").groups.items():
        sub = df.loc[idx]
        df.loc[idx, "primary_rank"] = sub[primary_metric].rank(ascending=False, na_option="bottom")

        robust_metrics = robust_metrics_by_group.get(group, DEFAULT_ROBUST_METRICS)
        norm_cols = []
        for m, direction in robust_metrics:
            if m not in sub.columns or sub[m].isna().all():
                continue
            col = sub[m]
            lo, hi = col.min(), col.max()
            norm_name = f"_norm_{m}"
            if pd.isna(lo) or pd.isna(hi) or hi == lo:
                df.loc[idx, norm_name] = 0.5
            else:
                normed = (col - lo) / (hi - lo)
                df.loc[idx, norm_name] = normed if direction == "max" else (1 - normed)
            norm_cols.append(norm_name)

        if norm_cols:
            df.loc[idx, "robust_score"] = df.loc[idx, norm_cols].mean(axis=1)
            df.loc[idx, "robust_rank"] = df.loc[idx, "robust_score"].rank(ascending=False, na_option="bottom")
            df.loc[idx, "robust_metrics_used"] = ", ".join(m for m, _ in robust_metrics if f"_norm_{m}" in norm_cols)

    df = df.drop(columns=[c for c in df.columns if c.startswith("_norm_")])
    return df


def detect_missing_models(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Para cada grupo, aponta modelos que existem em OUTROS grupos mas não
    neste — sinal de que o experimento pode ter falhado silenciosamente."""
    all_models = set(df["model"].unique())
    missing_by_group = {}
    for group, sub in df.groupby("group"):
        present = set(sub["model"].unique())
        missing = sorted(all_models - present)
        if missing:
            missing_by_group[group] = missing
    return missing_by_group


def build_verdicts(
    df: pd.DataFrame,
    primary_metric: str,
    missing_by_group: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Gera, por grupo, uma lista de frases (o 'veredito em texto')."""
    verdicts: Dict[str, List[str]] = {}
    for group, sub in df.groupby("group"):
        msgs = []
        n_models = sub["model"].nunique()

        if group in missing_by_group:
            msgs.append(
                f"⚠️ Modelo(s) ausente(s) neste grupo, mas presentes em outros: "
                f"{', '.join(missing_by_group[group])}. Verifique se o experimento rodou ou falhou."
            )

        if n_models < 2:
            msgs.append(
                f"Só há 1 modelo neste grupo ({sub['model'].iloc[0]}) — não há comparação real "
                f"a fazer ainda. Rode outros modelos para esta task/subgroup antes de tirar conclusões."
            )
            verdicts[group] = msgs
            continue

        best_primary = sub.loc[sub["primary_rank"].idxmin()]
        if sub["robust_rank"].notna().any():
            best_robust = sub.loc[sub["robust_rank"].idxmin()]
        else:
            best_robust = best_primary

        if best_primary["model"] == best_robust["model"]:
            msgs.append(
                f"Métrica primária ({primary_metric}) e score robusto concordam: "
                f"'{best_primary['model']}' é o melhor pelos dois critérios."
            )
        else:
            msgs.append(
                f"Métrica primária ({primary_metric}) escolheria '{best_primary['model']}' "
                f"({best_primary[primary_metric]:.4f}), mas o score robusto "
                f"(usando: {best_robust['robust_metrics_used']}) escolheria "
                f"'{best_robust['model']}' (score={best_robust['robust_score']:.3f})."
            )

        has_diag = sub["has_diagnostics"].any()
        if has_diag:
            diag_sub = sub[sub["has_diagnostics"]]
            worst_bias = diag_sub.loc[diag_sub["underestimate_rate"].idxmax()]
            if worst_bias["underestimate_rate"] > 0.25:
                msgs.append(
                    f"⚠️ '{worst_bias['model']}' subestima a severidade real em "
                    f"{worst_bias['underestimate_rate']*100:.1f}% dos casos de teste "
                    f"(mean_signed_error={worst_bias['mean_signed_error']:.2f}) — isso é "
                    f"potencialmente mais perigoso clinicamente do que superestimar. "
                    f"Veja o heatmap de viés por classe verdadeira antes de decidir."
                )
            best_qwk = diag_sub.loc[diag_sub["qwk"].idxmax()]
            msgs.append(
                f"Pelo critério ordinal (QWK — penaliza erros grandes mais que pequenos), "
                f"o melhor modelo é '{best_qwk['model']}' (QWK={best_qwk['qwk']:.3f}, "
                f"MAE={best_qwk['mae']:.2f} níveis, within±1={best_qwk['within_1_rate']*100:.1f}%)."
            )
        else:
            msgs.append(
                "Sem diagnósticos ordinais para este grupo (rode com o patch de "
                "metrics/ordinal_diagnostics.py para obter matriz de confusão, QWK e viés direcional)."
            )

        small_n = sub[sub["n_test"].fillna(0) < 50]
        if not small_n.empty:
            msgs.append(
                f"⚠️ n_test pequeno neste grupo ({sorted(small_n['n_test'].unique())}) — "
                f"estimativas de métrica têm margem de erro grande; use com cautela."
            )

        verdicts[group] = msgs
    return verdicts


# ---------------------------------------------------------------------------
# Visualizações — métricas agregadas
# ---------------------------------------------------------------------------

plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def _model_colors(models: List[str]) -> Dict[str, str]:
    return {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(sorted(models))}


def _safe_slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_")


def plot_metric_bars(sub: pd.DataFrame, group: str, metric: str, colors: Dict[str, str], out_path: Path,
                      higher_is_better: bool = True) -> None:
    sub = sub.dropna(subset=[metric]).sort_values(metric, ascending=not higher_is_better)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(max(4.5, 0.9 * len(sub) + 2), 4))
    yerr = sub[f"{metric}_cv_std"].fillna(0.0) if f"{metric}_cv_std" in sub.columns else None
    bars = ax.bar(sub["model"], sub[metric], yerr=yerr, capsize=4, color=[colors[m] for m in sub["model"]])
    for bar, val in zip(bars, sub[metric]):
        ax.annotate(f"{val:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax.set_ylabel(metric)
    arrow = "maior é melhor" if higher_is_better else "menor é melhor"
    ax.set_title(f"{metric} por modelo — {group} ({arrow})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_directional_bias(sub: pd.DataFrame, group: str, colors: Dict[str, str], out_path: Path) -> None:
    """Barras divergentes: taxa de subestimação (esquerda) vs. superestimação (direita)."""
    sub = sub.dropna(subset=["underestimate_rate", "overestimate_rate"]).sort_values("underestimate_rate")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, max(3, 0.6 * len(sub) + 1.5)))
    y = np.arange(len(sub))
    ax.barh(y, -sub["underestimate_rate"], color="#4C72B0", label="subestima severidade (mais arriscado)")
    ax.barh(y, sub["overestimate_rate"], color="#DD8452", label="superestima severidade")
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(-row["underestimate_rate"] - 0.01, i, f"{row['underestimate_rate']*100:.0f}%", ha="right", va="center", fontsize=9)
        ax.text(row["overestimate_rate"] + 0.01, i, f"{row['overestimate_rate']*100:.0f}%", ha="left", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["model"])
    ax.axvline(0, color="black", linewidth=0.8)
    max_extent = max(sub["underestimate_rate"].max(), sub["overestimate_rate"].max()) + 0.15
    ax.set_xlim(-max_extent, max_extent)
    ax.set_xlabel("fração dos casos de teste")
    ax.set_title(f"Viés direcional do erro — {group}")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_tradeoff_scatter(sub: pd.DataFrame, group: str, x_metric: str, y_metric: str, colors: Dict[str, str], out_path: Path) -> None:
    sub = sub.dropna(subset=[x_metric, y_metric])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.8))
    for _, row in sub.iterrows():
        ax.scatter(row[x_metric], row[y_metric], s=140, color=colors[row["model"]], edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(row["model"], (row[x_metric], row[y_metric]), textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f"Trade-off: {x_metric} vs. {y_metric} — {group}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualizações — diagnósticos ordinais (matriz de confusão / por classe)
# ---------------------------------------------------------------------------

def plot_confusion_matrices(experiments: List[ExperimentResult], group: str, out_path: Path) -> None:
    """Grade com a matriz de confusão (normalizada por linha = recall) de
    cada modelo do grupo que tem diagnósticos."""
    exps = [e for e in experiments if e.group_key == group and e.diagnostics]
    if not exps:
        return
    n = len(exps)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)

    for i, exp in enumerate(exps):
        ax = axes[i // ncols][i % ncols]
        diag = exp.diagnostics
        cm = np.array(diag["confusion_matrix"], dtype=float)
        class_names = diag["class_names"]
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_xlabel("previsto")
        ax.set_ylabel("real")
        ax.set_title(exp.model)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                color = "white" if cm_norm[r, c] > 0.55 else "black"
                ax.text(c, r, int(cm[r, c]), ha="center", va="center", color=color, fontsize=9)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Matriz de confusão por modelo (cor = recall da classe) — {group}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_recall_heatmap(experiments: List[ExperimentResult], group: str, out_path: Path) -> None:
    exps = [e for e in experiments if e.group_key == group and e.diagnostics]
    if not exps:
        return
    class_names = exps[0].diagnostics["class_names"]
    models = [e.model for e in exps]
    matrix = np.array([[e.diagnostics["per_class"][c]["recall"] for c in class_names] for e in exps])

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(class_names) + 2), max(3, 0.6 * len(models) + 1.5)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels([f"nível {c}" for c in class_names])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < 0.4 or val > 0.75 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="recall (sensibilidade) da classe")
    ax.set_title(f"Recall por nível de severidade x modelo — {group}\n(quais níveis cada modelo realmente reconhece)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_signed_error_by_true_class_heatmap(experiments: List[ExperimentResult], group: str, out_path: Path) -> None:
    """O gráfico que responde diretamente a: '4 virar 3 é diferente de 4
    virar 0'. Para cada classe VERDADEIRA, mostra o erro médio com sinal
    (negativo = modelo tende a subestimar aquele nível; positivo = tende a
    superestimar)."""
    exps = [e for e in experiments if e.group_key == group and e.diagnostics]
    if not exps:
        return
    class_names = exps[0].diagnostics["class_names"]
    models = [e.model for e in exps]

    matrix = np.full((len(models), len(class_names)), np.nan)
    for i, e in enumerate(exps):
        for j, c in enumerate(class_names):
            info = e.diagnostics["error_by_true_class"].get(c)
            if info is not None:
                matrix[i, j] = info["mean_signed_error"]

    vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    vmax = max(vmax, 0.1)

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(class_names) + 2), max(3, 0.6 * len(models) + 1.5)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels([f"real = {c}" for c in class_names])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            if np.isnan(val):
                continue
            ax.text(c, r, f"{val:+.2f}", ha="center", va="center",
                    color="white" if abs(val) > vmax * 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="erro médio com sinal (previsto − real)\nazul = subestima · vermelho = superestima")
    ax.set_title(f"Direção e magnitude do erro por classe verdadeira — {group}\n(azul escuro = modelo perigosamente otimista)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Relatório HTML
# ---------------------------------------------------------------------------

def _img_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_html_report(
    df: pd.DataFrame,
    group_figures: Dict[str, Dict[str, Path]],
    verdicts: Dict[str, List[str]],
    primary_metric: str,
    out_html: Path,
) -> None:
    base_cols = ["model", "dataset", "n_test", primary_metric, "robust_score", "primary_rank", "robust_rank"]
    ordinal_cols = ["qwk", "mae", "within_1_rate", "underestimate_rate", "overestimate_rate", "clinical_risk_score"]

    sections = []
    for group, sub in df.groupby("group"):
        sub_sorted = sub.sort_values("primary_rank")
        cols = [c for c in base_cols if c in sub_sorted.columns]
        if sub_sorted["has_diagnostics"].any():
            cols += [c for c in ordinal_cols if c in sub_sorted.columns]
        cols += ["hyperparameters"]
        table_html = sub_sorted[cols].round(4).to_html(index=False, border=0, classes="tbl")

        verdict_html = "".join(f"<li>{m}</li>" for m in verdicts.get(group, []))

        figs_html = ""
        for fig_name, fig_path in group_figures.get(group, {}).items():
            if fig_path.exists():
                b64 = _img_to_base64(fig_path)
                figs_html += f'<div class="fig"><h4>{fig_name}</h4><img src="data:image/png;base64,{b64}" /></div>'

        sections.append(f"""
        <section>
          <h2>{group}</h2>
          <div class="callout"><h3>Veredito</h3><ul>{verdict_html}</ul></div>
          {table_html}
          <div class="fig-grid">{figs_html}</div>
        </section>
        """)

    html = f"""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8" />
<title>Comparação de modelos</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; color: #1a1a1a; }}
  h1 {{ margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-top: 0; }}
  table.tbl {{ border-collapse: collapse; width: 100%; margin: 14px 0 22px 0; font-size: 13px; }}
  table.tbl th, table.tbl td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
  table.tbl th {{ background: #f4f4f6; }}
  section {{ margin-bottom: 48px; padding-bottom: 24px; border-bottom: 1px solid #eee; }}
  .fig-grid {{ display: flex; flex-wrap: wrap; gap: 18px; }}
  .fig {{ flex: 1 1 460px; max-width: 680px; }}
  .fig img {{ width: 100%; height: auto; border: 1px solid #eee; border-radius: 6px; }}
  .fig h4 {{ margin: 6px 0; color: #444; font-weight: 600; }}
  .callout {{ background: #fff7e6; border: 1px solid #ffe1a8; border-radius: 8px; padding: 12px 18px; margin: 12px 0 18px 0; }}
  .callout h3 {{ margin-top: 0; font-size: 15px; }}
  .callout li {{ margin-bottom: 4px; }}
</style>
</head>
<body>
<h1>Comparação de modelos entre experimentos</h1>
<p class="subtitle">Baseado nos <code>test_results.json</code> de cada experimento. Métrica primária: <b>{primary_metric}</b>.
Para grupos com diagnósticos ordinais, o score robusto usa QWK + within-1-rate + risco clínico ao invés de
balanced_accuracy/mcc/f1.</p>

{"".join(sections)}

</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

def run_comparison(
    results_dir: Path,
    output_dir: Path,
    primary_metric: str,
    task_filter: Optional[str],
    subgroup_filter: Optional[str],
    open_browser: bool,
) -> pd.DataFrame:
    experiments = discover_experiments(results_dir)
    if not experiments:
        print(f"Nenhum test_results.json encontrado em '{results_dir}'.", file=sys.stderr)
        return pd.DataFrame()

    if task_filter:
        experiments = [e for e in experiments if e.task == task_filter]
    if subgroup_filter:
        experiments = [e for e in experiments if e.subgroup == subgroup_filter]

    if not experiments:
        print("Nenhum experimento restou após aplicar os filtros.", file=sys.stderr)
        return pd.DataFrame()

    df = to_dataframe(experiments)

    robust_metrics_by_group = {}
    for group, sub in df.groupby("group"):
        robust_metrics_by_group[group] = ORDINAL_ROBUST_METRICS if sub["has_diagnostics"].any() else DEFAULT_ROBUST_METRICS

    df = compute_rankings(df, primary_metric=primary_metric, robust_metrics_by_group=robust_metrics_by_group)
    missing_by_group = detect_missing_models(df)
    verdicts = build_verdicts(df, primary_metric=primary_metric, missing_by_group=missing_by_group)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    group_figures: Dict[str, Dict[str, Path]] = {}
    for group, sub in df.groupby("group"):
        slug = _safe_slug(group)
        colors = _model_colors(list(sub["model"]))
        figs: Dict[str, Path] = {}
        n_models = sub["model"].nunique()
        has_diag = sub["has_diagnostics"].any()

        p = figures_dir / f"{slug}__bars_{primary_metric}.png"
        plot_metric_bars(sub, group, primary_metric, colors, p)
        figs[f"Barras — {primary_metric} (com erro da CV)"] = p

        if n_models >= 2:
            if has_diag:
                p = figures_dir / f"{slug}__bars_qwk.png"
                plot_metric_bars(sub, group, "qwk", colors, p)
                figs["Barras — QWK (concordância ponderada, ordinal)"] = p

                p = figures_dir / f"{slug}__bars_mae.png"
                plot_metric_bars(sub, group, "mae", colors, p, higher_is_better=False)
                figs["Barras — MAE (níveis de distância, menor é melhor)"] = p

                p = figures_dir / f"{slug}__bars_risk.png"
                plot_metric_bars(sub, group, "clinical_risk_score", colors, p, higher_is_better=False)
                figs["Barras — risco clínico (assimétrico, menor é melhor)"] = p

                p = figures_dir / f"{slug}__bias.png"
                plot_directional_bias(sub, group, colors, p)
                figs["Viés direcional (subestima vs. superestima)"] = p

                p = figures_dir / f"{slug}__confusion.png"
                plot_confusion_matrices(experiments, group, p)
                figs["Matriz de confusão por modelo"] = p

                p = figures_dir / f"{slug}__recall_heatmap.png"
                plot_per_class_recall_heatmap(experiments, group, p)
                figs["Recall por classe x modelo"] = p

                p = figures_dir / f"{slug}__signed_error_heatmap.png"
                plot_signed_error_by_true_class_heatmap(experiments, group, p)
                figs["Direção do erro por classe verdadeira x modelo"] = p

                p = figures_dir / f"{slug}__tradeoff.png"
                plot_tradeoff_scatter(sub, group, "qwk", "clinical_risk_score", colors, p)
                figs["Trade-off — QWK vs. risco clínico"] = p
            else:
                p = figures_dir / f"{slug}__bars_balanced_accuracy.png"
                plot_metric_bars(sub, group, "balanced_accuracy", colors, p)
                figs["Barras — balanced_accuracy"] = p

                p = figures_dir / f"{slug}__bars_mcc.png"
                plot_metric_bars(sub, group, "mcc", colors, p)
                figs["Barras — MCC"] = p

                p = figures_dir / f"{slug}__tradeoff.png"
                plot_tradeoff_scatter(sub, group, primary_metric, "balanced_accuracy", colors, p)
                figs[f"Trade-off — {primary_metric} vs. balanced_accuracy"] = p

        group_figures[group] = figs

    export_df = df.copy()
    csv_path = output_dir / "comparison_table.csv"
    export_df.to_csv(csv_path, index=False)

    html_path = output_dir / "comparison_report.html"
    build_html_report(df, group_figures, verdicts, primary_metric, html_path)

    print(f"\nTabela completa salva em: {csv_path}")
    print(f"Relatório visual salvo em: {html_path}")
    print("\nVeredito por grupo (tarefa / subgrupo):")
    for group, msgs in verdicts.items():
        print(f"\n[{group}]")
        for m in msgs:
            print(f"  - {m}")

    if open_browser:
        webbrowser.open(html_path.resolve().as_uri())

    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compara modelos usando os test_results.json de cada experimento.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model_comparison"))
    parser.add_argument("--primary-metric", type=str, default="roc_auc")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--subgroup", type=str, default=None)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--rerun-missing", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.rerun_missing:
        rerun_missing(args.results_dir)

    run_comparison(
        results_dir=args.results_dir, output_dir=args.output_dir, primary_metric=args.primary_metric,
        task_filter=args.task, subgroup_filter=args.subgroup, open_browser=args.open_browser,
    )


if __name__ == "__main__":
    main()