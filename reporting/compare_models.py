"""
reporting/compare_models.py
============================

Compara, entre si, os modelos já treinados pelo pipeline (`pipeline.run_model`),
usando os `test_results.json` gravados em `results/{model}__{dataset}__{task}__{subgroup}/`.

Cada `test_results.json` tem o formato:

    {
      "model": "logistic",
      "dataset": "linear",
      "task": "severity",
      "subgroup": "all",
      "best_hyperparameters": {...},
      "n_train_total": 736,
      "n_test": 184,
      "metrics": {
        "accuracy": ..., "balanced_accuracy": ..., "precision": ..., "recall": ...,
        "f1": ..., "f2": ..., "roc_auc": ..., "mcc": ...
      },
      "timestamp": "..."
    }

Este script:
  1. varre `results/` e carrega todos os `test_results.json` encontrados;
  2. quando existe um `summary.json` irmão (mesma pasta), recupera o desvio-padrão
     da validação cruzada (5x5 CV) da métrica vencedora, para desenhar barras de
     erro nos gráficos — um único ponto de teste, sozinho, não diz nada sobre
     estabilidade;
  3. agrupa os experimentos por (task, subgroup) — é essa a unidade de comparação
     "justa": modelos diferentes podem exigir datasets diferentes (`tree` vs
     `linear`), mas se atacam a mesma tarefa e o mesmo subgrupo, são comparáveis;
  4. calcula, por grupo, um ranking pela métrica primária (default: roc_auc) E um
     ranking "robusto" (default: média de balanced_accuracy, mcc e f1), sinalizando
     quando os dois rankings discordam sobre qual é o "melhor" modelo — isso é
     especialmente relevante em tarefas desbalanceadas (ex.: severity);
  5. gera visualizações (PNG) por grupo: barras por métrica (com erro da CV),
     grade de métricas lado a lado, scatter de trade-off (métrica primária x
     métrica robusta), radar normalizado e heatmap modelos x métricas;
  6. gera um relatório HTML único, autocontido, com tabelas + gráficos embutidos;
  7. opcionalmente (--rerun-missing) identifica experimentos que têm summary.json
     mas não têm test_results.json (grid search rodou, mas o retreino final /
     avaliação em teste não) e oferece re-executar `pipeline.run_model.run_model`
     para completá-los.

Onde colocar este arquivo no seu projeto
-----------------------------------------
Coloque-o como um novo módulo, no mesmo nível de `models/`, `metrics/`,
`pipeline/`, `persistence/` e `training/`:

    seu_projeto/
    ├── models/
    ├── metrics/
    ├── persistence/
    ├── pipeline/
    ├── training/
    ├── results/                      <- já existe, gerado pelo pipeline
    └── reporting/
        ├── __init__.py               <- crie um arquivo vazio
        └── compare_models.py         <- este arquivo

Ele só lê arquivos em `results/` — não importa nada do resto do projeto, EXCETO
se você usar a flag `--rerun-missing`, que importa `pipeline.run_model.run_model`
(por isso, para usar essa flag, rode a partir da raiz do projeto, com
`python -m reporting.compare_models --rerun-missing`).

Uso
---
    # comparação simples, tudo com default (results/ -> reports/model_comparison/)
    python -m reporting.compare_models

    # só a tarefa severity, métrica primária customizada
    python -m reporting.compare_models --task severity --primary-metric roc_auc

    # apontando para outro diretório de resultados/saída
    python -m reporting.compare_models --results-dir meus_resultados --output-dir meus_relatorios

    # completar experimentos que faltam test_results.json (roda o pipeline de novo)
    python -m reporting.compare_models --rerun-missing
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # não depende de display; salva direto em PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuração de métricas
# ---------------------------------------------------------------------------

# Todas as métricas que existem em `metrics` dentro de test_results.json.
ALL_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "f2", "roc_auc", "mcc"]

# accuracy é enganosa sob desbalanceamento (ver relatório técnico da tarefa
# severity) -> não entra nas métricas "robustas" usadas para ranking de equidade.
DEFAULT_ROBUST_METRICS = ["balanced_accuracy", "mcc", "f1"]

# mcc vai de -1 a 1; todas as outras vão de 0 a 1. Para poder normalizar/plotar
# junto (radar, heatmap, score composto), reescalamos mcc para [0, 1].
def _rescale_mcc(value: Optional[float]) -> Optional[float]:
    if value is None:
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

    @property
    def experiment_key(self) -> str:
        return f"{self.model}__{self.dataset}__{self.task}__{self.subgroup}"

    @property
    def group_key(self) -> str:
        # unidade de comparação "justa": mesma tarefa + mesmo subgrupo,
        # independentemente do dataset exigido pelo modelo (tree/linear).
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
    """Carrega test_results.json de uma pasta de experimento; se existir
    summary.json irmão, tenta anexar o desvio-padrão (CV) da combinação
    vencedora, para servir de barra de erro nos gráficos."""
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
            same_combo = _hyperparams_match(
                test.get("best_hyperparameters", {}), best_cfg.get("hyperparameters", {})
            )
            if same_combo:
                cv_n_runs = best_cfg.get("n_runs")
                for m, stats in (best_cfg.get("metrics") or {}).items():
                    if isinstance(stats, dict):
                        cv_std[m] = stats.get("std")
            else:
                print(
                    f"[aviso] {summary_path}: best_configuration não bate com "
                    f"best_hyperparameters de {test_path}; CV std não será usado.",
                    file=sys.stderr,
                )

    return ExperimentResult(
        model=test["model"],
        dataset=test["dataset"],
        task=test["task"],
        subgroup=test["subgroup"],
        best_hyperparameters=test.get("best_hyperparameters", {}),
        n_train_total=test.get("n_train_total"),
        n_test=test.get("n_test"),
        metrics={m: test["metrics"].get(m) for m in ALL_METRICS},
        timestamp=test.get("timestamp"),
        source_dir=dir_path,
        cv_std=cv_std,
        cv_n_runs=cv_n_runs,
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
    """Pastas que têm summary.json (grid search + CV concluídos) mas não têm
    test_results.json (retreino final / avaliação em teste pendente)."""
    incomplete = []
    if not results_dir.exists():
        return incomplete
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        has_summary = (child / "summary.json").exists()
        has_test = (child / "test_results.json").exists()
        if has_summary and not has_test:
            incomplete.append(child)
    return incomplete


def rerun_missing(results_dir: Path) -> None:
    """Tenta completar experimentos incompletos chamando pipeline.run_model.
    Requer rodar a partir da raiz do projeto (para os imports abaixo funcionarem)."""
    incomplete = find_incomplete_experiments(results_dir)
    if not incomplete:
        print("Nenhum experimento incompleto encontrado (todos têm test_results.json).")
        return

    try:
        from pipeline.run_model import run_model  # import tardio e opcional
    except ImportError:
        print(
            "[erro] não consegui importar pipeline.run_model. Rode este comando a partir "
            "da raiz do projeto, ex.: `python -m reporting.compare_models --rerun-missing`.",
            file=sys.stderr,
        )
        return

    for dir_path in incomplete:
        # nome da pasta: {model}__{dataset}__{task}__{subgroup}
        parts = dir_path.name.split("__")
        if len(parts) != 4:
            print(f"[aviso] não consegui interpretar o nome da pasta {dir_path.name}; pulando.")
            continue
        model_name, dataset, task, subgroup = parts
        print(f"Completando experimento: {dir_path.name} ...")
        run_model(model_name=model_name, dataset=dataset, task=task, subgroup=subgroup)
    print("Concluído. Rode o script novamente sem --rerun-missing para gerar a comparação.")


# ---------------------------------------------------------------------------
# Montagem da tabela de comparação
# ---------------------------------------------------------------------------

def to_dataframe(experiments: List[ExperimentResult]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        row = {
            "model": exp.model,
            "dataset": exp.dataset,
            "task": exp.task,
            "subgroup": exp.subgroup,
            "group": exp.group_key,
            "n_train_total": exp.n_train_total,
            "n_test": exp.n_test,
            "timestamp": exp.timestamp,
            "hyperparameters": json.dumps(exp.best_hyperparameters, sort_keys=True),
            "cv_n_runs": exp.cv_n_runs,
            "source_dir": str(exp.source_dir),
        }
        for m in ALL_METRICS:
            row[m] = exp.metrics.get(m)
            row[f"{m}_cv_std"] = exp.cv_std.get(m)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["mcc_scaled"] = df["mcc"].apply(_rescale_mcc)
    return df.sort_values(["group", "model"]).reset_index(drop=True)


def compute_rankings(
    df: pd.DataFrame,
    primary_metric: str,
    robust_metrics: List[str],
) -> pd.DataFrame:
    """Adiciona, por grupo (task/subgroup): rank pela métrica primária, um
    score robusto (média min-max normalizada das métricas robustas) e o rank
    correspondente. Retorna uma cópia do df com essas colunas."""
    df = df.copy()
    df["primary_rank"] = np.nan
    df["robust_score"] = np.nan
    df["robust_rank"] = np.nan

    for group, idx in df.groupby("group").groups.items():
        sub = df.loc[idx]

        # rank pela métrica primária (maior é melhor); NaN vai para o fim
        primary_vals = sub[primary_metric]
        df.loc[idx, "primary_rank"] = primary_vals.rank(ascending=False, na_option="bottom")

        # score robusto: média das métricas robustas normalizadas (min-max) no grupo
        norm_cols = []
        for m in robust_metrics:
            col = sub[m]
            lo, hi = col.min(), col.max()
            norm_col_name = f"_norm_{m}"
            if pd.isna(lo) or pd.isna(hi) or hi == lo:
                df.loc[idx, norm_col_name] = 0.5  # sem variação -> neutro
            else:
                df.loc[idx, norm_col_name] = (col - lo) / (hi - lo)
            norm_cols.append(norm_col_name)

        df.loc[idx, "robust_score"] = df.loc[idx, norm_cols].mean(axis=1)
        df.loc[idx, "robust_rank"] = df.loc[idx, "robust_score"].rank(ascending=False, na_option="bottom")

    # limpa colunas auxiliares de normalização
    df = df.drop(columns=[c for c in df.columns if c.startswith("_norm_")])
    return df


def summarize_disagreements(df: pd.DataFrame, primary_metric: str) -> List[str]:
    """Para cada grupo, verifica se o vencedor pela métrica primária é o mesmo
    vencedor pelo score robusto. Retorna mensagens legíveis sobre discordâncias."""
    messages = []
    for group, sub in df.groupby("group"):
        best_primary = sub.loc[sub["primary_rank"].idxmin()]
        best_robust = sub.loc[sub["robust_rank"].idxmin()]
        if best_primary["model"] != best_robust["model"]:
            messages.append(
                f"[{group}] métrica primária ({primary_metric}) escolheria "
                f"'{best_primary['model']}' ({best_primary[primary_metric]:.4f}), mas o "
                f"score robusto (equidade entre classes) escolheria "
                f"'{best_robust['model']}' (robust_score={best_robust['robust_score']:.3f}). "
                f"Avalie manualmente qual critério importa mais para o caso de uso."
            )
        else:
            messages.append(
                f"[{group}] métrica primária e score robusto concordam: '{best_primary['model']}' "
                f"é o melhor pelos dois critérios."
            )
    return messages


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
})

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def _model_colors(models: List[str]) -> Dict[str, str]:
    return {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(sorted(models))}


def _safe_slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_")


def plot_metric_bars(sub: pd.DataFrame, group: str, metric: str, colors: Dict[str, str], out_path: Path) -> None:
    sub = sub.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(max(4.5, 0.9 * len(sub) + 2), 4))
    yerr = sub[f"{metric}_cv_std"].fillna(0.0) if f"{metric}_cv_std" in sub.columns else None
    bars = ax.bar(sub["model"], sub[metric], yerr=yerr, capsize=4,
                   color=[colors[m] for m in sub["model"]])
    for bar, val in zip(bars, sub[metric]):
        ax.annotate(f"{val:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} por modelo — {group}\n(barra de erro = desvio-padrão da CV, quando disponível)")
    ax.set_ylim(0, max(1.0, float((sub[metric] + (yerr if yerr is not None else 0)).max()) + 0.08))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_grouped_metrics(sub: pd.DataFrame, group: str, metrics_list: List[str], colors: Dict[str, str], out_path: Path) -> None:
    models = list(sub["model"])
    n_metrics = len(metrics_list)
    x = np.arange(n_metrics)
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * n_metrics), 5))
    for i, (_, row) in enumerate(sub.iterrows()):
        vals = [row[m] if pd.notna(row[m]) else 0 for m in metrics_list]
        offsets = x + (i - (len(models) - 1) / 2) * width
        ax.bar(offsets, vals, width, label=row["model"], color=colors[row["model"]])

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics_list, rotation=0)
    ax.set_ylabel("valor")
    ax.set_title(f"Comparação de métricas — {group}")
    ax.legend(frameon=False, ncol=min(len(models), 4), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_tradeoff_scatter(sub: pd.DataFrame, group: str, x_metric: str, y_metric: str, colors: Dict[str, str], out_path: Path) -> None:
    sub = sub.dropna(subset=[x_metric, y_metric])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.8))
    for _, row in sub.iterrows():
        ax.scatter(row[x_metric], row[y_metric], s=140, color=colors[row["model"]],
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(row["model"], (row[x_metric], row[y_metric]), textcoords="offset points",
                    xytext=(7, 5), fontsize=9)
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f"Trade-off: {x_metric} vs. {y_metric} — {group}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_radar(sub: pd.DataFrame, group: str, metrics_list: List[str], colors: Dict[str, str], out_path: Path) -> None:
    labels = metrics_list
    n = len(labels)
    angles = [i / n * 2 * np.pi for i in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for _, row in sub.iterrows():
        values = []
        for m in labels:
            v = row["mcc_scaled"] if m == "mcc" else row[m]
            values.append(v if pd.notna(v) else 0)
        values += values[:1]
        ax.plot(angles, values, label=row["model"], color=colors[row["model"]], linewidth=2)
        ax.fill(angles, values, color=colors[row["model"]], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m if m != "mcc" else "mcc\n(reescalado 0-1)" for m in labels])
    ax.set_ylim(0, 1)
    ax.set_title(f"Perfil normalizado de métricas — {group}", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_heatmap(sub: pd.DataFrame, group: str, metrics_list: List[str], out_path: Path) -> None:
    matrix = sub.set_index("model")[metrics_list].astype(float)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(metrics_list) + 2), max(3, 0.6 * len(matrix) + 1.5)))
    im = ax.imshow(matrix.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics_list)))
    ax.set_xticklabels(metrics_list, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            txt = "-" if pd.isna(val) else f"{val:.2f}"
            color = "white" if (pd.notna(val) and val < 0.6) else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="valor (0–1; mcc reescalado)")
    ax.set_title(f"Mapa de calor modelo x métrica — {group}")
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
    disagreement_msgs: List[str],
    primary_metric: str,
    robust_metrics: List[str],
    out_html: Path,
) -> None:
    display_cols = ["model", "dataset", "n_test", primary_metric, *robust_metrics,
                     "primary_rank", "robust_score", "robust_rank", "hyperparameters"]
    display_cols = [c for c in display_cols if c in df.columns]

    sections = []
    for group, sub in df.groupby("group"):
        sub_sorted = sub.sort_values("primary_rank")
        table_html = sub_sorted[display_cols].round(4).to_html(index=False, border=0, classes="tbl")

        figs_html = ""
        for fig_name, fig_path in group_figures.get(group, {}).items():
            if fig_path.exists():
                b64 = _img_to_base64(fig_path)
                figs_html += (
                    f'<div class="fig"><h4>{fig_name}</h4>'
                    f'<img src="data:image/png;base64,{b64}" /></div>'
                )

        sections.append(f"""
        <section>
          <h2>{group}</h2>
          {table_html}
          <div class="fig-grid">{figs_html}</div>
        </section>
        """)

    disagreements_html = "".join(f"<li>{m}</li>" for m in disagreement_msgs)

    html = f"""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8" />
<title>Comparação de modelos</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; color: #1a1a1a; }}
  h1 {{ margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-top: 0; }}
  table.tbl {{ border-collapse: collapse; width: 100%; margin: 14px 0 22px 0; font-size: 13.5px; }}
  table.tbl th, table.tbl td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
  table.tbl th {{ background: #f4f4f6; }}
  table.tbl tr:nth-child(1) td {{ background: #eef7ee; font-weight: 600; }}
  section {{ margin-bottom: 48px; padding-bottom: 24px; border-bottom: 1px solid #eee; }}
  .fig-grid {{ display: flex; flex-wrap: wrap; gap: 18px; }}
  .fig {{ flex: 1 1 420px; max-width: 620px; }}
  .fig img {{ width: 100%; height: auto; border: 1px solid #eee; border-radius: 6px; }}
  .fig h4 {{ margin: 6px 0; color: #444; font-weight: 600; }}
  .callout {{ background: #fff7e6; border: 1px solid #ffe1a8; border-radius: 8px; padding: 14px 18px; margin: 18px 0 32px 0; }}
  .callout h3 {{ margin-top: 0; }}
</style>
</head>
<body>
<h1>Comparação de modelos entre experimentos</h1>
<p class="subtitle">Baseado nos <code>test_results.json</code> de cada experimento. Métrica primária:
<b>{primary_metric}</b> · Métricas robustas (equidade entre classes): <b>{", ".join(robust_metrics)}</b>.
Linha destacada em cada tabela = melhor pela métrica primária.</p>

<div class="callout">
  <h3>Concordância entre critérios de seleção</h3>
  <ul>{disagreements_html}</ul>
</div>

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
    robust_metrics: List[str],
    task_filter: Optional[str],
    subgroup_filter: Optional[str],
    open_browser: bool,
) -> pd.DataFrame:
    experiments = discover_experiments(results_dir)
    if not experiments:
        print(f"Nenhum test_results.json encontrado em '{results_dir}'.", file=sys.stderr)
        return pd.DataFrame()

    df = to_dataframe(experiments)

    if task_filter:
        df = df[df["task"] == task_filter]
    if subgroup_filter:
        df = df[df["subgroup"] == subgroup_filter]

    if df.empty:
        print("Nenhum experimento restou após aplicar os filtros.", file=sys.stderr)
        return df

    df = compute_rankings(df, primary_metric=primary_metric, robust_metrics=robust_metrics)
    disagreements = summarize_disagreements(df, primary_metric=primary_metric)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    radar_metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]
    heatmap_metrics = [m if m != "mcc" else "mcc_scaled" for m in radar_metrics]

    group_figures: Dict[str, Dict[str, Path]] = {}
    for group, sub in df.groupby("group"):
        slug = _safe_slug(group)
        colors = _model_colors(list(sub["model"]))
        figs: Dict[str, Path] = {}

        p = figures_dir / f"{slug}__bars_{primary_metric}.png"
        plot_metric_bars(sub, group, primary_metric, colors, p)
        figs[f"Barras — {primary_metric} (com erro da CV)"] = p

        for rm in robust_metrics:
            p = figures_dir / f"{slug}__bars_{rm}.png"
            plot_metric_bars(sub, group, rm, colors, p)
            figs[f"Barras — {rm}"] = p

        p = figures_dir / f"{slug}__grouped.png"
        plot_grouped_metrics(sub, group, ALL_METRICS, colors, p)
        figs["Todas as métricas, lado a lado"] = p

        if len(robust_metrics) > 0:
            p = figures_dir / f"{slug}__tradeoff.png"
            plot_tradeoff_scatter(sub, group, primary_metric, robust_metrics[0], colors, p)
            figs[f"Trade-off — {primary_metric} vs. {robust_metrics[0]}"] = p

        p = figures_dir / f"{slug}__radar.png"
        plot_radar(sub, group, radar_metrics, colors, p)
        figs["Radar (perfil normalizado)"] = p

        p = figures_dir / f"{slug}__heatmap.png"
        plot_heatmap(sub, group, heatmap_metrics, p)
        figs["Heatmap modelo x métrica"] = p

        group_figures[group] = figs

    csv_path = output_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)

    html_path = output_dir / "comparison_report.html"
    build_html_report(df, group_figures, disagreements, primary_metric, robust_metrics, html_path)

    print(f"\nTabela completa salva em: {csv_path}")
    print(f"Relatório visual salvo em: {html_path}")
    print("\nResumo por grupo (tarefa / subgrupo):")
    for msg in disagreements:
        print(f"  - {msg}")

    if open_browser:
        webbrowser.open(html_path.resolve().as_uri())

    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara modelos treinados pelo pipeline usando os test_results.json de cada experimento."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                         help="Diretório onde estão as pastas {model}__{dataset}__{task}__{subgroup}/ (default: results)")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model_comparison"),
                         help="Diretório onde salvar tabelas, gráficos e o relatório HTML (default: reports/model_comparison)")
    parser.add_argument("--primary-metric", type=str, default="roc_auc",
                         help="Métrica usada como critério principal de seleção (default: roc_auc)")
    parser.add_argument("--robust-metrics", type=str, default=",".join(DEFAULT_ROBUST_METRICS),
                         help="Lista separada por vírgula de métricas robustas a desbalanceamento "
                              f"(default: {','.join(DEFAULT_ROBUST_METRICS)})")
    parser.add_argument("--task", type=str, default=None, help="Filtra por uma única task (ex.: severity)")
    parser.add_argument("--subgroup", type=str, default=None, help="Filtra por um único subgroup (ex.: all)")
    parser.add_argument("--open-browser", action="store_true", help="Abre o relatório HTML no navegador ao final")
    parser.add_argument("--rerun-missing", action="store_true",
                         help="Antes de comparar, tenta completar (via pipeline.run_model) experimentos que "
                              "têm summary.json mas não têm test_results.json")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    robust_metrics = [m.strip() for m in args.robust_metrics.split(",") if m.strip()]

    if args.rerun_missing:
        rerun_missing(args.results_dir)

    run_comparison(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        primary_metric=args.primary_metric,
        robust_metrics=robust_metrics,
        task_filter=args.task,
        subgroup_filter=args.subgroup,
        open_browser=args.open_browser,
    )


if __name__ == "__main__":
    main()