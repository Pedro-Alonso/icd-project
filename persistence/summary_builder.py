"""
Construcao do resumo final (summary.json).

Le todos os run_*.json de um experimento, agrupa por combinacao de
hiperparametros (combo_id), calcula media e desvio padrao de cada
metrica entre folds/repeticoes, rankeia as combinacoes pela metrica
"primaria" definida na config, e grava summary.json.
"""

import statistics
from collections import defaultdict
from typing import Any, Dict, List

from persistence.run_writer import load_all_runs, write_summary


def build_summary(
    model_name: str,
    dataset: str,
    task: str,
    subgroup: str,
    primary_metric: str,
) -> Dict[str, Any]:
    runs = load_all_runs(model_name, dataset, task, subgroup)
    if not runs:
        raise RuntimeError(
            f"Nenhum run encontrado para {model_name}/{dataset}/{task}/{subgroup}. "
            "Rode o grid search antes de gerar o summary."
        )

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["combo_id"]].append(run)

    ranking = []
    for combo_id, group_runs in grouped.items():
        metric_names = list(group_runs[0]["metrics"].keys())
        agg_metrics = {}
        for m in metric_names:
            values = [r["metrics"][m] for r in group_runs if r["metrics"].get(m) is not None]
            if values:
                agg_metrics[m] = {
                    "mean": statistics.fmean(values),
                    "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
            else:
                agg_metrics[m] = {"mean": None, "std": None, "n": 0}

        times = [r["execution_time_seconds"] for r in group_runs]

        ranking.append(
            {
                "combo_id": combo_id,
                "hyperparameters": group_runs[0]["hyperparameters"],
                "n_runs": len(group_runs),
                "metrics": agg_metrics,
                "mean_execution_time_seconds": statistics.fmean(times),
            }
        )

    def sort_key(entry):
        mean_val = entry["metrics"].get(primary_metric, {}).get("mean")
        # combinacoes sem a metrica primaria calculavel vao para o fim
        return (mean_val is None, -(mean_val or 0))

    ranking.sort(key=sort_key)

    summary = {
        "model": model_name,
        "dataset": dataset,
        "task": task,
        "subgroup": subgroup,
        "primary_metric": primary_metric,
        "n_combinations_evaluated": len(ranking),
        "total_runs": len(runs),
        "best_configuration": ranking[0] if ranking else None,
        "ranking": ranking,
    }

    write_summary(model_name, dataset, task, subgroup, summary)
    return summary
