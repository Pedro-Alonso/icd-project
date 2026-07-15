"""
Pipeline unico.

Este e o unico ponto de entrada para rodar um experimento completo:
carregar dados -> checkpoint -> grid search + CV -> summary -> retreino
final -> avaliacao no teste. Nao conhece nenhum algoritmo especifico -
so fala com plugins atraves do contrato definido em models/base.py.

Uso:
    from pipeline.run_model import run_model
    run_model(model_name="random_forest", dataset="tree", task="binary", subgroup="all")
"""

import datetime as dt
from typing import Any, Dict, Optional

from datasets.loader import load_dataset
from metrics.registry import available_metrics
from models.registry import get_model_plugin
from persistence.checkpoint import get_completed_keys, is_done
from persistence.run_writer import next_run_path, write_run_atomic, write_test_results
from persistence.summary_builder import build_summary
from training.final_fit import fit_final_model
from tuning.cv_strategy import build_cv_splits
from tuning.grid_search import combo_id, expand_param_grid
from utils.logging_config import get_logger
from utils.seeding import derive_seed, set_global_seed
from utils.timing import timer
from evaluation.cv_evaluator import evaluate_fold
from evaluation.test_evaluator import evaluate_on_test

logger = get_logger("pipeline.run_model")

DEFAULT_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "f2", "roc_auc", "mcc"]


def run_model(
    model_name: str,
    dataset: str,
    task: str = "binary",
    subgroup: str = "all",
    n_splits: int = 5,
    n_repeats: int = 5,
    master_seed: int = 42,
    metric_names: Optional[list] = None,
    primary_metric: str = "roc_auc",
    run_final_test: bool = True,
) -> Dict[str, Any]:
    """Executa (ou continua) um experimento completo para um modelo.

    Idempotente: pode ser chamado repetidamente com os mesmos argumentos -
    combinacoes ja executadas (presentes em results/) sao puladas.
    """
    metric_names = metric_names or DEFAULT_METRICS
    set_global_seed(master_seed)

    logger.info(
        f"=== run_model: model={model_name} dataset={dataset} task={task} "
        f"subgroup={subgroup} n_splits={n_splits} n_repeats={n_repeats} ==="
    )

    plugin = get_model_plugin(model_name)

    if plugin.expected_dataset != dataset:
        logger.warning(
            f"Modelo '{model_name}' normalmente espera dataset '{plugin.expected_dataset}', "
            f"mas foi chamado com dataset='{dataset}'. Prosseguindo mesmo assim."
        )

    data = load_dataset(dataset_name=dataset, task=task, subgroup=subgroup)
    X_train, y_train = data.X_train, data.y_train

    param_combinations = expand_param_grid(plugin.parameter_grid())
    logger.info(f"{len(param_combinations)} combinacoes de hiperparametros a avaliar.")

    completed_keys = get_completed_keys(model_name, dataset, task, subgroup)
    logger.info(f"{len(completed_keys)} runs ja existentes (checkpoint).")

    n_executed = 0
    n_skipped = 0

    for params in param_combinations:
        cid = combo_id(params)

        for repeat_idx, fold_idx, train_idx, val_idx in build_cv_splits(
            X_train, y_train, n_splits=n_splits, n_repeats=n_repeats, master_seed=master_seed
        ):
            if is_done(completed_keys, cid, repeat_idx, fold_idx):
                n_skipped += 1
                continue

            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            fold_seed = derive_seed(master_seed, model_name, cid, f"repeat={repeat_idx}", f"fold={fold_idx}")

            with timer() as t:
                metric_values = evaluate_fold(
                    plugin, params, X_tr, y_tr, X_val, y_val, metric_names, random_state=fold_seed
                )

            payload = {
                "model": model_name,
                "dataset": dataset,
                "task": task,
                "subgroup": subgroup,
                "combo_id": cid,
                "hyperparameters": params,
                "repeat_idx": repeat_idx,
                "fold_idx": fold_idx,
                "seed": fold_seed,
                "n_train_fold": len(train_idx),
                "n_val_fold": len(val_idx),
                "metrics": metric_values,
                "execution_time_seconds": t.elapsed_seconds,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }

            run_path = next_run_path(model_name, dataset, task, subgroup)
            write_run_atomic(run_path, payload)
            n_executed += 1

    logger.info(f"Grid search concluido: {n_executed} runs novos, {n_skipped} pulados (ja existiam).")

    summary = build_summary(model_name, dataset, task, subgroup, primary_metric=primary_metric)
    logger.info(f"Melhor combinacao ({primary_metric}): {summary['best_configuration']['combo_id']}")

    result = {"summary": summary, "test_results": None}

    if run_final_test:
        best_params = summary["best_configuration"]["hyperparameters"]
        final_estimator = fit_final_model(plugin, best_params, X_train, y_train, master_seed=master_seed)

        test_metrics = evaluate_on_test(
            plugin, final_estimator, data.X_test, data.y_test, y_train_reference=y_train, metric_names=metric_names
        )

        test_payload = {
            "model": model_name,
            "dataset": dataset,
            "task": task,
            "subgroup": subgroup,
            "best_hyperparameters": best_params,
            "n_train_total": len(X_train),
            "n_test": len(data.X_test),
            "metrics": test_metrics,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        write_test_results(model_name, dataset, task, subgroup, test_payload)
        logger.info(f"Avaliacao final no teste concluida: {test_metrics}")
        result["test_results"] = test_payload

    return result
