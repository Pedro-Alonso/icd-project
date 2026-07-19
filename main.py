"""
Ponto de entrada de linha de comando.

Uso:
    python main.py --experiment configs/experiments/random_forest_tree_binary.yaml
    python main.py --experiment configs/experiments/*.yaml   # roda varios (glob expandido pelo shell)
    python main.py --model xgboost --dataset tree --task binary --subgroup all   # sem arquivo de config

Le configs/base.yaml para os defaults e o YAML do experimento para os
parametros especificos, e chama pipeline.run_model.run_model().
"""

import argparse
from pathlib import Path

import yaml

from pipeline.run_model import run_model

BASE_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "base.yaml"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_from_experiment_file(experiment_path: Path) -> None:
    base_cfg = load_yaml(BASE_CONFIG_PATH)
    exp_cfg = load_yaml(experiment_path)

    cv_cfg = base_cfg.get("cross_validation", {})

    run_model(
        model_name=exp_cfg["model_name"],
        dataset=exp_cfg["dataset"],
        task=exp_cfg.get("task", "binary"),
        subgroup=exp_cfg.get("subgroup", "all"),
        n_splits=exp_cfg.get("n_splits", cv_cfg.get("n_splits", 5)),
        n_repeats=exp_cfg.get("n_repeats", cv_cfg.get("n_repeats", 5)),
        master_seed=exp_cfg.get("master_seed", base_cfg.get("master_seed", 42)),
        metric_names=exp_cfg.get("metrics", base_cfg.get("metrics")),
        primary_metric=exp_cfg.get("primary_metric", base_cfg.get("primary_metric", "roc_auc")),
        run_final_test=exp_cfg.get("run_final_test", base_cfg.get("run_final_test", True)),
    )


def main():
    parser = argparse.ArgumentParser(description="Pipeline de treino/tuning/avaliacao de modelos.")
    parser.add_argument("--experiment", type=str, nargs="+", help="Caminho(s) para YAML(s) de experimento.")
    parser.add_argument("--model", type=str, help="Nome do modelo (alternativa a --experiment).")
    parser.add_argument("--dataset", type=str, choices=["linear", "tree"])
    parser.add_argument("--task", type=str, choices=["binary", "severity"], default="binary")
    parser.add_argument("--subgroup", type=str, choices=["all", "male", "female"], default="all")
    args = parser.parse_args()

    if args.experiment:
        for exp_path in args.experiment:
            run_from_experiment_file(Path(exp_path))
    elif args.model and args.dataset:
        run_model(model_name=args.model, dataset=args.dataset, task=args.task, subgroup=args.subgroup)
    else:
        parser.error("Forneca --experiment <arquivo.yaml> ou --model + --dataset.")

def main2():
    from pipeline.run_model import run_model
        for m in ["logistic", "random_forest", "svm", "xgboost", "lightgbm", "catboost"]:
            run_model(model_name=m, dataset=<dataset esperado do modelo>, task="severity", subgroup="all")
            

if __name__ == "__main__":
    main()
