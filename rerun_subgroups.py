"""
rerun_subgroups.py

Roda todos os modelos disponiveis para os subgrupos "male" e "female" da
tarefa binaria (ou troque TASK abaixo se quiser fazer o mesmo para
"severity"). Ate agora so o random_forest tinha rodado nesses subgrupos -
isso completa os outros 5, pra dar pra comparar de verdade (o script de
comparacao trata grupo com 1 modelo so como "sem comparacao real ainda").

Coloque este arquivo na RAIZ do projeto (mesmo nivel de pipeline/, models/,
results/) e rode com:

    python rerun_subgroups.py
"""

from pipeline.run_model import run_model

TASK = "binary"          # troque para "severity" se quiser rodar por subgrupo lá tambem
SUBGROUPS = ["male", "female"]

# (model_name, dataset) - dataset e o que cada modelo espera (ver models/*.py)
MODELS = [
    ("logistic", "linear"),
    ("random_forest", "tree"),
    ("svm", "linear"),
    ("xgboost", "tree"),
    ("lightgbm", "tree"),
    ("catboost", "tree"),
]

if __name__ == "__main__":
    for subgroup in SUBGROUPS:
        for model_name, dataset in MODELS:
            print(f"\n=== {model_name} | dataset={dataset} | task={TASK} | subgroup={subgroup} ===")
            run_model(model_name=model_name, dataset=dataset, task=TASK, subgroup=subgroup)