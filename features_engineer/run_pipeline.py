"""
Executa o pipeline completo de engenharia de features sobre o dataset
Heart Disease UCI e produz:

  - datasets processados (treino/teste) prontos para modelos de árvore
    e para modelos lineares, salvos em CSV
  - relatório de missing antes/depois
  - relatório de VIF e informação mútua para apoiar a seleção de features
  - um pequeno teste de sanidade validando que a imputação é refeita
    corretamente dentro de um fold de validação cruzada (anti-leakage)

Uso:
    python run_pipeline.py --input data/heart_disease_uci.csv --outdir out/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from feature_engineering_pipeline import (
    ALL_IMPUTED_COLS,
    FEStep,
    HeartDiseaseFeatureEngineeringPipeline,
    build_column_transformer,
    compute_mutual_information,
    compute_vif,
    missing_summary,
)


def run_cv_sanity_check(df: pd.DataFrame, n_splits: int = 5):
    """Roda o pipeline completo (FE + encoding + modelo) dentro de
    validação cruzada, com a imputação sendo refeita em cada fold —
    prova de que não há vazamento de informação treino<->validação."""
    full_pipeline = Pipeline([
        ("fe", FEStep(n_imputations_bc=1, random_state=42)),
        ("encode", build_column_transformer(model_family="linear")),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(
        full_pipeline, df, df["doenca"], cv=cv, scoring="roc_auc", n_jobs=1,
    )
    print(f"AUC (cross-validation, Regressão Logística, FE refeita em cada fold): "
          f"{scores.mean():.3f} +/- {scores.std():.3f}")
    print("(Este número é apenas um teste de sanidade do pipeline, não uma "
          "otimização de modelo — serve para confirmar que fit/transform "
          "não vazam informação do fold de validação para o de treino.)")
    return scores


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normaliza fbs/exang para string/objeto (evita dtype 'boolean' nullable
    # do pandas atrapalhar OrdinalEncoder/OneHotEncoder mais adiante)
    for col in ["fbs", "exang"]:
        if col in df.columns:
            df[col] = df[col].map({True: "True", False: "False"})
    df["doenca"] = (df["num"] > 0).astype(int)
    return df


def stratified_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Split treino/teste estratificado simultaneamente por doença e
    sexo (Seção 9, passo 2). Cai para estratificação simples por
    doença caso alguma combinação doença×sexo×centro seja rara demais."""
    strat_key = df["doenca"].astype(str) + "_" + df["sex"].astype(str) + "_" + df["dataset"].astype(str)
    counts = strat_key.value_counts()
    if (counts < 2).any():
        print("[aviso] combinação doença×sexo×centro com <2 exemplos; "
              "caindo para estratificação por doença×sexo.")
        strat_key = df["doenca"].astype(str) + "_" + df["sex"].astype(str)
        counts = strat_key.value_counts()
        if (counts < 2).any():
            print("[aviso] caindo para estratificação apenas por doença.")
            strat_key = df["doenca"]

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat_key
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def run(input_path: str, outdir: str, n_imputations_bc: int = 5, quick: bool = False, skip_cv_check: bool = False):
    os.makedirs(outdir, exist_ok=True)

    print("=" * 70)
    print("1) Carregando dados")
    print("=" * 70)
    df = load_data(input_path)
    print(f"Shape original: {df.shape}")

    print("\nResumo de missing (dados brutos, zeros disfarçados ainda não corrigidos):")
    print(missing_summary(df))

    print("\n" + "=" * 70)
    print("2) Split treino/teste estratificado (doença x sexo x centro)")
    print("=" * 70)
    train_df, test_df = stratified_split(df)
    print(f"Treino: {train_df.shape} | Teste: {test_df.shape}")

    print("\n" + "=" * 70)
    print("3) Fit do pipeline de FE APENAS no treino (anti-leakage)")
    print("=" * 70)
    n_imp = 1 if quick else n_imputations_bc
    fe_pipeline = HeartDiseaseFeatureEngineeringPipeline(
        n_imputations_bc=n_imp, random_state=42, winsorize=True
    )
    fe_pipeline.fit(train_df)

    train_fe = fe_pipeline.transform(train_df)
    test_fe = fe_pipeline.transform(test_df)

    print("\nResumo de missing pós-pipeline (treino) — deve ser ~0 nas colunas imputadas:")
    cols_check = ALL_IMPUTED_COLS
    print(missing_summary(train_fe[cols_check]))

    print("\nAmostra de features derivadas criadas (treino):")
    derived = [
        "age_bin", "chol_category", "bp_category", "log_oldpeak", "ca_present",
        "sex_x_age", "sex_x_thalch", "sex_x_oldpeak", "n_missing_row",
        "ca_missing", "thal_missing", "slope_missing",
    ]
    print(train_fe[derived].head(5).to_string())

    print("\n" + "=" * 70)
    print("4) Construindo matrizes finais (versão 'tree' e versão 'linear')")
    print("=" * 70)
    ct_tree = build_column_transformer(model_family="tree", include_dataset_as_feature=False)
    ct_linear = build_column_transformer(model_family="linear", include_dataset_as_feature=False)

    X_train_tree = ct_tree.fit_transform(train_fe)
    X_test_tree = ct_tree.transform(test_fe)
    feat_names_tree = ct_tree.get_feature_names_out()

    X_train_linear = ct_linear.fit_transform(train_fe)
    X_test_linear = ct_linear.transform(test_fe)
    feat_names_linear = ct_linear.get_feature_names_out()

    print(f"Matriz 'tree'   -> treino {X_train_tree.shape}, teste {X_test_tree.shape}")
    print(f"Matriz 'linear' -> treino {X_train_linear.shape}, teste {X_test_linear.shape}")

    # -----------------------------------------------------------------
    # 5) Relatórios de seleção de features (exploratório)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5) Seleção de features — VIF e Informação Mútua (treino, versão numérica)")
    print("=" * 70)
    numeric_for_vif = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca",
                        "sex_x_age", "sex_x_thalch", "sex_x_oldpeak"]
    try:
        vif_df = compute_vif(train_fe, numeric_for_vif)
        print("\nVIF (Variance Inflation Factor):")
        print(vif_df.to_string(index=False))
    except Exception as e:
        print(f"[aviso] VIF não pôde ser calculado: {e}")

    mi_feature_cols = [
        "age", "trestbps", "chol", "thalch", "oldpeak", "ca", "sex", "cp",
        "fbs", "restecg", "exang", "slope", "thal",
        "ca_missing", "thal_missing", "slope_missing",
    ]
    mi_df = compute_mutual_information(train_fe, mi_feature_cols, "doenca")
    print("\nInformação Mútua com o target binário 'doenca':")
    print(mi_df.to_string(index=False))

    # -----------------------------------------------------------------
    # 6) Teste de sanidade anti-leakage: imputação refeita em cada fold de CV
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6) Sanidade: pipeline completo dentro de cross-validation (sem vazamento)")
    print("=" * 70)

    if skip_cv_check:
        print("(pulado via --skip-cv-check; rode sem essa flag para validar "
              "anti-leakage via cross-validation completa — mais lento)")
    else:
        run_cv_sanity_check(df)

    # -----------------------------------------------------------------
    # 7) Salvando artefatos
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("7) Salvando artefatos em: " + outdir)
    print("=" * 70)

    train_fe.to_csv(os.path.join(outdir, "train_features_ricas.csv"), index=False)
    test_fe.to_csv(os.path.join(outdir, "test_features_ricas.csv"), index=False)

    pd.DataFrame(X_train_tree, columns=feat_names_tree).assign(
        doenca=train_fe["doenca"].values, num=train_fe["num"].values
    ).to_csv(os.path.join(outdir, "train_matrix_tree.csv"), index=False)
    pd.DataFrame(X_test_tree, columns=feat_names_tree).assign(
        doenca=test_fe["doenca"].values, num=test_fe["num"].values
    ).to_csv(os.path.join(outdir, "test_matrix_tree.csv"), index=False)

    pd.DataFrame(X_train_linear, columns=feat_names_linear).assign(
        doenca=train_fe["doenca"].values, num=train_fe["num"].values
    ).to_csv(os.path.join(outdir, "train_matrix_linear.csv"), index=False)
    pd.DataFrame(X_test_linear, columns=feat_names_linear).assign(
        doenca=test_fe["doenca"].values, num=test_fe["num"].values
    ).to_csv(os.path.join(outdir, "test_matrix_linear.csv"), index=False)

    missing_summary(train_fe[cols_check]).to_csv(os.path.join(outdir, "missing_report_pos_imputacao.csv"))
    mi_df.to_csv(os.path.join(outdir, "mutual_information_report.csv"), index=False)
    try:
        vif_df.to_csv(os.path.join(outdir, "vif_report.csv"), index=False)
    except NameError:
        pass

    print("Arquivos gerados:")
    for f in sorted(os.listdir(outdir)):
        print(" -", f)

    print("\nConcluído.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/heart_disease_uci.csv")
    parser.add_argument("--outdir", default="out")
    parser.add_argument("--n-imputations-bc", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="usa m=1 imputação para rodar mais rápido")
    parser.add_argument("--skip-cv-check", action="store_true",
                         help="pula o teste de sanidade de cross-validation (mais lento)")
    args = parser.parse_args()

    run(args.input, args.outdir, args.n_imputations_bc, args.quick, args.skip_cv_check)
