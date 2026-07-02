"""
Pipeline de Engenharia de Features — Heart Disease UCI
========================================================

Implementa, em código, o plano metodológico descrito no documento
`plano_engenharia_features_heart_disease.md`:

    1. Correção de missing disfarçado (zeros clinicamente impossíveis -> NaN)
    2. Indicadores binários de missingness (sinal MNAR)
    3. Imputação em 3 sub-pipelines (A: leve/moderado quase-MAR;
       B: moderado-alto com componente estrutural; C: MNAR extremo
       estrutural por centro), sempre condicionada por `dataset`
    4. Engenharia de features derivadas (bins clínicos, interações,
       transformação log)
    5. Tratamento de outliers via winsorização (não remoção de pacientes)
    6. Encoding categórico + normalização condicionada ao tipo de modelo
       (RobustScaler para modelos lineares/distância; nenhuma
       normalização para modelos de árvore)
    7. Utilitários de seleção de features (correlação, VIF, informação
       mútua) para uso exploratório

Todos os transformers seguem a API scikit-learn (fit/transform), o que
permite usá-los dentro de um `sklearn.pipeline.Pipeline` e, portanto,
dentro de validação cruzada sem vazamento de dados: o `fit` aprende
parâmetros (imputadores, limites de winsorização, escalonador) apenas
no fold de treino; o `transform` aplica esses mesmos parâmetros ao
fold de validação/teste.

Autor: plano gerado a partir da EDA `03_eda_final.ipynb`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Definições de colunas (a partir do dicionário de variáveis da EDA)
# ---------------------------------------------------------------------------

TARGET_MULTICLASS = "num"
TARGET_BINARY = "doenca"

NUM_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
BIN_COLS = ["sex", "fbs", "exang"]          # binárias (após normalização de tipo)
CAT_COLS = ["cp", "restecg", "slope", "thal"]  # categóricas nominais
CENTER_COL = "dataset"
ID_COL = "id"

# Colunas que tiveram zeros clinicamente impossíveis identificados na EDA
ZERO_AS_MISSING_COLS = ["chol", "trestbps"]

# Agrupamento das variáveis por mecanismo de missing (Seção 3 do plano)
PIPELINE_A_COLS = ["trestbps", "thalch", "exang", "oldpeak", "fbs", "restecg"]   # leve/moderado, quase-MAR
PIPELINE_B_COLS = ["chol", "slope"]                                              # moderado-alto, componente estrutural
PIPELINE_C_COLS = ["ca", "thal"]                                                 # MNAR extremo estrutural por centro

ALL_IMPUTED_COLS = PIPELINE_A_COLS + PIPELINE_B_COLS + PIPELINE_C_COLS

# Limites de domínio fisiológico plausível, usados apenas para marcar/nulificar
# erros de digitação evidentes (Seção 5 do plano) — não usados para descartar
# "outliers extremos mas plausíveis" (ex. chol alto, oldpeak alto).
DOMAIN_BOUNDS = {
    # variável: (min plausível, max plausível)
    "trestbps": (50, 260),     # pressão arterial em repouso (mmHg)
    "chol": (50, 700),         # colesterol sérico (mg/dL)
    "thalch": (60, 220),       # frequência cardíaca máxima
    "oldpeak": (-3, 8),        # depressão/elevação de ST (valores negativos são fisiologicamente documentados neste dataset)
    "ca": (0, 3),              # número de vasos (definição do próprio exame)
    "age": (0, 120),
}


# ---------------------------------------------------------------------------
# Etapa 1 — Correção de missing disfarçado + validação de domínio
# ---------------------------------------------------------------------------

class DomainCleaner(BaseEstimator, TransformerMixin):
    """Converte valores clinicamente impossíveis (zeros disfarçados e
    valores fora do domínio fisiológico plausível) em NaN.

    Etapa estateless (não aprende nada do treino) — pode ser aplicada
    antes do split, pois não introduz vazamento (é apenas uma correção
    de qualidade de dado, não uma estimativa aprendida).
    """

    def __init__(self, zero_as_missing_cols=None, domain_bounds=None):
        self.zero_as_missing_cols = zero_as_missing_cols or ZERO_AS_MISSING_COLS
        self.domain_bounds = domain_bounds or DOMAIN_BOUNDS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        report = {}

        # 1) zeros clinicamente impossíveis -> NaN
        for col in self.zero_as_missing_cols:
            if col in X.columns:
                n_zeros = int((X[col] == 0).sum())
                X.loc[X[col] == 0, col] = np.nan
                report[f"{col}_zeros_converted"] = n_zeros

        # 2) valores fora do domínio fisiológico plausível -> NaN
        # (erro de digitação; a linha é preservada, só o valor é nulificado)
        for col, (lo, hi) in self.domain_bounds.items():
            if col in X.columns:
                mask = X[col].notna() & ((X[col] < lo) | (X[col] > hi))
                n_out = int(mask.sum())
                if n_out > 0:
                    X.loc[mask, col] = np.nan
                report[f"{col}_out_of_domain_converted"] = n_out

        self.report_ = report
        return X


# ---------------------------------------------------------------------------
# Etapa 2 — Indicadores binários de missingness (sinal MNAR)
# ---------------------------------------------------------------------------

class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    """Adiciona colunas `<col>_missing` (1/0) para as colunas informadas,
    capturadas ANTES da imputação. Também adiciona `n_missing_row`
    (contagem de valores ausentes por paciente) como proxy agregado de
    "completude do protocolo aplicado a este paciente".
    """

    def __init__(self, cols=None, add_row_count=True):
        self.cols = cols or ALL_IMPUTED_COLS
        self.add_row_count = add_row_count

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols_present = [c for c in self.cols if c in X.columns]
        for col in cols_present:
            X[f"{col}_missing"] = X[col].isna().astype(int)
        if self.add_row_count:
            X["n_missing_row"] = X[cols_present].isna().sum(axis=1)
        return X


# ---------------------------------------------------------------------------
# Etapa 3 — Imputação em 3 sub-pipelines, condicionada por `dataset`
# ---------------------------------------------------------------------------

@dataclass
class GroupImputerConfig:
    """Configuração de um grupo de imputação (Pipeline A/B/C)."""
    numeric_cols: list = field(default_factory=list)
    categorical_cols: list = field(default_factory=list)  # colunas categóricas/ordinais a imputar
    estimator: str = "bayesian_ridge"   # "bayesian_ridge" ou "random_forest"
    max_iter: int = 10
    n_estimators: int = 50  # usado apenas se estimator == "random_forest"
    max_depth: int = 6


class ConditionalIterativeImputer(BaseEstimator, TransformerMixin):
    """Imputador multivariado (MICE) que sempre inclui `dataset`
    (one-hot) como covariável de condicionamento, além de `age` e
    `sex`, mas NUNCA deixa `dataset` no vetor de saída de features do
    modelo final — ele é usado apenas para "guiar" a imputação.

    Colunas categóricas são convertidas para códigos inteiros (ordinal
    encoding) antes de entrar no IterativeImputer (que exige entrada
    numérica), imputadas, e então arredondadas/limitadas ao intervalo
    de códigos válido antes de decodificar de volta para a categoria
    original.

    Implementa também **imputação múltipla** (parâmetro `n_imputations`):
    quando > 1, gera várias versões imputadas com sementes diferentes.
    Por padrão, o `transform` devolve a média/moda combinada (estimativa
    pontual); as imputações individuais ficam disponíveis via
    `transform_multiple` para quem quiser propagar a incerteza (regra
    de Rubin) em vez de usar um único valor pontual.
    """

    def __init__(
        self,
        config: GroupImputerConfig,
        condition_cols=("age", "sex"),
        center_col=CENTER_COL,
        n_imputations: int = 1,
        random_state: int = 42,
    ):
        self.config = config
        self.condition_cols = list(condition_cols)
        self.center_col = center_col
        self.n_imputations = n_imputations
        self.random_state = random_state

    # -- helpers ------------------------------------------------------
    def _build_estimator(self, seed):
        if self.config.estimator == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=seed,
                n_jobs=-1,
            )
        return BayesianRidge()

    def _encode_categoricals(self, X):
        """Ordinal-encode as colunas categóricas do grupo (fit em treino)."""
        self.ordinal_encoders_ = {}
        X = X.copy()
        for col in self.config.categorical_cols:
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=np.nan
            )
            mask = X[col].notna()
            # fit apenas nos valores observados
            enc.fit(X.loc[mask, [col]])
            self.ordinal_encoders_[col] = enc
            codes = pd.Series(np.nan, index=X.index)
            if mask.any():
                codes.loc[mask] = enc.transform(X.loc[mask, [col]]).ravel()
            X[col] = codes
        return X

    def _decode_categoricals(self, X):
        X = X.copy()
        for col, enc in self.ordinal_encoders_.items():
            n_cats = len(enc.categories_[0])
            codes = X[col].round().clip(0, n_cats - 1).astype(int)
            X[col] = enc.inverse_transform(codes.values.reshape(-1, 1)).ravel()
        return X

    def _build_matrix(self, X):
        """Monta a matriz numérica usada pelo IterativeImputer:
        colunas do grupo (numéricas + categóricas ordinal-encoded) +
        colunas de condicionamento (age, sex, one-hot de dataset)."""
        cols_group = self.config.numeric_cols + self.config.categorical_cols
        mat = X[cols_group].copy()

        # condicionamento: sex binário
        if "sex" in self.condition_cols and "sex" in X.columns:
            mat["__sex_enc"] = (X["sex"].astype(str) == "Male").astype(float)
        if "age" in self.condition_cols and "age" in X.columns:
            mat["__age"] = X["age"].astype(float)

        # condicionamento: dataset one-hot (apenas para guiar a imputação,
        # nunca sai como feature do modelo final)
        if self.center_col in X.columns:
            dataset_dummies = pd.get_dummies(
                X[self.center_col], prefix="__center", dtype=float
            )
            mat = pd.concat([mat, dataset_dummies], axis=1)

        return mat, cols_group

    # -- API sklearn ----------------------------------------------------
    def fit(self, X, y=None):
        X_enc = self._encode_categoricals(X)
        mat, cols_group = self._build_matrix(X_enc)
        self.matrix_columns_ = mat.columns.tolist()
        self.group_cols_ = cols_group

        self.imputers_ = []
        for i in range(self.n_imputations):
            seed = self.random_state + i
            imputer = IterativeImputer(
                estimator=self._build_estimator(seed),
                max_iter=self.config.max_iter,
                random_state=seed,
                sample_posterior=False,
                initial_strategy="median",
            )
            imputer.fit(mat.values)
            self.imputers_.append(imputer)
        return self

    def transform_multiple(self, X):
        """Retorna uma lista de DataFrames (uma por imputação múltipla)."""
        X_enc = self._encode_categoricals_transform(X)
        mat, cols_group = self._build_matrix(X_enc)
        mat = mat.reindex(columns=self.matrix_columns_, fill_value=0.0)

        outputs = []
        for imputer in self.imputers_:
            arr = imputer.transform(mat.values)
            out = pd.DataFrame(arr, columns=mat.columns, index=X.index)
            out_group = out[self.group_cols_]
            X_out = X.copy()
            X_out[self.config.numeric_cols] = out_group[self.config.numeric_cols]
            for col in self.config.categorical_cols:
                X_out[col] = out_group[col]
            X_out = self._decode_categoricals(X_out)
            outputs.append(X_out)
        return outputs

    def _encode_categoricals_transform(self, X):
        X = X.copy()
        for col, enc in self.ordinal_encoders_.items():
            mask = X[col].notna()
            codes = pd.Series(np.nan, index=X.index)
            if mask.any():
                # valores nunca vistos no treino viram NaN (serão imputados)
                known = X.loc[mask, col].isin(enc.categories_[0])
                idx_known = X.loc[mask].index[known.values]
                if len(idx_known) > 0:
                    codes.loc[idx_known] = enc.transform(
                        X.loc[idx_known, [col]]
                    ).ravel()
            X[col] = codes
        return X

    def transform(self, X):
        """Retorna a estimativa pontual combinada (média para numéricas,
        moda entre as imputações para categóricas)."""
        outputs = self.transform_multiple(X)
        if len(outputs) == 1:
            return outputs[0]

        combined = X.copy()
        for col in self.config.numeric_cols:
            combined[col] = np.mean([o[col].values for o in outputs], axis=0)
        for col in self.config.categorical_cols:
            stacked = pd.concat([o[col].rename(i) for i, o in enumerate(outputs)], axis=1)
            combined[col] = stacked.mode(axis=1)[0]
        return combined


class MultiGroupImputer(BaseEstimator, TransformerMixin):
    """Orquestra os 3 sub-pipelines de imputação (A, B, C) descritos no
    plano, aplicando cada `ConditionalIterativeImputer` em sequência
    sobre o mesmo DataFrame."""

    def __init__(self, n_imputations_bc=5, random_state=42):
        self.n_imputations_bc = n_imputations_bc
        self.random_state = random_state

    def fit(self, X, y=None):
        # Pipeline A — leve/moderado, quase-MAR
        cfg_a = GroupImputerConfig(
            numeric_cols=["trestbps", "thalch", "oldpeak"],
            categorical_cols=["exang", "fbs", "restecg"],
            estimator="bayesian_ridge",
        )
        self.imp_a_ = ConditionalIterativeImputer(
            cfg_a, n_imputations=1, random_state=self.random_state
        ).fit(X)

        Xa = self.imp_a_.transform(X)

        # Pipeline B — moderado-alto, componente estrutural (chol, slope)
        cfg_b = GroupImputerConfig(
            numeric_cols=["chol"],
            categorical_cols=["slope"],
            estimator="random_forest",
        )
        self.imp_b_ = ConditionalIterativeImputer(
            cfg_b, n_imputations=self.n_imputations_bc, random_state=self.random_state + 100
        ).fit(Xa)

        Xb = self.imp_b_.transform(Xa)

        # Pipeline C — MNAR extremo estrutural por centro (ca, thal)
        cfg_c = GroupImputerConfig(
            numeric_cols=["ca"],
            categorical_cols=["thal"],
            estimator="random_forest",
        )
        self.imp_c_ = ConditionalIterativeImputer(
            cfg_c, n_imputations=self.n_imputations_bc, random_state=self.random_state + 200
        ).fit(Xb)

        return self

    def transform(self, X):
        Xa = self.imp_a_.transform(X)
        Xb = self.imp_b_.transform(Xa)
        Xc = self.imp_c_.transform(Xb)
        return Xc

    def transform_with_uncertainty(self, X):
        """Retorna as múltiplas versões imputadas de `ca`/`thal`/`chol`/
        `slope` (pipelines B e C), úteis para quem quiser propagar
        incerteza via regra de Rubin em vez de usar o ponto médio."""
        Xa = self.imp_a_.transform(X)
        b_versions = self.imp_b_.transform_multiple(Xa)
        outputs = []
        for xb in b_versions:
            c_versions = self.imp_c_.transform_multiple(xb)
            outputs.extend(c_versions)
        return outputs


# ---------------------------------------------------------------------------
# Etapa 4 — Engenharia de features derivadas
# ---------------------------------------------------------------------------

class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Cria as features derivadas propostas no plano (Seção 2.2):
    faixas etárias clínicas, categorias de colesterol/pressão,
    transformação log de `oldpeak`, indicador `ca_present`, e termos
    de interação sexo × variáveis clínicas.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Faixa etária clínica
        X["age_bin"] = pd.cut(
            X["age"], bins=[0, 45, 55, 65, 120],
            labels=["<45", "45-55", "55-65", "65+"], right=False,
        ).astype(str)

        # Categoria de colesterol (NCEP/ATP III)
        X["chol_category"] = pd.cut(
            X["chol"], bins=[-1, 200, 240, np.inf],
            labels=["desejavel", "limitrofe", "alto"],
        ).astype(str)

        # Categoria de pressão arterial (referência AHA, simplificada)
        X["bp_category"] = pd.cut(
            X["trestbps"], bins=[-1, 120, 130, 140, np.inf],
            labels=["normal", "elevada", "hipertensao_1", "hipertensao_2"],
        ).astype(str)

        # Transformação log para reduzir assimetria de oldpeak
        # (log1p aplicado ao valor deslocado para garantir positividade,
        # já que oldpeak pode ser levemente negativo)
        shift = X["oldpeak"].min()
        shift = 0 if pd.isna(shift) or shift >= 0 else -shift
        X["log_oldpeak"] = np.log1p(X["oldpeak"] + shift)

        # Indicador clínico simplificado de vaso obstruído presente
        X["ca_present"] = (X["ca"].fillna(0) > 0).astype(int)

        # Interações sexo × variáveis clínicas (Seção 2.2)
        sex_male = (X["sex"].astype(str) == "Male").astype(float)
        X["sex_x_age"] = sex_male * X["age"]
        X["sex_x_thalch"] = sex_male * X["thalch"]
        X["sex_x_oldpeak"] = sex_male * X["oldpeak"]

        return X


# ---------------------------------------------------------------------------
# Etapa 5 — Tratamento de outliers via winsorização (fit apenas no treino)
# ---------------------------------------------------------------------------

class Winsorizer(BaseEstimator, TransformerMixin):
    """Limita valores extremos aos percentis [lower, upper] aprendidos
    no treino (capping), sem remover nenhum paciente/linha."""

    def __init__(self, cols=None, lower=0.01, upper=0.99):
        self.cols = cols or ["trestbps", "chol", "thalch", "oldpeak", "log_oldpeak"]
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.bounds_ = {}
        for col in self.cols:
            if col in X.columns:
                lo = X[col].quantile(self.lower)
                hi = X[col].quantile(self.upper)
                self.bounds_[col] = (lo, hi)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lo, upper=hi)
        return X


# ---------------------------------------------------------------------------
# Etapa 6 — Encoding + normalização condicionada ao modelo
# ---------------------------------------------------------------------------

def build_column_transformer(
    model_family: str = "tree",
    include_dataset_as_feature: bool = False,
    feature_columns: Optional[list] = None,
):
    """Monta o ColumnTransformer final de encoding/escalonamento.

    Parameters
    ----------
    model_family : {"tree", "linear"}
        "tree"   -> sem normalização (Random Forest, XGBoost, LightGBM,
                    CatBoost são invariantes a escala); categóricas via
                    OrdinalEncoder (mais compacto, árvores lidam bem).
        "linear" -> RobustScaler nas numéricas (robusto a outliers/mediana
                    e IQR em vez de média/desvio) + OneHotEncoder nas
                    categóricas (Regressão Logística, SVM, KNN, MLP).
    include_dataset_as_feature : bool
        Ver Seção 6 do plano — por padrão `dataset` NÃO entra como
        feature do modelo (evita atalho de confounding centro->target).
        Ativar apenas para um modelo explicitamente "interno multi-centro".
    """

    numeric_features = [
        "age", "trestbps", "chol", "thalch", "oldpeak", "log_oldpeak", "ca",
        "sex_x_age", "sex_x_thalch", "sex_x_oldpeak", "n_missing_row",
    ]
    binary_features = ["sex", "fbs", "exang", "ca_present"] + [
        f"{c}_missing" for c in ALL_IMPUTED_COLS
    ]
    categorical_features = [
        "cp", "restecg", "slope", "thal", "age_bin", "chol_category", "bp_category",
    ]
    if include_dataset_as_feature:
        categorical_features.append(CENTER_COL)

    if feature_columns is not None:
        numeric_features = [c for c in numeric_features if c in feature_columns]
        binary_features = [c for c in binary_features if c in feature_columns]
        categorical_features = [c for c in categorical_features if c in feature_columns]

    if model_family == "linear":
        numeric_pipe = Pipeline([("scaler", RobustScaler())])
        cat_pipe = Pipeline(
            [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        )
        binary_pipe = Pipeline(
            [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"))]
        )
    else:  # tree
        numeric_pipe = "passthrough"
        cat_pipe = Pipeline(
            [("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]
        )
        binary_pipe = Pipeline(
            [("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]
        )

    ct = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("bin", binary_pipe, binary_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return ct


# ---------------------------------------------------------------------------
# Orquestração completa: DataFrame bruto -> DataFrame pronto para o modelo
# ---------------------------------------------------------------------------

class HeartDiseaseFeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """Pipeline de alto nível que encadeia todas as etapas 1–5 do plano
    e devolve um DataFrame "rico" (todas as colunas originais + derivadas
    + indicadores), pronto para a Etapa 6 (encoding/escalonamento
    específico do modelo, feito separadamente via `build_column_transformer`
    para permitir comparar `tree` vs `linear` sem refazer a imputação).
    """

    def __init__(self, n_imputations_bc=5, random_state=42, winsorize=True):
        self.n_imputations_bc = n_imputations_bc
        self.random_state = random_state
        self.winsorize = winsorize

    def fit(self, X, y=None):
        self.domain_cleaner_ = DomainCleaner()
        X1 = self.domain_cleaner_.fit_transform(X)

        self.indicator_adder_ = MissingIndicatorAdder()
        X2 = self.indicator_adder_.fit_transform(X1)

        self.imputer_ = MultiGroupImputer(
            n_imputations_bc=self.n_imputations_bc, random_state=self.random_state
        )
        self.imputer_.fit(X2)
        X3 = self.imputer_.transform(X2)

        self.feature_engineer_ = ClinicalFeatureEngineer()
        X4 = self.feature_engineer_.fit_transform(X3)

        if self.winsorize:
            self.winsorizer_ = Winsorizer()
            self.winsorizer_.fit(X4)

        return self

    def transform(self, X):
        X1 = self.domain_cleaner_.transform(X)
        X2 = self.indicator_adder_.transform(X1)
        X3 = self.imputer_.transform(X2)
        X4 = self.feature_engineer_.transform(X3)
        if self.winsorize:
            X4 = self.winsorizer_.transform(X4)
        return X4


class FEStep(BaseEstimator, TransformerMixin):
    """Adapta `HeartDiseaseFeatureEngineeringPipeline` (que opera sobre
    DataFrame bruto) ao formato fit/transform esperado dentro de um
    `sklearn.pipeline.Pipeline`, permitindo que toda a imputação seja
    refeita a cada fold de validação cruzada (anti-leakage) e usada
    diretamente em `cross_val_score` / `GridSearchCV`.
    """

    def __init__(self, n_imputations_bc=2, random_state=42):
        self.n_imputations_bc = n_imputations_bc
        self.random_state = random_state

    def fit(self, X, y=None):
        self.fe_ = HeartDiseaseFeatureEngineeringPipeline(
            n_imputations_bc=self.n_imputations_bc, random_state=self.random_state
        ).fit(X)
        return self

    def transform(self, X):
        return self.fe_.transform(X)


# ---------------------------------------------------------------------------
# Utilitários de seleção de features (Seção 8 — uso exploratório)
# ---------------------------------------------------------------------------

def compute_vif(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Calcula o VIF (Variance Inflation Factor) para as colunas
    numéricas informadas. Requer `statsmodels`."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = df[numeric_cols].dropna().astype(float)
    X = (X - X.mean()) / X.std(ddof=0)
    X.insert(0, "__const", 1.0)
    vifs = []
    for i, col in enumerate(X.columns):
        if col == "__const":
            continue
        vif = variance_inflation_factor(X.values, i)
        vifs.append({"feature": col, "VIF": round(vif, 2)})
    return pd.DataFrame(vifs).sort_values("VIF", ascending=False).reset_index(drop=True)


def compute_mutual_information(
    df: pd.DataFrame, feature_cols: list, target_col: str, random_state: int = 42
) -> pd.DataFrame:
    """Ranking de informação mútua entre cada feature e o target
    (após ordinal-encoding simples de categóricas, apenas para o cálculo)."""
    X = df[feature_cols].copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ).fit_transform(X[[col]].astype(object))
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

    mi = mutual_info_classif(X, df[target_col], random_state=random_state)
    return (
        pd.DataFrame({"feature": feature_cols, "mutual_info": mi})
        .sort_values("mutual_info", ascending=False)
        .reset_index(drop=True)
    )


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "missing_n": df.isnull().sum(),
            "missing_%": (df.isnull().mean() * 100).round(1),
        }
    ).sort_values("missing_n", ascending=False)
