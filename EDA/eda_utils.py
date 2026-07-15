# eda_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro
from scipy.stats import skew
from scipy.stats import kurtosis


# ---------------------------------------------------
# 1. Informações estruturais
# ---------------------------------------------------

def info_geral(df):

    print("="*60)
    print("DIMENSÕES")
    print("="*60)

    print(f"Linhas: {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")

    print("\nTIPOS DAS VARIÁVEIS")
    print(df.dtypes)



# ---------------------------------------------------
# 2. Qualidade dos dados
# ---------------------------------------------------

def qualidade_dados(df):

    print("\nVALORES AUSENTES")
    print(df.isnull().sum())

    print("\nPERCENTUAL DE AUSÊNCIA")
    print(df.isnull().mean()*100)

    print("\nDUPLICATAS")
    print(df.duplicated().sum())

    print("\nCARDINALIDADE")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} valores únicos")


# ---------------------------------------------------
# 3. Estatísticas descritivas
# ---------------------------------------------------

def estatisticas_descritivas(df):

    print("\nESTATÍSTICAS DESCRITIVAS")
    print(df.describe(include='all'))

    numeric_cols = df.select_dtypes(include=np.number).columns

    print("\nASSIMETRIA (Skewness)")
    for c in numeric_cols:
        print(c, skew(df[c].dropna()))

    print("\nCURTOSE")
    for c in numeric_cols:
        print(c, kurtosis(df[c].dropna()))


# ---------------------------------------------------
# 4. Teste de normalidade
# ---------------------------------------------------

def normalidade(df):

    numeric_cols = df.select_dtypes(include=np.number).columns

    print("\nTESTE SHAPIRO-WILK")

    for c in numeric_cols:

        dados = df[c].dropna()

        if len(dados) < 3:
            continue

        amostra = dados.sample(
            min(5000, len(dados)),
            random_state=42
        )

        stat, p = shapiro(amostra)

        print(f"{c} -> p={p:.4f}")

        if p > 0.05:
            print("Aproximadamente normal")
        else:
            print("Não normal")


# ---------------------------------------------------
# 5. Outliers por IQR
# ---------------------------------------------------

def detectar_outliers(df):

    numeric_cols = df.select_dtypes(include=np.number).columns

    print("\nOUTLIERS (IQR)")

    for c in numeric_cols:

        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)

        iqr = q3-q1

        inferior = q1 - 1.5*iqr
        superior = q3 + 1.5*iqr

        outliers = df[
            (df[c] < inferior) |
            (df[c] > superior)
        ].shape[0]

        print(f"{c}: {outliers}")


# ---------------------------------------------------
# 6. Distribuições
# ---------------------------------------------------

def plot_distribuicoes(df):

    numeric_cols = df.select_dtypes(include=np.number).columns

    for c in numeric_cols:

        plt.figure()
        sns.histplot(df[c], kde=True)
        plt.title(f"Distribuição {c}")
        plt.show()

        plt.figure()
        sns.boxplot(x=df[c])
        plt.title(f"Boxplot {c}")
        plt.show()


# ---------------------------------------------------
# 7. Missingness visual
# ---------------------------------------------------

def plot_missing(df):

    plt.figure()

    sns.heatmap(
        df.isnull(),
        cbar=False
    )

    plt.title("Mapa de Ausências")
    plt.show()


# ---------------------------------------------------
# 8. Correlação
# ---------------------------------------------------

def correlacoes(df):

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) < 2:
        return

    plt.figure(figsize=(10,7))

    sns.heatmap(
        df[numeric_cols].corr(),
        annot=True
    )

    plt.title("Matriz de Correlação")
    plt.show()


# ---------------------------------------------------
# 9. Variáveis categóricas
# ---------------------------------------------------

def analisar_categoricas(df):

    cat_cols = df.select_dtypes(
        exclude=np.number
    ).columns

    for c in cat_cols:

        print(f"\nFrequências {c}")
        print(df[c].value_counts())

        plt.figure()

        sns.countplot(
            data=df,
            x=c
        )

        plt.title(c)
        plt.show()


# ---------------------------------------------------
# 10. Análise do target
# ---------------------------------------------------

def analisar_target(df, target):

    if target not in df.columns:
        return

    print("\nDISTRIBUIÇÃO DE CLASSES")
    print(df[target].value_counts())

    print("\nPROPORÇÕES")
    print(
        df[target].value_counts(normalize=True)
    )

    plt.figure()

    sns.countplot(
        data=df,
        x=target
    )

    plt.title("Balanceamento")
    plt.show()


    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for c in numeric_cols:

        if c == target:
            continue

        plt.figure()

        sns.violinplot(
            data=df,
            x=target,
            y=c
        )

        plt.title(f"{c} por classe")
        plt.show()


# ---------------------------------------------------
# 11. Função principal
# ---------------------------------------------------

def run_eda(df, target=None):

    info_geral(df)

    qualidade_dados(df)

    estatisticas_descritivas(df)

    normalidade(df)

    detectar_outliers(df)

    plot_missing(df)

    plot_distribuicoes(df)

    correlacoes(df)

    analisar_categoricas(df)

    if target:
        analisar_target(
            df,
            target
        )


def salvar_relatorio(df, nome='relatorio.txt'):

    with open(nome, 'w', encoding='utf-8') as f:

        f.write("="*60 + "\n")
        f.write("DIMENSÕES\n")
        f.write("="*60 + "\n")
        f.write(f"Linhas: {df.shape[0]}\n")
        f.write(f"Colunas: {df.shape[1]}\n\n")

        f.write("TIPOS DAS VARIÁVEIS\n")
        f.write(df.dtypes.to_string())
        f.write("\n\n")

        f.write("VALORES AUSENTES\n")
        f.write(df.isnull().sum().to_string())
        f.write("\n\n")

        f.write("DUPLICATAS\n")
        f.write(str(df.duplicated().sum()))
        f.write("\n\n")

        f.write("ESTATÍSTICAS DESCRITIVAS\n")
        f.write(
            df.describe(
                include='all'
            ).to_string()
        )
        f.write("\n\n")