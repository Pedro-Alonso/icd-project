"""
Divide o dataset Heart Disease UCI em conjuntos de treino e validação,
estratificados por sexo e pela variável alvo (num).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

CAMINHO_DADOS = 'data/heart_disease_uci.csv'
SEMENTE = 42
PROPORCAO_VALIDACAO = 0.15
COLUNA_ALVO = 'num'
COLUNA_SEXO = 'sex'

SAIDAS = {
    'homem_treino': 'data/conjunto_homem_treino.csv',
    'homem_validacao': 'data/conjunto_homem_validacao.csv',
    'mulher_treino': 'data/conjunto_mulher_treino.csv',
    'mulher_validacao': 'data/conjunto_mulher_validacao.csv',
}


def imprimir_distribuicao(nome, df):
    print(f"\n  {nome} ({len(df)} registros)")
    dist = df[COLUNA_ALVO].value_counts().sort_index()
    for classe, contagem in dist.items():
        pct = contagem / len(df) * 100
        print(f"    num={classe}: {contagem:>4}  ({pct:.1f}%)")


def main():
    df = pd.read_csv(CAMINHO_DADOS)
    print(f"Dataset carregado: {len(df)} registros")

    df_homem = df[df[COLUNA_SEXO] == 'Male'].copy()
    df_mulher = df[df[COLUNA_SEXO] == 'Female'].copy()
    print(f"Homens: {len(df_homem)} | Mulheres: {len(df_mulher)}")

    for nome, grupo in [('Mulheres', df_mulher), ('Homens', df_homem)]:
        min_classe = grupo[COLUNA_ALVO].value_counts().min()
        if min_classe < 2:
            print(f"\n⚠ AVISO: {nome} possui classe com apenas {min_classe} "
                  f"amostra(s) — estratificação pode falhar.")

    homem_treino, homem_val = train_test_split(
        df_homem,
        test_size=PROPORCAO_VALIDACAO,
        random_state=SEMENTE,
        stratify=df_homem[COLUNA_ALVO],
    )

    mulher_treino, mulher_val = train_test_split(
        df_mulher,
        test_size=PROPORCAO_VALIDACAO,
        random_state=SEMENTE,
        stratify=df_mulher[COLUNA_ALVO],
    )

    conjuntos = {
        'homem_treino': homem_treino,
        'homem_validacao': homem_val,
        'mulher_treino': mulher_treino,
        'mulher_validacao': mulher_val,
    }

    for chave, conj in conjuntos.items():
        caminho = SAIDAS[chave]
        conj.to_csv(caminho, index=False, encoding='utf-8')

    print("\n" + "=" * 60)
    print("RESUMO DOS CONJUNTOS GERADOS")
    print("=" * 60)

    for chave, conj in conjuntos.items():
        imprimir_distribuicao(SAIDAS[chave], conj)

    total = sum(len(c) for c in conjuntos.values())
    print(f"\nTotal de registros nos 4 arquivos: {total}")
    print(f"Total original: {len(df)}")
    print(f"Conferência: {'OK' if total == len(df) else 'DIVERGENTE'}")


if __name__ == '__main__':
    main()
