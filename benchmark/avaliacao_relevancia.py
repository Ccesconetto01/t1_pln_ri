"""Avaliação de relevância (MRR@10 e Recall@10) dos modelos ranqueados no corpus completo.

Uso, a partir da raiz do projeto::

    python -m benchmark.avaliacao_relevancia

Busca Linear e Booleano não entram: eles não ranqueiam (score fixo 1.0, ordem do corpus).
"""

import os

import pandas as pd

from src.avaliacao import K_AVALIACAO, avaliar_modelo
from src.buscador import ModeloBusca, criar_modelo
from src.dados import carregar_dados, montar_consultas, montar_documentos, montar_qrels
from src.modelo_bm25 import ModeloBM25

DIR_RESULTADOS = os.path.join("benchmark", "resultados")
ARQUIVO_CSV = os.path.join(DIR_RESULTADOS, "relevancia.csv")
MODELOS_RANQUEADOS = ("Vetorial", "BM25", "LSA")


def criar_modelos_avaliados() -> dict[str, ModeloBusca]:
    """Cria os modelos ranqueados e a variante do BM25 sem normalização de comprimento.

    Returns:
        Dicionário nome -> modelo ainda sem índice.
    """
    modelos: dict[str, ModeloBusca] = {nome: criar_modelo(nome) for nome in MODELOS_RANQUEADOS}
    modelos["BM25 (b=0)"] = ModeloBM25(b=0.0)
    return modelos


def main() -> None:
    """Indexa o corpus completo, avalia cada modelo e grava o CSV de resultados."""
    df_corpus, df_queries, df_qrels = carregar_dados()
    documentos = montar_documentos(df_corpus)
    consultas = montar_consultas(df_queries)
    qrels = montar_qrels(df_qrels)
    linhas = []
    for nome, modelo in criar_modelos_avaliados().items():
        modelo.indexar(documentos)
        metricas = avaliar_modelo(modelo, consultas, qrels, K_AVALIACAO)
        linhas.append({
            "modelo": nome,
            f"mrr_{K_AVALIACAO}": round(metricas["mrr"], 4),
            f"recall_{K_AVALIACAO}": round(metricas["recall"], 4),
            "n_consultas": metricas["n_consultas"],
        })
    tabela = pd.DataFrame(linhas)
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    tabela.to_csv(ARQUIVO_CSV, index=False)
    print(tabela.to_string(index=False))
    print(f"\nResultados gravados em {ARQUIVO_CSV}")


if __name__ == "__main__":
    main()
