"""Testes da montagem de consultas e qrels, com DataFrames pequenos em memória."""

import pandas as pd

from src.dados import montar_consultas, montar_qrels


def test_montar_consultas_mantem_ordem_e_textos() -> None:
    df_queries = pd.DataFrame({"_id": ["q1", "q0"], "text": ["Como usar o Pix?", "O que é CDB?"]})
    assert montar_consultas(df_queries) == {"q1": "Como usar o Pix?", "q0": "O que é CDB?"}


def test_montar_qrels_agrupa_relevantes_por_consulta() -> None:
    df_qrels = pd.DataFrame({
        "query-id": ["q0", "q0", "q1"],
        "corpus-id": ["a1", "a2", "a3"],
        "score": [1, 1, 1],
    })
    assert montar_qrels(df_qrels) == {"q0": {"a1", "a2"}, "q1": {"a3"}}


def test_montar_qrels_ignora_score_nao_positivo() -> None:
    df_qrels = pd.DataFrame({
        "query-id": ["q0", "q0", "q1"],
        "corpus-id": ["a1", "a2", "a3"],
        "score": [1, 0, 0],
    })
    assert montar_qrels(df_qrels) == {"q0": {"a1"}}
