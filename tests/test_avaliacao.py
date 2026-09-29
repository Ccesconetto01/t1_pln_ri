"""Testes das métricas de relevância, com respostas conhecidas."""

import pytest

from src.avaliacao import avaliar_modelo, mrr_em_k, recall_em_k
from src.modelo_bm25 import ModeloBM25

RANKING = ["d1", "d2", "d3", "d4", "d5"]


def test_mrr_relevante_na_primeira_posicao() -> None:
    assert mrr_em_k(RANKING, {"d1"}) == 1.0


def test_mrr_relevante_na_terceira_posicao() -> None:
    assert mrr_em_k(RANKING, {"d3"}) == pytest.approx(1 / 3)


def test_mrr_usa_o_primeiro_relevante() -> None:
    assert mrr_em_k(RANKING, {"d4", "d2"}) == pytest.approx(1 / 2)


def test_mrr_relevante_alem_de_k() -> None:
    assert mrr_em_k(RANKING, {"d4"}, k=3) == 0.0


def test_mrr_sem_relevantes_ou_ranking_vazio() -> None:
    assert mrr_em_k(RANKING, set()) == 0.0
    assert mrr_em_k([], {"d1"}) == 0.0


def test_recall_fracao_dos_relevantes_no_top_k() -> None:
    assert recall_em_k(RANKING, {"d2"}) == 1.0
    assert recall_em_k(RANKING, {"d2", "d9"}) == pytest.approx(1 / 2)
    assert recall_em_k(RANKING, {"d1", "d5"}, k=3) == pytest.approx(1 / 2)


def test_recall_sem_relevantes_ou_ranking_vazio() -> None:
    assert recall_em_k(RANKING, set()) == 0.0
    assert recall_em_k([], {"d1"}) == 0.0


def test_avaliar_modelo_no_mini_corpus(documentos: dict[str, str]) -> None:
    modelo = ModeloBM25()
    modelo.indexar(documentos)
    consultas = {
        "q1": "consentimento revogar",  # d3 em 1º -> MRR 1
        "q2": "pix pagamento",  # ranking [d1, d4] -> MRR 1/2
        "q3": "termoinexistente",  # ranking vazio -> MRR 0
        "q4": "boleto",  # sem qrels: fica fora da média
    }
    qrels = {"q1": {"d3"}, "q2": {"d4"}, "q3": {"d2"}}
    resultado = avaliar_modelo(modelo, consultas, qrels)
    assert resultado["n_consultas"] == 3
    assert resultado["mrr"] == pytest.approx((1 + 1 / 2 + 0) / 3)
    assert resultado["recall"] == pytest.approx(2 / 3)


def test_avaliar_modelo_sem_consultas_com_relevantes(documentos: dict[str, str]) -> None:
    modelo = ModeloBM25()
    modelo.indexar(documentos)
    assert avaliar_modelo(modelo, {"q1": "pix"}, {}) == {
        "mrr": 0.0, "recall": 0.0, "n_consultas": 0
    }
