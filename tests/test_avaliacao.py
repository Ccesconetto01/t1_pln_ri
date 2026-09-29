"""Testes das métricas de relevância, com respostas conhecidas."""

import pytest

from src.avaliacao import mrr_em_k, recall_em_k

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
