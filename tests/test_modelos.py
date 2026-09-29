"""Testes dos modelos de recuperação, sobre o mini-corpus fixo."""

import pytest

from src.busca_linear import BuscaLinear
from src.modelo_bm25 import ModeloBM25
from src.modelo_booleano import ModeloBooleano
from src.modelo_vetorial import ModeloVetorial

CONSULTAS_SEM_OPERADORES = [
    "pix",
    "Pix pagamento",
    "consentimento revogar",
    "juros consignado",
    "termoinexistente",
    "pix termoinexistente",
    "",
]


@pytest.fixture
def busca_linear(documentos: dict[str, str], dir_documentos: str) -> BuscaLinear:
    modelo = BuscaLinear(dir_documentos)
    modelo.indexar(documentos)
    return modelo


@pytest.fixture
def booleano(documentos: dict[str, str]) -> ModeloBooleano:
    modelo = ModeloBooleano()
    modelo.indexar(documentos)
    return modelo


@pytest.fixture
def vetorial(documentos: dict[str, str]) -> ModeloVetorial:
    modelo = ModeloVetorial()
    modelo.indexar(documentos)
    return modelo


@pytest.fixture
def bm25(documentos: dict[str, str]) -> ModeloBM25:
    modelo = ModeloBM25()
    modelo.indexar(documentos)
    return modelo


def ids(resultado: list[tuple[str, float]]) -> list[str]:
    return [_id for _id, _ in resultado]


# --- Busca Linear -----------------------------------------------------------------------


def test_busca_linear_retorna_documentos_com_todos_os_termos(busca_linear: BuscaLinear) -> None:
    assert ids(busca_linear.buscar("pix")) == ["d1", "d4"]
    assert ids(busca_linear.buscar("pix boleto")) == ["d4"]


def test_busca_linear_score_e_um_e_respeita_k(busca_linear: BuscaLinear) -> None:
    resultado = busca_linear.buscar("pix", k=1)
    assert resultado == [("d1", 1.0)]


def test_busca_linear_consulta_vazia_ou_sem_correspondencia(busca_linear: BuscaLinear) -> None:
    assert busca_linear.buscar("") == []
    assert busca_linear.buscar("termoinexistente") == []


# --- Modelo Booleano --------------------------------------------------------------------


@pytest.mark.parametrize("consulta", CONSULTAS_SEM_OPERADORES)
def test_booleano_equivale_a_busca_linear(
    consulta: str, booleano: ModeloBooleano, busca_linear: BuscaLinear
) -> None:
    assert set(ids(booleano.buscar(consulta))) == set(ids(busca_linear.buscar(consulta)))


def test_booleano_and_or_not(booleano: ModeloBooleano) -> None:
    assert ids(booleano.buscar("pix AND boleto")) == ["d4"]
    assert ids(booleano.buscar("consignado OR imobiliario")) == ["d2", "d5"]
    assert ids(booleano.buscar("pix AND NOT boleto")) == ["d1"]
    assert ids(booleano.buscar("NOT pix")) == ["d2", "d3", "d5"]


def test_booleano_avalia_da_esquerda_para_a_direita(booleano: ModeloBooleano) -> None:
    # (consignado OR pix) AND boleto = {d2, d1, d4} AND {d4}; sem precedência de AND sobre OR.
    assert ids(booleano.buscar("consignado OR pix AND boleto")) == ["d4"]


def test_booleano_operadores_so_em_maiusculas(booleano: ModeloBooleano) -> None:
    # "and" minúsculo é um termo comum (fora do vocabulário), logo o AND implícito zera tudo.
    assert booleano.buscar("pix and boleto") == []


def test_booleano_termo_fora_do_vocabulario(booleano: ModeloBooleano) -> None:
    assert booleano.buscar("termoinexistente") == []
    assert ids(booleano.buscar("pix OR termoinexistente")) == ["d1", "d4"]
    assert ids(booleano.buscar("NOT termoinexistente")) == ["d1", "d2", "d3", "d4", "d5"]


def test_booleano_ordem_do_corpus_score_e_k(booleano: ModeloBooleano) -> None:
    assert booleano.buscar("NOT pix", k=2) == [("d2", 1.0), ("d3", 1.0)]


def test_booleano_consulta_vazia_ou_so_operadores(booleano: ModeloBooleano) -> None:
    assert booleano.buscar("") == []
    assert booleano.buscar("AND OR NOT") == []


# --- Modelos ranqueados (Vetorial e BM25) -----------------------------------------------

MODELOS_RANQUEADOS = ["vetorial", "bm25"]


@pytest.fixture
def ranqueado(request: pytest.FixtureRequest):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("ranqueado", MODELOS_RANQUEADOS, indirect=True)
def test_ranqueado_documento_mais_obvio_em_primeiro(ranqueado) -> None:
    assert ids(ranqueado.buscar("consentimento revogar"))[0] == "d3"
    assert ids(ranqueado.buscar("alienação fiduciária"))[0] == "d5"
    assert ids(ranqueado.buscar("consignado"))[0] == "d2"
    # d1 contém "pix" e "pagamento"; d4 contém apenas "pix".
    assert ids(ranqueado.buscar("pix pagamento"))[:2] == ["d1", "d4"]


@pytest.mark.parametrize("ranqueado", MODELOS_RANQUEADOS, indirect=True)
def test_ranqueado_scores_em_ordem_decrescente_e_respeita_k(ranqueado) -> None:
    resultado = ranqueado.buscar("pix pagamento boleto")
    scores = [score for _, score in resultado]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)
    assert len(ranqueado.buscar("pix pagamento boleto", k=1)) == 1


@pytest.mark.parametrize("ranqueado", MODELOS_RANQUEADOS, indirect=True)
def test_ranqueado_so_retorna_documentos_com_algum_termo(ranqueado) -> None:
    assert set(ids(ranqueado.buscar("pix"))) == {"d1", "d4"}


@pytest.mark.parametrize("ranqueado", MODELOS_RANQUEADOS, indirect=True)
def test_ranqueado_consulta_vazia_stopwords_ou_fora_do_vocabulario(ranqueado) -> None:
    assert ranqueado.buscar("") == []
    assert ranqueado.buscar("de a o") == []
    assert ranqueado.buscar("termoinexistente") == []


def test_bm25_guarda_hiperparametros(documentos: dict[str, str]) -> None:
    modelo = ModeloBM25(k1=1.2, b=0.5)
    assert (modelo.k1, modelo.b) == (1.2, 0.5)
    modelo = ModeloBM25()
    assert (modelo.k1, modelo.b) == (1.5, 0.75)


def test_bm25_b_controla_normalizacao_por_comprimento() -> None:
    # "curto" e "longo" têm um "pix" cada; só o comprimento do documento é diferente.
    corpus = {
        "curto": "pix pagamento",
        "longo": "pix pagamento instantaneo limite noturno transferencias boleto lotericas",
        "x1": "consignado juros",
        "x2": "consentimento revogar",
        "x3": "alienacao fiduciaria",
    }
    sem_norm = ModeloBM25(b=0.0)
    sem_norm.indexar(corpus)
    com_norm = ModeloBM25(b=0.75)
    com_norm.indexar(corpus)
    scores_sem = dict(sem_norm.buscar("pix"))
    scores_com = dict(com_norm.buscar("pix"))
    assert scores_sem["curto"] == pytest.approx(scores_sem["longo"])
    assert scores_com["curto"] > scores_com["longo"]
