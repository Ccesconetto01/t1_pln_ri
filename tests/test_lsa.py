"""Testes do LSA, sobre o mini-corpus fixo e sobre corpora ainda menores."""

import pytest

from src.modelo_lsa import ModeloLSA


@pytest.fixture
def lsa(documentos: dict[str, str]) -> ModeloLSA:
    modelo = ModeloLSA()
    modelo.indexar(documentos)
    return modelo


def test_lsa_ajusta_componentes_ao_tamanho_do_corpus(
    lsa: ModeloLSA, documentos: dict[str, str]
) -> None:
    # 5 documentos: o limite é n_documentos - 1, bem abaixo dos 100 padrão.
    assert lsa.n_componentes == len(documentos) - 1
    assert lsa.svd.n_components == lsa.n_componentes
    assert lsa.matriz_lsa.shape == (len(documentos), lsa.n_componentes)


def test_lsa_respeita_limite_de_componentes_informado(documentos: dict[str, str]) -> None:
    modelo = ModeloLSA(componentes=2)
    modelo.indexar(documentos)
    assert modelo.n_componentes == 2


def test_lsa_funciona_com_dois_documentos() -> None:
    modelo = ModeloLSA()
    modelo.indexar({"a": "pix pagamento instantaneo", "b": "boleto bancario loterica"})
    assert modelo.n_componentes == 1
    assert modelo.buscar("pix")[0][0] == "a"


def test_lsa_corpus_pequeno_demais_levanta_erro() -> None:
    modelo = ModeloLSA()
    with pytest.raises(ValueError):
        modelo.indexar({"a": "pix pagamento instantaneo"})


def test_lsa_documento_mais_obvio_em_primeiro(lsa: ModeloLSA) -> None:
    assert lsa.buscar("consentimento revogar")[0][0] == "d3"
    assert lsa.buscar("alienação fiduciária")[0][0] == "d5"
    assert lsa.buscar("consignado")[0][0] == "d2"


def test_lsa_scores_em_ordem_decrescente_e_respeita_k(lsa: ModeloLSA) -> None:
    resultado = lsa.buscar("pix pagamento boleto")
    scores = [score for _, score in resultado]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)
    assert len(lsa.buscar("pix pagamento boleto", k=1)) == 1


def test_lsa_consulta_vazia_stopwords_ou_fora_do_vocabulario(lsa: ModeloLSA) -> None:
    assert lsa.buscar("") == []
    assert lsa.buscar("de a o") == []
    assert lsa.buscar("termoinexistente") == []


def test_lsa_e_deterministico(documentos: dict[str, str]) -> None:
    primeiro = ModeloLSA()
    primeiro.indexar(documentos)
    segundo = ModeloLSA()
    segundo.indexar(documentos)
    assert primeiro.buscar("pix pagamento") == segundo.buscar("pix pagamento")
