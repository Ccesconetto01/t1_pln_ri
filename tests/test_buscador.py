"""Testes da interface unificada dos modelos."""

import pytest

from src.buscador import MODELOS, buscar_todos, criar_modelo, indexar_todos


def test_modelos_tem_os_cinco_algoritmos() -> None:
    assert list(MODELOS) == ["Busca Linear", "Booleano", "Vetorial", "BM25", "LSA"]


def test_criar_modelo_desconhecido_levanta_erro(dir_documentos: str) -> None:
    with pytest.raises(ValueError):
        criar_modelo("Inexistente", dir_documentos)


def test_indexar_todos_devolve_modelos_e_tempos(
    documentos: dict[str, str], dir_documentos: str
) -> None:
    modelos, tempos_ms = indexar_todos(documentos, dir_documentos)
    assert list(modelos) == list(MODELOS)
    assert list(tempos_ms) == list(MODELOS)
    assert all(tempo >= 0 for tempo in tempos_ms.values())


def test_buscar_todos_consulta_todos_os_modelos(
    documentos: dict[str, str], dir_documentos: str
) -> None:
    modelos, _ = indexar_todos(documentos, dir_documentos)
    resultados = buscar_todos(modelos, "pix", k=5)
    assert list(resultados) == list(MODELOS)
    for nome, resultado in resultados.items():
        # O LSA pode devolver mais documentos (similaridade no espaço latente),
        # mas os dois que contêm "pix" vêm primeiro em todos os modelos.
        assert {_id for _id, _ in resultado[:2]} == {"d1", "d4"}, nome
        assert len(resultado) <= 5


def test_buscar_todos_consulta_sem_resultado(
    documentos: dict[str, str], dir_documentos: str
) -> None:
    modelos, _ = indexar_todos(documentos, dir_documentos)
    resultados = buscar_todos(modelos, "termoinexistente")
    assert all(resultado == [] for resultado in resultados.values())
