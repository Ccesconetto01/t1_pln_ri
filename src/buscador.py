"""Interface unificada dos cinco modelos de recuperação.

A demo e o benchmark usam somente este módulo para criar, indexar e consultar os modelos.
"""

import time
from typing import Protocol

from src.busca_linear import BuscaLinear
from src.dados import DIR_DOCUMENTOS
from src.modelo_bm25 import ModeloBM25
from src.modelo_booleano import ModeloBooleano
from src.modelo_lsa import ModeloLSA
from src.modelo_vetorial import ModeloVetorial

K_PADRAO = 10


class ModeloBusca(Protocol):
    """Interface comum a todos os modelos de recuperação."""

    def indexar(self, documentos: dict[str, str]) -> None:
        """Indexa os documentos (``_id`` -> texto)."""

    def buscar(self, consulta: str, k: int = K_PADRAO) -> list[tuple[str, float]]:
        """Retorna até ``k`` pares ``(_id, score)`` em ordem decrescente de score."""


MODELOS: dict[str, type] = {
    "Busca Linear": BuscaLinear,
    "Booleano": ModeloBooleano,
    "Vetorial": ModeloVetorial,
    "BM25": ModeloBM25,
    "LSA": ModeloLSA,
}


def criar_modelo(nome: str, dir_documentos: str = DIR_DOCUMENTOS) -> ModeloBusca:
    """Instancia um modelo (ainda sem índice) pelo nome.

    Args:
        nome: Chave de ``MODELOS``.
        dir_documentos: Diretório dos ``.txt``, usado somente pela busca linear.

    Returns:
        Instância do modelo, pronta para ``indexar``.

    Raises:
        ValueError: Se o nome não for de um modelo conhecido.
    """
    if nome not in MODELOS:
        raise ValueError(f"Modelo desconhecido: {nome!r}. Opções: {', '.join(MODELOS)}.")
    classe = MODELOS[nome]
    if classe is BuscaLinear:
        return classe(dir_documentos)
    return classe()


def indexar_todos(
    documentos: dict[str, str], dir_documentos: str = DIR_DOCUMENTOS
) -> tuple[dict[str, ModeloBusca], dict[str, float]]:
    """Cria e indexa os cinco modelos, cronometrando cada indexação.

    Args:
        documentos: Corpus (``_id`` -> texto).
        dir_documentos: Diretório dos ``.txt``, usado somente pela busca linear.

    Returns:
        Tupla ``(modelos, tempos_indexacao_ms)``, ambos indexados pelo nome do modelo.
    """
    modelos: dict[str, ModeloBusca] = {}
    tempos_indexacao_ms: dict[str, float] = {}
    for nome in MODELOS:
        modelo = criar_modelo(nome, dir_documentos)
        inicio = time.perf_counter()
        modelo.indexar(documentos)
        tempos_indexacao_ms[nome] = (time.perf_counter() - inicio) * 1000
        modelos[nome] = modelo
    return modelos, tempos_indexacao_ms


def buscar_todos(
    modelos: dict[str, ModeloBusca], consulta: str, k: int = K_PADRAO
) -> dict[str, list[tuple[str, float]]]:
    """Executa a mesma consulta em todos os modelos informados.

    Args:
        modelos: Modelos já indexados, indexados pelo nome.
        consulta: Texto da consulta.
        k: Quantidade máxima de resultados por modelo.

    Returns:
        Dicionário nome do modelo -> lista de pares ``(_id, score)``.
    """
    return {nome: modelo.buscar(consulta, k) for nome, modelo in modelos.items()}
