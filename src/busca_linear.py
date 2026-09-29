"""Busca Linear Ingênua: baseline O(N) sem índice."""

import os

from src.preprocessamento import tokenizar_e_filtrar


class BuscaLinear:
    """Varre todos os documentos do disco a cada consulta (AND implícito dos termos)."""

    def __init__(self, dir_documentos: str) -> None:
        """Inicializa a busca."""

        self.dir_documentos = dir_documentos
        self.ids: list[str] = []

    def indexar(self, documentos: dict[str, str]) -> None:
        """Guarda apenas os ids dos documentos; o texto é lido do disco na consulta."""

        self.ids = list(documentos)

    def buscar(self, consulta: str, k: int = 10) -> list[tuple[str, float]]:
        """Retorna os documentos que contêm todos os termos da consulta."""

        q_tokens = set(tokenizar_e_filtrar(consulta))
        if not q_tokens:
            return []
        encontrados: list[tuple[str, float]] = []
        for _id in self.ids:
            caminho = os.path.join(self.dir_documentos, f"{_id}.txt")
            with open(caminho, encoding="utf-8") as arquivo:
                tokens_doc = set(tokenizar_e_filtrar(arquivo.read()))
            if q_tokens <= tokens_doc:
                encontrados.append((_id, 1.0))
        return encontrados[:k]
