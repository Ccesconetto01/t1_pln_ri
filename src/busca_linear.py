"""Busca Linear Ingênua: baseline O(N) sem índice."""

import os

from src.preprocessamento import tokenizar_e_filtrar


class BuscaLinear:
    """Varre todos os documentos do disco a cada consulta (AND implícito dos termos).

    Não há índice: cada consulta abre, lê e tokeniza os ``N`` arquivos ``.txt``, o que torna
    o custo da consulta O(N · L), com L o tamanho médio do documento. Serve de baseline para
    os demais modelos.

    Attributes:
        dir_documentos: Diretório com um arquivo ``<_id>.txt`` por documento.
        ids: Ids dos documentos, na ordem do corpus.
    """

    def __init__(self, dir_documentos: str) -> None:
        """Inicializa a busca.

        Args:
            dir_documentos: Diretório com os arquivos ``<_id>.txt``, gerados por
                ``src.dados.salvar_documentos_txt``.
        """
        self.dir_documentos = dir_documentos
        self.ids: list[str] = []

    def indexar(self, documentos: dict[str, str]) -> None:
        """Guarda apenas os ids dos documentos; o texto é lido do disco na consulta.

        Args:
            documentos: Dicionário ``_id`` -> texto do documento. Só as chaves são usadas.
        """
        self.ids = list(documentos)

    def buscar(self, consulta: str, k: int = 10) -> list[tuple[str, float]]:
        """Retorna os documentos que contêm todos os termos da consulta.

        Args:
            consulta: Texto livre da consulta. Todos os termos são exigidos (AND implícito).
            k: Quantidade máxima de resultados.

        Returns:
            Até ``k`` pares ``(_id, 1.0)`` na ordem do corpus: o modelo não ranqueia, então
            o score é fixo. Lista vazia se a consulta não tiver nenhum termo válido.
        """
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
