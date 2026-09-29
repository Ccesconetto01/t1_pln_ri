"""Modelo Probabilístico Okapi BM25."""

import numpy as np
from rank_bm25 import BM25Okapi

from src.preprocessamento import tokenizar_e_filtrar


class ModeloBM25:
    """Ranqueia documentos com Okapi BM25.

    Attributes:
        k1: Saturação da frequência do termo (quanto maior, mais o TF pesa).
        b: Normalização pelo comprimento do documento (0 = ignora, 1 = normaliza por completo).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Inicializa o modelo com os hiperparâmetros do BM25.

        Args:
            k1: Parâmetro de saturação de frequência.
            b: Parâmetro de normalização de comprimento.
        """
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self.bm25: BM25Okapi | None = None

    def indexar(self, documentos: dict[str, str]) -> None:
        """Tokeniza os documentos e constrói as estatísticas do BM25.

        Args:
            documentos: Dicionário ``_id`` -> texto do documento.
        """
        self.ids = list(documentos)
        corpus_tokenizado = [tokenizar_e_filtrar(documentos[_id]) for _id in self.ids]
        self.bm25 = BM25Okapi(corpus_tokenizado, k1=self.k1, b=self.b)

    def buscar(self, consulta: str, k: int = 10) -> list[tuple[str, float]]:
        """Retorna os ``k`` documentos com maior score BM25.

        Args:
            consulta: Texto livre da consulta.
            k: Quantidade máxima de resultados.

        Returns:
            Pares ``(_id, score)`` em ordem decrescente de score. Documentos com score
            não positivo (sem nenhum termo da consulta) não são retornados.
        """
        q_tokens = tokenizar_e_filtrar(consulta)
        if not q_tokens:
            return []
        scores_bm25 = self.bm25.get_scores(q_tokens)
        posicoes = np.argsort(scores_bm25)[::-1][:k]
        return [(self.ids[p], float(scores_bm25[p])) for p in posicoes if scores_bm25[p] > 0]
