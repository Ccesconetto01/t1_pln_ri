"""Modelo Vetorial clássico: ponderação TF-IDF e similaridade do cosseno."""

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessamento import tokenizar_e_filtrar


def criar_vetorizador() -> TfidfVectorizer:
    """Cria o vetorizador TF-IDF usado pelo Vetorial e pelo LSA.

    Returns:
        ``TfidfVectorizer`` que usa ``tokenizar_e_filtrar`` como analisador e TF sublinear.
    """
    return TfidfVectorizer(analyzer=tokenizar_e_filtrar, sublinear_tf=True)


class ModeloVetorial:
    """Ranqueia documentos pelo cosseno entre os vetores TF-IDF da consulta e do documento."""

    def __init__(self) -> None:
        """Inicializa o modelo sem índice."""
        self.ids: list[str] = []
        self.vetorizador: TfidfVectorizer = criar_vetorizador()
        self.matriz_tfidf: Any = None  # matriz esparsa (documentos x termos)

    def indexar(self, documentos: dict[str, str]) -> None:
        """Ajusta o vocabulário e constrói a matriz TF-IDF dos documentos.

        Args:
            documentos: Dicionário ``_id`` -> texto do documento.
        """
        self.ids = list(documentos)
        self.matriz_tfidf = self.vetorizador.fit_transform(
            [documentos[_id] for _id in self.ids]
        )

    def buscar(self, consulta: str, k: int = 10) -> list[tuple[str, float]]:
        """Retorna os ``k`` documentos mais parecidos com a consulta.

        Args:
            consulta: Texto livre da consulta.
            k: Quantidade máxima de resultados.

        Returns:
            Pares ``(_id, score)`` em ordem decrescente de cosseno. Documentos com cosseno
            zero (nenhum termo em comum) não são retornados.
        """
        q_vetor = self.vetorizador.transform([consulta])
        if q_vetor.nnz == 0:
            return []
        scores = cosine_similarity(q_vetor, self.matriz_tfidf).ravel()
        posicoes = np.argsort(scores)[::-1][:k]
        return [(self.ids[p], float(scores[p])) for p in posicoes if scores[p] > 0]
