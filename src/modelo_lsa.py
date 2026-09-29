"""Modelo Algébrico Alternativo: Análise Semântica Latente (LSA) via TruncatedSVD."""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from src.modelo_vetorial import criar_vetorizador

COMPONENTES_PADRAO = 100
SEMENTE_SVD = 42


class ModeloLSA:
    """Projeta a matriz TF-IDF em um espaço latente e ranqueia pelo cosseno nesse espaço.

    Attributes:
        n_componentes: Dimensão do espaço latente, definida em ``indexar``.
    """

    def __init__(self, componentes: int = COMPONENTES_PADRAO) -> None:
        """Inicializa o modelo.

        Args:
            componentes: Dimensão máxima desejada do espaço latente. O valor efetivo é
                reduzido se o corpus for pequeno demais.
        """
        self.componentes = componentes
        self.n_componentes = 0
        self.ids: list[str] = []
        self.vetorizador = criar_vetorizador()
        self.svd: TruncatedSVD | None = None
        self.matriz_lsa: np.ndarray | None = None

    def indexar(self, documentos: dict[str, str]) -> None:
        """Constrói a matriz TF-IDF e a reduz com ``TruncatedSVD``.

        O número de componentes é ``min(componentes, n_documentos - 1, n_termos - 1)``,
        para que o SVD funcione também em corpora pequenos (ex.: N = 100).

        Args:
            documentos: Dicionário ``_id`` -> texto do documento.

        Raises:
            ValueError: Se o corpus for pequeno demais para gerar ao menos 1 componente.
        """
        self.ids = list(documentos)
        matriz_tfidf = self.vetorizador.fit_transform([documentos[_id] for _id in self.ids])
        n_documentos, n_termos = matriz_tfidf.shape
        self.n_componentes = min(self.componentes, n_documentos - 1, n_termos - 1)
        if self.n_componentes < 1:
            raise ValueError(
                f"Corpus pequeno demais para o LSA ({n_documentos} documentos, "
                f"{n_termos} termos)."
            )
        self.svd = TruncatedSVD(n_components=self.n_componentes, random_state=SEMENTE_SVD)
        self.matriz_lsa = self.svd.fit_transform(matriz_tfidf)

    def buscar(self, consulta: str, k: int = 10) -> list[tuple[str, float]]:
        """Retorna os ``k`` documentos mais próximos da consulta no espaço latente.

        Args:
            consulta: Texto livre da consulta.
            k: Quantidade máxima de resultados.

        Returns:
            Pares ``(_id, score)`` em ordem decrescente de cosseno. Documentos com cosseno
            não positivo não são retornados.
        """
        q_vetor = self.vetorizador.transform([consulta])
        if q_vetor.nnz == 0:
            return []
        q_lsa = self.svd.transform(q_vetor)
        scores_lsa = cosine_similarity(q_lsa, self.matriz_lsa).ravel()
        posicoes = np.argsort(scores_lsa)[::-1][:k]
        return [(self.ids[p], float(scores_lsa[p])) for p in posicoes if scores_lsa[p] > 0]
