"""Métricas de relevância (MRR@k e Recall@k) sobre os julgamentos dos qrels."""

K_AVALIACAO = 10


def mrr_em_k(ranking: list[str], relevantes: set[str], k: int = K_AVALIACAO) -> float:
    """Calcula o recíproco da posição do primeiro documento relevante no Top-k.

    Args:
        ranking: Ids dos documentos retornados, do mais para o menos relevante.
        relevantes: Ids dos documentos relevantes para a consulta.
        k: Tamanho do corte do ranking.

    Returns:
        ``1 / posição`` do primeiro relevante (posições a partir de 1), ou ``0.0`` se
        nenhum relevante aparecer entre os ``k`` primeiros.
    """
    for posicao, _id in enumerate(ranking[:k], start=1):
        if _id in relevantes:
            return 1.0 / posicao
    return 0.0


def recall_em_k(ranking: list[str], relevantes: set[str], k: int = K_AVALIACAO) -> float:
    """Calcula a fração dos documentos relevantes que aparece no Top-k.

    Args:
        ranking: Ids dos documentos retornados, do mais para o menos relevante.
        relevantes: Ids dos documentos relevantes para a consulta.
        k: Tamanho do corte do ranking.

    Returns:
        ``|relevantes ∩ Top-k| / |relevantes|``, ou ``0.0`` se não houver relevantes.
    """
    if not relevantes:
        return 0.0
    return len(relevantes & set(ranking[:k])) / len(relevantes)
