"""Métricas de relevância (MRR@k e Recall@k) sobre os julgamentos dos qrels."""

from src.buscador import ModeloBusca

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


def avaliar_modelo(
    modelo: ModeloBusca,
    consultas: dict[str, str],
    qrels: dict[str, set[str]],
    k: int = K_AVALIACAO,
) -> dict[str, float]:
    """Calcula MRR@k e Recall@k médios de um modelo já indexado.

    Só entram na média as consultas que têm ao menos um documento relevante nos qrels.

    Args:
        modelo: Modelo já indexado.
        consultas: Dicionário ``query-id`` -> texto da consulta.
        qrels: Dicionário ``query-id`` -> conjunto de ``corpus-id`` relevantes.
        k: Tamanho do corte do ranking.

    Returns:
        Dicionário com as chaves ``"mrr"``, ``"recall"`` e ``"n_consultas"``.
    """
    avaliadas = [query_id for query_id in consultas if qrels.get(query_id)]
    if not avaliadas:
        return {"mrr": 0.0, "recall": 0.0, "n_consultas": 0}
    soma_mrr = 0.0
    soma_recall = 0.0
    for query_id in avaliadas:
        ranking = [_id for _id, _ in modelo.buscar(consultas[query_id], k)]
        soma_mrr += mrr_em_k(ranking, qrels[query_id], k)
        soma_recall += recall_em_k(ranking, qrels[query_id], k)
    return {
        "mrr": soma_mrr / len(avaliadas),
        "recall": soma_recall / len(avaliadas),
        "n_consultas": len(avaliadas),
    }
