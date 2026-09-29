"""Download, cache local e amostragem do dataset faq-bacen (MTEB-BR).

O download do HuggingFace acontece uma única vez. Depois disso, tudo é lido dos parquets
salvos em ``data/raw/``, para que a demo funcione sem internet.
"""

import os
import shutil

import numpy as np
import pandas as pd

REPOSITORIO_HF = "MTEB-BR/faq-bacen"
NOMES_CONJUNTOS = ("corpus", "queries", "qrels")
DIR_RAW = os.path.join("data", "raw")
DIR_DOCUMENTOS = os.path.join("data", "processed", "documentos")
SEMENTE = 42
TAMANHO_TITULO = 30


def baixar_dataset(dir_raw: str = DIR_RAW) -> None:
    """Baixa os três parquets do HuggingFace e os salva em ``dir_raw``.

    Arquivos que já existem localmente não são baixados de novo.

    Args:
        dir_raw: Diretório onde os parquets serão gravados.
    """
    # Import tardio: só é necessário quando o cache local ainda não existe.
    from huggingface_hub import hf_hub_download

    os.makedirs(dir_raw, exist_ok=True)
    for nome in NOMES_CONJUNTOS:
        destino = os.path.join(dir_raw, f"{nome}.parquet")
        if os.path.exists(destino):
            continue
        origem = hf_hub_download(
            REPOSITORIO_HF, f"{nome}/test-00000-of-00001.parquet", repo_type="dataset"
        )
        shutil.copyfile(origem, destino)


def carregar_dados(
    dir_raw: str = DIR_RAW,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega corpus, queries e qrels do cache local, baixando-os se necessário.

    Args:
        dir_raw: Diretório com ``corpus.parquet``, ``queries.parquet`` e ``qrels.parquet``.

    Returns:
        Tupla ``(df_corpus, df_queries, df_qrels)``.
    """
    faltando = [
        nome for nome in NOMES_CONJUNTOS
        if not os.path.exists(os.path.join(dir_raw, f"{nome}.parquet"))
    ]
    if faltando:
        baixar_dataset(dir_raw)
    df_corpus, df_queries, df_qrels = (
        pd.read_parquet(os.path.join(dir_raw, f"{nome}.parquet")) for nome in NOMES_CONJUNTOS
    )
    return df_corpus, df_queries, df_qrels


def montar_texto_documento(titulo: str | None, texto: str | None) -> str:
    """Concatena título e texto de um documento, tratando valores vazios ou nulos.

    Args:
        titulo: Título do documento (pode ser ``None`` ou vazio).
        texto: Corpo do documento (pode ser ``None``).

    Returns:
        ``titulo + " " + texto`` sem espaços sobrando nas pontas.
    """
    partes = [p.strip() for p in (titulo, texto) if isinstance(p, str) and p.strip()]
    return " ".join(partes)


def montar_documentos(df_corpus: pd.DataFrame) -> dict[str, str]:
    """Converte o corpus em um dicionário ``_id`` -> texto do documento.

    Args:
        df_corpus: DataFrame com as colunas ``_id``, ``title`` e ``text``.

    Returns:
        Dicionário na ordem do corpus.
    """
    return {
        str(_id): montar_texto_documento(titulo, texto)
        for _id, titulo, texto in zip(df_corpus["_id"], df_corpus["title"], df_corpus["text"])
    }


def gerar_titulo(texto: str, limite: int = TAMANHO_TITULO) -> str:
    """Gera um título curto para exibição a partir do início do texto.

    O ``title`` do faq-bacen está vazio em todos os documentos, então a demo usa os
    primeiros caracteres do texto no lugar dele.

    Args:
        texto: Texto completo do documento.
        limite: Número máximo de caracteres do título.

    Returns:
        Início do texto com espaços normalizados, seguido de reticências se foi cortado.
    """
    limpo = " ".join(texto.split())
    if len(limpo) <= limite:
        return limpo
    return limpo[:limite].rstrip() + "..."


def salvar_documentos_txt(
    documentos: dict[str, str], dir_documentos: str = DIR_DOCUMENTOS
) -> int:
    """Grava cada documento em ``<dir_documentos>/<_id>.txt`` (uso da busca linear).

    Args:
        documentos: Dicionário ``_id`` -> texto.
        dir_documentos: Diretório de destino.

    Returns:
        Quantidade de arquivos gravados.
    """
    os.makedirs(dir_documentos, exist_ok=True)
    for _id, texto in documentos.items():
        with open(os.path.join(dir_documentos, f"{_id}.txt"), "w", encoding="utf-8") as arq:
            arq.write(texto)
    return len(documentos)


def ordem_embaralhada(ids: list[str], semente: int = SEMENTE) -> list[str]:
    """Embaralha os ids uma única vez com semente fixa.

    Os subconjuntos do benchmark são prefixos dessa ordem, logo o de 100 documentos
    está dentro do de 500, que está dentro do de 1.000.

    Args:
        ids: Lista de ids do corpus.
        semente: Semente do gerador aleatório.

    Returns:
        Nova lista com os mesmos ids, embaralhada.
    """
    gerador = np.random.default_rng(semente)
    return [str(i) for i in gerador.permutation(ids)]


def amostrar_documentos(
    documentos: dict[str, str], n: int, semente: int = SEMENTE
) -> dict[str, str]:
    """Retorna o subconjunto aninhado de tamanho ``n`` do corpus.

    Args:
        documentos: Corpus completo (``_id`` -> texto).
        n: Tamanho do subconjunto.
        semente: Semente da ordem embaralhada.

    Returns:
        Dicionário com os ``n`` primeiros ids da ordem embaralhada.
    """
    ordem = ordem_embaralhada(list(documentos), semente)
    return {_id: documentos[_id] for _id in ordem[:n]}
