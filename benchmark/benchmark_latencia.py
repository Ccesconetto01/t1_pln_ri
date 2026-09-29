"""Benchmark de latência: tempo de indexação e de consulta dos cinco modelos em função de N.

Uso, a partir da raiz do projeto::

    python -m benchmark.benchmark_latencia

Os subconjuntos de documentos são aninhados (100 ⊂ 500 ⊂ 1.000 ⊂ 1.673) e as 50 consultas
sorteadas são as mesmas para todos os modelos e todos os tamanhos.
"""

import os
import statistics
import time

import pandas as pd

from src.buscador import MODELOS, criar_modelo
from src.dados import (
    DIR_DOCUMENTOS,
    SEMENTE,
    amostrar_documentos,
    carregar_dados,
    montar_consultas,
    montar_documentos,
    ordem_embaralhada,
    salvar_documentos_txt,
)

TAMANHOS_N = [100, 500, 1000, 1673]  # 1.673 = tamanho total do corpus faq-bacen
N_CONSULTAS = 50
N_AQUECIMENTO = 3
REPETICOES = 3
K = 10
DIR_RESULTADOS = os.path.join("benchmark", "resultados")
ARQUIVO_CSV = os.path.join(DIR_RESULTADOS, "latencia.csv")
COLUNAS = [
    "modelo",
    "n_documentos",
    "tempo_indexacao_ms",
    "tempo_medio_ms",
    "tempo_mediana_ms",
    "desvio_padrao_ms",
]


def sortear_consultas(consultas: dict[str, str], n: int = N_CONSULTAS) -> list[str]:
    """Sorteia ``n`` consultas com a semente fixa do projeto.

    Args:
        consultas: Dicionário ``query-id`` -> texto.
        n: Quantidade de consultas sorteadas.

    Returns:
        Textos das consultas sorteadas.
    """
    ids = ordem_embaralhada(list(consultas), SEMENTE)[:n]
    return [consultas[query_id] for query_id in ids]


def medir_modelo(
    nome: str, documentos: dict[str, str], consultas: list[str]
) -> dict[str, float | int | str]:
    """Indexa um modelo e cronometra cada consulta ``REPETICOES`` vezes.

    A indexação também é repetida ``REPETICOES`` vezes (com um modelo novo a cada vez) e
    o tempo registrado é a mediana, para reduzir o ruído de uma medição única.

    Args:
        nome: Nome do modelo em ``MODELOS``.
        documentos: Subconjunto do corpus a indexar.
        consultas: Consultas cronometradas.

    Returns:
        Linha do CSV com os tempos em milissegundos.
    """
    tempos_indexacao_ms: list[float] = []
    for _ in range(REPETICOES):
        modelo = criar_modelo(nome)
        inicio = time.perf_counter()
        modelo.indexar(documentos)
        tempos_indexacao_ms.append((time.perf_counter() - inicio) * 1000)
    tempo_indexacao_ms = statistics.median(tempos_indexacao_ms)

    for consulta in consultas[:N_AQUECIMENTO]:
        modelo.buscar(consulta, K)

    tempos_ms: list[float] = []
    for consulta in consultas:
        for _ in range(REPETICOES):
            inicio = time.perf_counter()
            modelo.buscar(consulta, K)
            tempos_ms.append((time.perf_counter() - inicio) * 1000)

    return {
        "modelo": nome,
        "n_documentos": len(documentos),
        "tempo_indexacao_ms": round(tempo_indexacao_ms, 4),
        "tempo_medio_ms": round(statistics.mean(tempos_ms), 4),
        "tempo_mediana_ms": round(statistics.median(tempos_ms), 4),
        "desvio_padrao_ms": round(statistics.stdev(tempos_ms), 4),
    }


def main() -> None:
    """Roda o benchmark para todos os N e modelos e grava ``latencia.csv``."""
    df_corpus, df_queries, _ = carregar_dados()
    documentos = montar_documentos(df_corpus)
    salvar_documentos_txt(documentos, DIR_DOCUMENTOS)  # a Busca Linear lê do disco
    consultas = sortear_consultas(montar_consultas(df_queries))

    linhas = []
    for n in TAMANHOS_N:
        subconjunto = amostrar_documentos(documentos, n, SEMENTE)
        for nome in MODELOS:
            linha = medir_modelo(nome, subconjunto, consultas)
            linhas.append(linha)
            print(
                f"N={n:>5}  {nome:<13} indexação {linha['tempo_indexacao_ms']:>10.2f} ms"
                f"  consulta média {linha['tempo_medio_ms']:>9.3f} ms"
            )

    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    pd.DataFrame(linhas, columns=COLUNAS).to_csv(ARQUIVO_CSV, index=False)
    print(f"\nResultados gravados em {ARQUIVO_CSV}")


if __name__ == "__main__":
    main()
