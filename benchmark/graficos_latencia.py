"""Gráficos de escalabilidade a partir de ``benchmark/resultados/latencia.csv``.

Uso, a partir da raiz do projeto (depois de rodar o benchmark)::

    python -m benchmark.graficos_latencia
"""

import os

import matplotlib

matplotlib.use("Agg")  # gera os PNGs sem abrir janela

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark.benchmark_latencia import ARQUIVO_CSV, DIR_RESULTADOS  # noqa: E402

# Cor e marcador fixos por modelo: a identidade não depende só da cor.
ESTILOS = {
    "Busca Linear": ("#2a78d6", "o"),
    "Booleano": ("#eb6834", "s"),
    "Vetorial": ("#1baf7a", "^"),
    "BM25": ("#eda100", "D"),
    "LSA": ("#e87ba4", "v"),
}
COR_TEXTO = "#52514e"
COR_GRADE = "#d9d8d4"
LIMIAR_ROTULO = 0.05  # fração do maior tempo abaixo da qual não há rótulo direto (linear)


def plotar_escalabilidade(tabela: pd.DataFrame, escala_log: bool, destino: str) -> None:
    """Desenha o tempo médio de consulta por N, uma linha por modelo, e salva o PNG.

    Args:
        tabela: Conteúdo de ``latencia.csv``.
        escala_log: Se ``True``, usa eixo Y logarítmico.
        destino: Caminho do PNG gerado.
    """
    figura, eixo = plt.subplots(figsize=(10, 6))
    maximo = tabela["tempo_medio_ms"].max()
    for modelo, (cor, marcador) in ESTILOS.items():
        serie = tabela[tabela["modelo"] == modelo].sort_values("n_documentos")
        eixo.plot(
            serie["n_documentos"], serie["tempo_medio_ms"],
            color=cor, marker=marcador, markersize=8, linewidth=2, label=modelo,
        )
        ultimo = serie.iloc[-1]
        # Na escala linear, as séries coladas no zero ficariam com rótulos sobrepostos;
        # elas são identificadas só pela legenda.
        if not escala_log and ultimo["tempo_medio_ms"] < LIMIAR_ROTULO * maximo:
            continue
        eixo.annotate(
            modelo, (ultimo["n_documentos"], ultimo["tempo_medio_ms"]),
            xytext=(8, 0), textcoords="offset points", va="center", color=COR_TEXTO,
        )
    if escala_log:
        eixo.set_yscale("log")
    escala = "logarítmica" if escala_log else "linear"
    eixo.set_title(f"Tempo médio de consulta × tamanho do corpus (escala {escala})")
    eixo.set_xlabel("Número de documentos (N)")
    eixo.set_ylabel("Tempo médio por consulta (ms)")
    eixo.set_xticks(sorted(tabela["n_documentos"].unique()))
    eixo.set_xlim(right=tabela["n_documentos"].max() * 1.15)  # espaço para os rótulos
    eixo.grid(True, linestyle="--", color=COR_GRADE, linewidth=0.8)
    eixo.legend(loc="upper left", frameon=False)
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    figura.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(figura)


def main() -> None:
    """Lê o CSV do benchmark e gera os gráficos em escala linear e logarítmica."""
    tabela = pd.read_csv(ARQUIVO_CSV)
    plotar_escalabilidade(
        tabela, False, os.path.join(DIR_RESULTADOS, "escalabilidade_linear.png")
    )
    plotar_escalabilidade(tabela, True, os.path.join(DIR_RESULTADOS, "escalabilidade_log.png"))
    print(f"Gráficos gravados em {DIR_RESULTADOS}")


if __name__ == "__main__":
    main()
