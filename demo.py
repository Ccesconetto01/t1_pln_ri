"""Demo interativa no terminal: busca nos cinco modelos sobre o FAQ do Banco Central.

Uso, a partir da raiz do projeto::

    python demo.py

Na primeira execução o dataset é baixado para ``data/raw/``; depois disso a demo funciona
sem internet.
"""

import os
import sys
import time

from src.buscador import MODELOS, ModeloBusca, indexar_todos
from src.dados import (
    DIR_DOCUMENTOS,
    carregar_dados,
    gerar_titulo,
    montar_documentos,
    salvar_documentos_txt,
)

TOP_K = 5
TAMANHO_TRECHO = 150
OPCAO_TODOS = "todos"
COMANDO_SAIR = "sair"
COMANDO_MODELO = ":modelo"
MODELOS_SEM_RANKING = ("Busca Linear", "Booleano")


def preparar_documentos() -> dict[str, str]:
    """Carrega o corpus do cache local e garante os ``.txt`` da busca linear.

    Returns:
        Dicionário ``_id`` -> texto do documento.
    """
    df_corpus, _, _ = carregar_dados()
    documentos = montar_documentos(df_corpus)
    if os.path.isdir(DIR_DOCUMENTOS):
        gravados = sum(1 for nome in os.listdir(DIR_DOCUMENTOS) if nome.endswith(".txt"))
    else:
        gravados = 0
    if gravados != len(documentos):
        salvar_documentos_txt(documentos)
    return documentos


def escolher_modelos() -> list[str] | None:
    """Pergunta ao usuário quais modelos usar nas buscas.

    Returns:
        Lista de nomes de modelos, ou ``None`` se o usuário digitou ``sair``.
    """
    nomes = list(MODELOS)
    print("\nModelos disponíveis:")
    for numero, nome in enumerate(nomes, start=1):
        print(f"  {numero}) {nome}")
    print(f"  {len(nomes) + 1}) Todos")
    while True:
        resposta = input("Escolha um modelo (número ou nome; Enter = todos): ").strip()
        if resposta.lower() == COMANDO_SAIR:
            return None
        if resposta == "" or resposta.lower() in (OPCAO_TODOS, str(len(nomes) + 1)):
            return nomes
        if resposta.isdigit() and 1 <= int(resposta) <= len(nomes):
            return [nomes[int(resposta) - 1]]
        correspondentes = [nome for nome in nomes if nome.lower() == resposta.lower()]
        if correspondentes:
            return correspondentes
        print("Opção inválida. Tente de novo.")


def formatar_trecho(texto: str, limite: int = TAMANHO_TRECHO) -> str:
    """Encurta o texto para exibição, normalizando os espaços.

    Args:
        texto: Texto completo do documento.
        limite: Número máximo de caracteres.

    Returns:
        Texto com até ``limite`` caracteres, com reticências se foi cortado.
    """
    limpo = " ".join(texto.split())
    if len(limpo) <= limite:
        return limpo
    return limpo[:limite].rstrip() + "..."


def formatar_total(total: int) -> str:
    """Formata uma quantidade com ponto como separador de milhar (ex.: 1.673).

    Args:
        total: Quantidade de documentos.

    Returns:
        Número formatado no padrão brasileiro.
    """
    return f"{total:,}".replace(",", ".")


def exibir_resultados(
    nome: str, resultados: list[tuple[str, float]], documentos: dict[str, str]
) -> None:
    """Imprime o Top-K de um modelo e, embaixo, o total de documentos encontrados.

    Args:
        nome: Nome do modelo.
        resultados: **Todos** os pares ``(_id, score)`` do modelo, em ordem decrescente de
            score. Só os ``TOP_K`` primeiros são exibidos; o total conta a lista inteira.
        documentos: Corpus (``_id`` -> texto), usado para título e trecho.
    """
    print(f"\n=== {nome} ===")
    if not resultados:
        print("  Nenhum documento encontrado.")
        if nome in MODELOS_SEM_RANKING:
            print("  (este modelo exige todos os termos; tente menos palavras)")
    for posicao, (_id, score) in enumerate(resultados[:TOP_K], start=1):
        texto = documentos[_id]
        print(f"  {posicao}. [{_id}] {gerar_titulo(texto)}  (score {score:.4f})")
        print(f"     {formatar_trecho(texto)}")
    print(f"  Total de documentos encontrados: {formatar_total(len(resultados))}")


def executar_busca(
    modelos: dict[str, ModeloBusca],
    escolhidos: list[str],
    consulta: str,
    documentos: dict[str, str],
) -> None:
    """Roda a consulta nos modelos escolhidos, isolando falhas de cada modelo.

    Cada modelo mostra o seu Top-5 e o seu total. Quando mais de um modelo responde, no
    final aparece o total geral: quantos documentos distintos foram encontrados por pelo
    menos um deles (um mesmo documento achado por vários modelos conta uma só vez).

    Args:
        modelos: Modelos já indexados.
        escolhidos: Nomes dos modelos que devem responder.
        consulta: Texto digitado pelo usuário.
        documentos: Corpus, usado para exibir os resultados.
    """
    encontrados: set[str] = set()
    for nome in escolhidos:
        try:
            # Pede todos os resultados (não só o Top-5) para poder contar o total.
            resultados = modelos[nome].buscar(consulta, k=len(documentos))
            exibir_resultados(nome, resultados, documentos)
            encontrados.update(_id for _id, _ in resultados)
        except Exception as erro:  # a demo não pode cair por causa de uma busca
            print(f"\n=== {nome} ===")
            print(f"  Não foi possível buscar neste modelo ({type(erro).__name__}: {erro}).")
    if len(escolhidos) > 1:
        print("\n=== Total geral ===")
        print(f"  Documentos distintos encontrados: {formatar_total(len(encontrados))}")


def laco_de_consultas(
    modelos: dict[str, ModeloBusca], documentos: dict[str, str]
) -> None:
    """Lê consultas do usuário até ele digitar ``sair``.

    Args:
        modelos: Modelos já indexados.
        documentos: Corpus, usado para exibir os resultados.
    """
    escolhidos = escolher_modelos()
    if escolhidos is None:
        return
    print(f"\nDigite uma consulta, '{COMANDO_MODELO}' para trocar de modelo "
          f"ou '{COMANDO_SAIR}' para encerrar.")
    if "Booleano" in escolhidos:
        print("No Booleano, use AND, OR e NOT em maiúsculas (ex.: pix AND limite).")
    while True:
        consulta = input("\nConsulta> ").strip()
        if consulta.lower() == COMANDO_SAIR:
            return
        if consulta.lower() == COMANDO_MODELO:
            novos = escolher_modelos()
            if novos is None:
                return
            escolhidos = novos
            continue
        if not consulta:
            print("Consulta vazia. Digite algum termo.")
            continue
        executar_busca(modelos, escolhidos, consulta, documentos)


def main() -> int:
    """Indexa o corpus completo uma vez e abre o laço interativo.

    Returns:
        Código de saída do processo (0 = sucesso).
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # evita traceback em consoles sem UTF-8
    print("Demo — Busca no FAQ do Banco Central do Brasil")
    try:
        documentos = preparar_documentos()
        print(f"Corpus carregado: {len(documentos)} documentos. Indexando os 5 modelos...")
        inicio = time.perf_counter()
        modelos, tempos_ms = indexar_todos(documentos)
    except KeyboardInterrupt:
        print("\nInterrompido.")
        return 130
    except Exception as erro:
        print(f"Não foi possível preparar o corpus ({type(erro).__name__}: {erro}).")
        print("Na primeira execução é preciso ter internet (para baixar o dataset) e as")
        print("dependências instaladas: pip install -r requirements.txt")
        return 1
    print(f"Indexação concluída em {time.perf_counter() - inicio:.2f} s:")
    for nome, tempo in tempos_ms.items():
        print(f"  {nome:<13} {tempo:>9.1f} ms")
    try:
        laco_de_consultas(modelos, documentos)
    except (KeyboardInterrupt, EOFError):
        print()
    print("Até logo!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
