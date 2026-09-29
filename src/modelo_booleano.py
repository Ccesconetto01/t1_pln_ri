"""Modelo Booleano: matriz de incidência binária com operações bitwise."""

import numpy as np

from src.preprocessamento import tokenizar_e_filtrar

OPERADORES = ("AND", "OR", "NOT")


class ModeloBooleano:
    """Recuperação booleana sobre uma matriz de incidência ``termo -> vetor de bool``.

    Cada termo do vocabulário guarda um vetor booleano de ``N`` posições (uma por
    documento). As consultas são resolvidas com ``&``, ``|`` e ``~`` do numpy sobre esses
    vetores.

    Attributes:
        ids: Ids dos documentos; a posição ``i`` dos vetores corresponde a ``ids[i]``.
        matriz_incidencia: Dicionário termo -> vetor booleano de presença nos documentos.
    """

    def __init__(self) -> None:
        """Inicializa o modelo sem índice."""
        self.ids: list[str] = []
        self.matriz_incidencia: dict[str, np.ndarray] = {}

    def indexar(self, documentos: dict[str, str]) -> None:
        """Constrói a matriz de incidência: um vetor booleano de N posições por termo.

        Args:
            documentos: Dicionário ``_id`` -> texto do documento.
        """
        self.ids = list(documentos)
        n_documentos = len(self.ids)
        self.matriz_incidencia = {}
        for posicao, _id in enumerate(self.ids):
            for termo in set(tokenizar_e_filtrar(documentos[_id])):
                vetor = self.matriz_incidencia.get(termo)
                if vetor is None:
                    vetor = self.matriz_incidencia[termo] = np.zeros(n_documentos, dtype=bool)
                vetor[posicao] = True

    def _vetor_do_termo(self, palavra: str) -> np.ndarray | None:
        """Retorna o vetor de uma palavra da consulta (AND se ela gerar vários tokens).

        Args:
            palavra: Uma palavra da consulta, antes do pré-processamento.

        Returns:
            Vetor booleano de ``N`` posições com os documentos que contêm a palavra (todo
            ``False`` se ela estiver fora do vocabulário), ou ``None`` se a palavra for
            descartada no pré-processamento (stopword, número ou pontuação).
        """
        tokens = tokenizar_e_filtrar(palavra)
        if not tokens:
            return None
        vazio = np.zeros(len(self.ids), dtype=bool)
        resultado = np.ones(len(self.ids), dtype=bool)
        for token in tokens:
            # Termo fora do vocabulário = vetor todo False.
            resultado = resultado & self.matriz_incidencia.get(token, vazio)
        return resultado

    def buscar(self, consulta: str, k: int = 10) -> list[tuple[str, float]]:
        """Avalia a consulta booleana da esquerda para a direita.

        Os operadores ``AND``, ``OR`` e ``NOT`` devem estar em maiúsculas. Entre dois termos
        sem operador vale ``AND``. Não há precedência nem parênteses: ``a OR b AND c`` é
        avaliado como ``(a OR b) AND c``.

        Args:
            consulta: Consulta booleana, ex.: ``"pix AND limite"`` ou ``"cheque NOT especial"``.
            k: Quantidade máxima de resultados.

        Returns:
            Até ``k`` pares ``(_id, 1.0)`` na ordem do corpus: o modelo não ranqueia, então
            o score é fixo. Lista vazia se a consulta não tiver nenhum termo válido.
        """
        resultado: np.ndarray | None = None
        operador = "AND"
        negar = False
        for palavra in consulta.split():
            if palavra == "NOT":
                negar = not negar
            elif palavra in ("AND", "OR"):
                operador = palavra
            else:
                vetor = self._vetor_do_termo(palavra)
                if vetor is None:
                    continue
                if negar:
                    vetor = ~vetor
                    negar = False
                if resultado is None:
                    resultado = vetor
                elif operador == "AND":
                    resultado = resultado & vetor
                else:
                    resultado = resultado | vetor
                operador = "AND"
        if resultado is None:
            return []
        posicoes = np.flatnonzero(resultado)[:k]
        return [(self.ids[p], 1.0) for p in posicoes]
