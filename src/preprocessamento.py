"""Pré-processamento de texto, compartilhado pelos cinco modelos."""

import re
import unicodedata

import nltk
from nltk.corpus import stopwords

# Usada apenas se o nltk não conseguir baixar as stopwords (ex.: sem internet).
STOP_WORDS_FALLBACK = (
    "a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sob sobre "
    "e ou mas que se como quando onde qual quais quem ao aos ate entre ja tambem mais menos "
    "muito muitos pouco ser estar ter haver foi era sao esta estao tem tinha ha isso isto "
    "esse essa esses essas este esta estes estas aquele aquela seu sua seus suas meu minha "
    "nao nem so pelo pela pelos pelas num numa lhe lhes me te nos vos eu tu ele ela eles elas"
).split()


def normalizar_texto(texto: str) -> str:
    """Converte para minúsculas e remove acentos.

    Args:
        texto: Texto original.

    Returns:
        Texto em minúsculas, sem os diacríticos (ex.: ``"Crédito"`` -> ``"credito"``).
    """
    decomposto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in decomposto if unicodedata.category(c) != "Mn")


def _carregar_stop_words() -> frozenset[str]:
    """Carrega as stopwords em português do nltk, já sem acentos.

    Se a lista não estiver instalada, tenta baixá-la. Sem internet, usa
    ``STOP_WORDS_FALLBACK``.

    Returns:
        Conjunto de stopwords normalizadas com ``normalizar_texto``.
    """
    try:
        try:
            lista = stopwords.words("portuguese")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            lista = stopwords.words("portuguese")
    except (LookupError, OSError):
        lista = STOP_WORDS_FALLBACK
    return frozenset(normalizar_texto(palavra) for palavra in lista)


STOP_WORDS = _carregar_stop_words()


def tokenizar_e_filtrar(texto: str) -> list[str]:
    """Normaliza o texto e retorna os tokens sem stopwords.

    Args:
        texto: Texto de um documento ou de uma consulta.

    Returns:
        Tokens em minúsculas e sem acentos, na ordem em que aparecem no texto (com
        repetições). Stopwords e tokens só com dígitos são descartados.
    """
    tokens = re.findall(r"\b\w+\b", normalizar_texto(texto))
    return [t for t in tokens if not t.isdigit() and t not in STOP_WORDS]
