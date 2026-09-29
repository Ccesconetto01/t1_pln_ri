"""Testes da normalização, tokenização e remoção de stopwords."""

from src.preprocessamento import STOP_WORDS, normalizar_texto, tokenizar_e_filtrar


def test_normalizar_texto_remove_acentos_e_converte_minusculas() -> None:
    assert normalizar_texto("Crédito NÃO Consignado") == "credito nao consignado"


def test_tokenizar_remove_pontuacao() -> None:
    assert tokenizar_e_filtrar("Empréstimo, financiamento; leasing!") == [
        "emprestimo", "financiamento", "leasing"
    ]


def test_tokenizar_remove_tokens_so_de_digitos() -> None:
    tokens = tokenizar_e_filtrar("Limite de 25000 reais em 2024")
    assert "25000" not in tokens
    assert "2024" not in tokens
    assert tokens == ["limite", "reais"]


def test_tokenizar_mantem_tokens_com_digitos_e_letras() -> None:
    assert tokenizar_e_filtrar("Resolução CMN4966") == ["resolucao", "cmn4966"]


def test_stopwords_sao_removidas_mesmo_sem_acento() -> None:
    # "não" é stopword; o texto normalizado vira "nao" e também precisa ser removido.
    assert "nao" in STOP_WORDS
    assert tokenizar_e_filtrar("O cliente não pode") == ["cliente", "pode"]
    assert tokenizar_e_filtrar("O cliente nao pode") == ["cliente", "pode"]


def test_tokenizar_texto_vazio_ou_so_stopwords() -> None:
    assert tokenizar_e_filtrar("") == []
    assert tokenizar_e_filtrar("   ") == []
    assert tokenizar_e_filtrar("de a o em") == []
