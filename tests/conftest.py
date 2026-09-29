"""Fixtures compartilhadas: mini-corpus fixo, sem dependência de internet ou do dataset."""

import os

import pytest

MINI_CORPUS = {
    "d1": "O Pix é um meio de pagamento instantâneo com limite noturno para transferências.",
    "d2": "Empréstimo consignado tem juros menores que o cartão de crédito rotativo.",
    "d3": "O cliente pode revogar o consentimento no Open Finance a qualquer momento.",
    "d4": "Boleto bancário pode ser pago em lotéricas e também via Pix.",
    "d5": "Financiamento imobiliário exige garantia de alienação fiduciária do imóvel.",
}


@pytest.fixture
def documentos() -> dict[str, str]:
    """Mini-corpus de 5 documentos (``_id`` -> texto)."""
    return dict(MINI_CORPUS)


@pytest.fixture
def dir_documentos(tmp_path, documentos: dict[str, str]) -> str:
    """Grava o mini-corpus em arquivos ``.txt`` temporários (usados pela busca linear)."""
    for _id, texto in documentos.items():
        with open(os.path.join(tmp_path, f"{_id}.txt"), "w", encoding="utf-8") as arquivo:
            arquivo.write(texto)
    return str(tmp_path)
