# T1 — Benchmark de Modelos Clássicos de Recuperação da Informação

Trabalho Prático 1 da disciplina de PLN e Sistemas de RI (FUCAPE).

Implementa, compara e avalia cinco algoritmos clássicos de recuperação da informação sobre o
FAQ público do Banco Central do Brasil (domínio de finanças e regulação bancária):

1. Busca Linear Ingênua (baseline O(N), sem índice)
2. Modelo Booleano (matriz de incidência com operações bitwise)
3. Modelo Vetorial (TF-IDF + similaridade do cosseno)
4. Okapi BM25 (`k1` e `b` parametrizáveis)
5. LSA (`TruncatedSVD` sobre a matriz TF-IDF)

## Dataset

_A preencher na etapa 1._

## Instalação

_A preencher._

## Arquitetura

_A preencher._

## Como executar

### Demo

A demo é interativa e roda no terminal, a partir da raiz do projeto:

```bash
python demo.py
```

Na primeira execução ela baixa o dataset para `data/raw/` (é preciso ter internet); depois
disso funciona offline. Ao iniciar, ela indexa os cinco modelos sobre o corpus completo e mostra
o tempo de cada indexação. Em seguida:

- escolha um modelo (número ou nome) ou **Todos** (Enter);
- digite a consulta: a demo mostra o Top-5 de cada modelo com posição, trecho do documento e score;
- no modelo **Booleano** use `AND`, `OR` e `NOT` em maiúsculas (ex.: `pix AND limite`); sem
  operadores, todos os termos são exigidos (o mesmo vale para a **Busca Linear**);
- digite `:modelo` para trocar de modelo e `sair` (ou `Ctrl+C`) para encerrar.

### Testes

```bash
pytest -q
```

Os testes usam um mini-corpus fixo (`tests/conftest.py`) e não dependem de internet nem do
dataset.

### Benchmark de latência

_A preencher na etapa 6._

### Notebooks

_A preencher na etapa 7._

## Resultados

_A preencher nas etapas 6 e 7._

## Limitações

_A preencher na etapa 8._
