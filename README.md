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

[`MTEB-BR/faq-bacen`](https://huggingface.co/datasets/MTEB-BR/faq-bacen) no HuggingFace,
licença **Apache-2.0**: perguntas e respostas do FAQ público do Banco Central do Brasil.

| Conjunto | Tamanho | Observação |
|---|---:|---|
| `corpus` | 1.673 documentos | as respostas do FAQ; `title` vazio em todos, só `text` é usado |
| `queries` | 373 perguntas | usadas na avaliação de relevância e no benchmark |
| `qrels` | 373 pares | todos com `score = 1`: **exatamente 1 documento relevante por pergunta** |

O download é feito por `src/dados.py` na primeira execução e fica em cache em `data/raw/`
(parquets). A Busca Linear lê os documentos de `data/processed/documentos/<_id>.txt`, gerados
automaticamente. As duas pastas estão no `.gitignore`.

## Instalação

Requer Python 3.13 (versão usada no desenvolvimento). A partir da raiz do projeto:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

Na primeira execução é preciso ter internet, para baixar o dataset do HuggingFace e as
stopwords em português do nltk. Sem internet, o pré-processamento usa uma lista de stopwords
de reserva (`STOP_WORDS_FALLBACK`).

## Arquitetura

```text
src/
├── preprocessamento.py  # tokenizar_e_filtrar: minúsculas, sem acentos, sem stopwords/números
├── dados.py             # download, cache, montagem de documentos/consultas/qrels, amostragem
├── busca_linear.py      # Busca Linear: relê e tokeniza cada .txt do disco a cada consulta
├── modelo_booleano.py   # Booleano: matriz de incidência (numpy bool) + AND/OR/NOT bitwise
├── modelo_vetorial.py   # Vetorial: TfidfVectorizer (TF sublinear) + cosseno
├── modelo_bm25.py       # BM25: rank_bm25.BM25Okapi, k1 = 1,5 e b = 0,75 por padrão
├── modelo_lsa.py        # LSA: TruncatedSVD (até 100 componentes) sobre a matriz TF-IDF
├── buscador.py          # interface comum (Protocol ModeloBusca) e fábrica dos 5 modelos
└── avaliacao.py         # MRR@k e Recall@k sobre os qrels
benchmark/               # scripts de latência, gráficos e relevância; saídas em resultados/
notebooks/               # análise de relevância com as consultas desafiadoras
tests/                   # pytest com mini-corpus fixo (conftest.py), sem internet
demo.py                  # demo interativa no terminal
```

Os cinco modelos seguem a mesma interface, definida pelo Protocol `ModeloBusca` em
`src/buscador.py`:

```python
modelo.indexar(documentos)            # documentos: dict[_id, texto]
modelo.buscar(consulta, k) -> list[tuple[_id, score]]   # ordem decrescente de score
```

A demo, o benchmark e o notebook criam os modelos apenas por `criar_modelo(nome)` e
`indexar_todos(documentos)`, sem importar as classes diretamente. Todos usam o mesmo
`tokenizar_e_filtrar`, então as diferenças de resultado vêm só do modelo, e não do
pré-processamento. Busca Linear e Booleano não ranqueiam: devolvem score fixo 1,0 na ordem
do corpus.

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

### Benchmark de latência e relevância

```bash
python -m benchmark.benchmark_latencia      # gera latencia.csv (leva alguns minutos)
python -m benchmark.graficos_latencia       # gera os 2 PNGs a partir do CSV
python -m benchmark.avaliacao_relevancia    # gera relevancia.csv
```

As saídas ficam em `benchmark/resultados/`. Metodologia da latência:

- **N = 100, 500, 1.000 e 1.673.** O último é o **tamanho real do corpus**: não há como medir
  acima disso sem inventar documentos. Os outros três cobrem uma ordem de grandeza abaixo dele,
  o suficiente para ver a forma da curva. Os subconjuntos são aninhados (100 ⊂ 500 ⊂ 1.000 ⊂
  1.673), sorteados uma vez com semente 42.
- **50 consultas** sorteadas das 373 (semente 42), as mesmas para todo N e todo modelo.
- **Indexação:** mediana de 3 indexações, com um modelo novo a cada vez.
- **Consulta:** 3 consultas de aquecimento; depois cada uma das 50 roda 3 vezes com
  `time.perf_counter()` (150 medições por modelo e N).

Os valores absolutos dependem da máquina. O que importa é como cada modelo cresce com N.

### Notebooks

`notebooks/02_analise_relevancia.ipynb` compara Vetorial, BM25 e LSA lado a lado em cinco
consultas desafiadoras (termo frequente, termo raro, sinônimo, polissemia e termo fora do
vocabulário) e mostra as métricas. Ele já está salvo com as saídas executadas. Para rodar de
novo, abra no VS Code (ou no Jupyter) e escolha o kernel do `.venv` do projeto; o `ipykernel`
já está no `requirements.txt`. O notebook muda sozinho para a raiz do projeto.

## Resultados

_A preencher nas etapas 6 e 7._

## Limitações

_A preencher na etapa 8._
