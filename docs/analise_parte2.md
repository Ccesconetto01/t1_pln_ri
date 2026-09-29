# Análise da Parte 2: latência e relevância

Todos os números abaixo saem de `benchmark/resultados/latencia.csv` e
`benchmark/resultados/relevancia.csv`, gerados por:

```bash
python -m benchmark.benchmark_latencia     # latencia.csv
python -m benchmark.graficos_latencia      # escalabilidade_linear.png e escalabilidade_log.png
python -m benchmark.avaliacao_relevancia   # relevancia.csv
```

Os valores absolutos dependem da máquina; o que importa é como cada modelo cresce com N.
Máquina da medição: Windows 11 e Python 3.13.2 no `.venv` do projeto.

## Dataset conferido

| Item | Valor |
|---|---|
| Documentos (`corpus`) | 1.673; `title` vazio em todos |
| Consultas (`queries`) | 373 |
| Qrels | 373 pares, todos com `score = 1`, **exatamente 1 documento relevante por consulta** |
| Stopwords | 200, da lista do nltk (não caiu no fallback) |

Como cada consulta tem um único relevante, o **Recall@10 equivale à taxa de acerto no Top-10**
(a fração de consultas cujo documento relevante aparece entre os 10 primeiros).

## Metodologia do benchmark

- **N = 100, 500, 1.000 e 1.673.** O último é o corpus inteiro, então não há como medir
  acima disso. Os subconjuntos são aninhados (`amostrar_documentos`, semente 42).
- **Consultas:** 50 perguntas sorteadas das 373 (semente 42), as mesmas para todo N e
  todo modelo.
- **Indexação:** medida separada das consultas, repetida 3 vezes com um modelo novo a cada
  vez; o CSV registra a mediana.
- **Consultas:** 3 de aquecimento sem cronometrar; depois cada uma das 50 é executada 3 vezes
  com `time.perf_counter()`. Média, mediana e desvio padrão vêm dessas 150 medições.
- Todos os modelos são criados por `src/buscador.py` e usam o mesmo `tokenizar_e_filtrar`.

## Latência de consulta: teórica × observada

Tempo médio por consulta (ms), com o fator de crescimento entre N = 100 e N = 1.673
(N aumenta **16,7×**):

| Modelo | N=100 | N=500 | N=1.000 | N=1.673 | Crescimento |
|---|---:|---:|---:|---:|---:|
| Busca Linear | 20,64 | 84,95 | 177,44 | 300,55 | 14,6× |
| Booleano | 0,040 | 0,037 | 0,038 | 0,040 | 1,0× |
| Vetorial | 0,68 | 0,92 | 0,95 | 1,16 | 1,7× |
| BM25 | 0,12 | 0,46 | 0,89 | 1,78 | 14,9× |
| LSA | 1,52 | 3,57 | 5,10 | 6,36 | 4,2× |

| Modelo | Complexidade teórica da consulta | O que foi observado |
|---|---|---|
| Busca Linear | O(N · L): lê e tokeniza do disco cada documento de tamanho médio L | Cresce 14,6× para N 16,7× maior, ou seja, **linear**. É de longe o mais lento: ~300 ms com o corpus inteiro, dominado por I/O e tokenização |
| Booleano | O(\|q\| · N) em AND bit a bit sobre vetores de N posições | **Praticamente constante** até 1.673. A operação é linear em N, mas vetorizada no numpy sobre N ≤ 1.673 booleanos, então o custo fixo do Python domina |
| Vetorial | O(\|q\|) para vetorizar e O(nnz) no cosseno esparso contra todos os documentos | Cresce só 1,7×: o custo fixo do `transform` do scikit-learn domina nesse tamanho. A parte que depende de N existe, mas ainda é pequena |
| BM25 | O(\|q\| · N): `rank_bm25.get_scores` calcula o score de **todos** os documentos para cada termo | Cresce 14,9×, **quase exatamente linear**. Com N = 1.000 ele empata com o Vetorial e com N = 1.673 já fica mais lento (1,78 × 1,16 ms) |
| LSA | O(\|q\| · k) para projetar a consulta e O(N · k) no cosseno denso (k ≤ 100 componentes) | Cresce 4,2×: linear em N, somado a um custo fixo maior (a projeção pelo SVD) |

**Achado principal.** Ter índice não garante consulta sublinear. BM25, Vetorial e LSA calculam
um score para **todos** os N documentos a cada consulta, antes de ordenar para pegar o Top-k,
então o custo da consulta continua O(N). O BM25 mostra isso com mais clareza. O índice elimina
a releitura e a retokenização dos documentos (o que torna a Busca Linear 170× mais lenta que o
BM25 com N = 1.673), mas não elimina a varredura. Para ficar sublinear seria preciso percorrer
só as listas invertidas dos termos da consulta, ou usar poda do tipo WAND/MaxScore, e nenhuma
das bibliotecas usadas faz isso.

## Latência de indexação

Mediana de 3 indexações (ms):

| Modelo | N=100 | N=500 | N=1.000 | N=1.673 | Crescimento |
|---|---:|---:|---:|---:|---:|
| Busca Linear | 0,002 | 0,002 | 0,004 | 0,006 | não indexa: só guarda os ids |
| Booleano | 12,1 | 39,3 | 82,8 | 143,5 | 11,9× |
| Vetorial | 16,7 | 42,0 | 84,5 | 137,6 | 8,3× |
| BM25 | 10,8 | 39,6 | 84,3 | 129,8 | 12,0× |
| LSA | 112,0 | 221,2 | 357,1 | 575,3 | 5,1× |

Booleano, Vetorial e BM25 indexam em tempo aproximadamente linear no tamanho do corpus, e a
tokenização domina. O LSA é o mais caro em valor absoluto (~4× os demais com N = 1.673), por
causa do `TruncatedSVD` sobre a matriz TF-IDF. Ele cresce menos que N porque o SVD tem um
custo fixo alto e porque k fica limitado a 100 componentes.

## Relevância: MRR@10 e Recall@10

373 consultas, corpus completo, relevante quando `score > 0` nos qrels:

| Modelo | MRR@10 | Recall@10 |
|---|---:|---:|
| **BM25** (k1 = 1,5; b = 0,75) | **0,4041** | **0,6381** |
| Vetorial (TF-IDF + cosseno) | 0,3768 | 0,6059 |
| BM25 (b = 0) | 0,3511 | 0,5576 |
| LSA (100 componentes) | 0,2407 | 0,5201 |

- O **BM25 é o melhor** nas duas métricas: em 63,8% das consultas o documento correto está
  no Top-10.
- **BM25 com b = 0,75 × b = 0.** Desligar a normalização por comprimento derruba o MRR@10 em
  0,053 (de 0,404 para 0,351) e o Recall@10 em 8 pontos (de 63,8% para 55,8%), ficando abaixo
  até do Vetorial. Os números mostram que, nesse corpus, normalizar pelo comprimento ajuda.
  A explicação provável, ainda não verificada documento a documento, é que sem normalização
  respostas longas acumulam ocorrências dos termos da consulta e sobem no ranking.
- O **LSA é o pior**. A hipótese, a conferir nas consultas desafiadoras do notebook, é que a
  projeção em 100 componentes aproxima documentos do mesmo tema, mas perde a correspondência
  exata de termos específicos que pergunta e resposta costumam compartilhar em um FAQ.
- **Booleano e Busca Linear não ranqueiam.** Devolvem score fixo 1,0 na ordem do corpus e
  exigem todos os termos (AND implícito). Com as perguntas em linguagem natural, o Booleano só
  retorna algo em **56 das 373** consultas (15%). Por isso eles ficam fora da tabela. Para
  referência, calculando mesmo assim pela ordem do corpus, o Booleano teria MRR@10 = 0,036 e
  Recall@10 = 0,062.

## Limitações da análise

- N vai só até 1.673, o tamanho total do corpus. Em N maiores os custos fixos (Booleano e
  Vetorial) tendem a perder peso para a parte linear, mas isso não foi medido.
- As medições são de uma única máquina e uma única execução do script. O desvio padrão está
  no CSV.
- Cada consulta tem um único relevante nos qrels. Um documento igualmente útil, mas não
  anotado, conta como erro.
