# Roteiro da apresentação (15 min + 5 min de Q&A)

Números tirados de `benchmark/resultados/latencia.csv`, `benchmark/resultados/relevancia.csv`
e do notebook `notebooks/02_analise_relevancia.ipynb`. Números que **não** vêm do projeto
aparecem marcados como **[hipótese]**.

## Visão geral

| # | Slide | Tempo | Acumulado |
|---:|---|---:|---:|
| 1 | Problema e objetivo | 1:00 | 1:00 |
| 2 | Dataset e metodologia | 1:30 | 2:30 |
| 3 | Os 5 modelos e a arquitetura comum | 1:30 | 4:00 |
| 4 | Demo ao vivo | 3:00 | 7:00 |
| 5 | Latência × N | 2:00 | 9:00 |
| 6 | Achado: ter índice não garante consulta sublinear | 1:30 | 10:30 |
| 7 | Relevância: MRR@10 e Recall@10 | 1:30 | 12:00 |
| 8 | Consultas desafiadoras | 1:00 | 13:00 |
| 9 | ROI para uso corporativo | 1:00 | 14:00 |
| 10 | Limitações e conclusão | 1:00 | 15:00 |

Os tempos são o teto de cada slide. Se a demo atrasar, cortar o slide 8, que já foi
mostrado em parte na própria demo.

---

## Slides

### 1. Problema e objetivo (1:00)
- Quando um cliente pergunta algo ao banco, qual algoritmo encontra a resposta certa no FAQ,
  e quão rápido?
- Objetivo: implementar, medir e comparar 5 modelos clássicos de RI sobre o FAQ do Banco
  Central.
- **Fala:** "Não é só qual acerta mais: é quanto custa acertar quando o corpus cresce."

### 2. Dataset e metodologia (1:30)
- `MTEB-BR/faq-bacen` (HuggingFace, Apache-2.0): **1.673 respostas**, **373 perguntas**,
  **1 resposta certa por pergunta**.
- Latência: N = 100, 500, 1.000 e **1.673 (o corpus inteiro)**, em subconjuntos aninhados;
  50 consultas × 3 repetições, após aquecimento.
- Relevância: MRR@10 e Recall@10 nas 373 perguntas.
- **Fala:** "1.673 não é um número escolhido: é o tamanho real do corpus."

### 3. Os 5 modelos e a arquitetura comum (1:30)
- Busca Linear (baseline, sem índice) · Booleano (matriz de incidência bitwise) ·
  Vetorial (TF-IDF + cosseno) · BM25 (k1 = 1,5; b = 0,75) · LSA (SVD com 100 componentes).
- Mesma interface para todos: `indexar(documentos)` e `buscar(consulta, k) -> [(id, score)]`.
- Mesmo pré-processamento (`tokenizar_e_filtrar`), então as diferenças vêm do modelo.
- Qualidade: 57 testes com pytest, flake8 sem apontamentos.

### 4. Demo ao vivo (3:00)
Ver a seção [Roteiro da demo](#roteiro-da-demo-ao-vivo) abaixo.

### 5. Latência × N (2:00)
- Mostrar `escalabilidade_log.png`.
- N cresce **16,7×** (100 → 1.673):
  - Busca Linear: 20,6 → **300,6 ms** (**14,6×**, linear).
  - BM25: 0,12 → **1,78 ms** (**14,9×**, também linear).
  - Vetorial: 1,7×. LSA: 4,2×. Booleano: constante (~0,04 ms).
- **Fala:** "No log, a Busca Linear fica duas ordens de grandeza acima de todos."

### 6. Achado: ter índice não garante consulta sublinear (1:30)
- BM25, Vetorial e LSA pontuam **todos os N documentos** em cada consulta e só depois
  ordenam o Top-k.
- O índice evita reler e retokenizar do disco (Busca Linear ~170× mais lenta que o BM25
  com N = 1.673), mas não evita a varredura.
- Para ficar sublinear: percorrer só as listas invertidas dos termos da consulta, ou usar
  poda WAND/MaxScore.

### 7. Relevância: MRR@10 e Recall@10 (1:30)

| Modelo | MRR@10 | Recall@10 |
|---|---:|---:|
| **BM25** | **0,404** | **63,8%** |
| Vetorial | 0,377 | 60,6% |
| BM25 (b = 0) | 0,351 | 55,8% |
| LSA | 0,241 | 52,0% |

- Desligar a normalização por comprimento (`b = 0`) derruba o BM25 para abaixo do Vetorial.
- Booleano: só responde em **56 das 373** perguntas, porque exige todos os termos. Por isso
  ele e a Busca Linear ficam fora do ranking.

### 8. Consultas desafiadoras (1:00)
- Termo raro, "idosos e gestantes": Vetorial e BM25 acertam em 1º; o LSA erra.
- Polissemia, "conta de luz": só o Vetorial põe a resposta em 1º.
- Sinônimo, "empréstimo para comprar casa": ninguém acerta; os lexicais trazem "Casa da
  Moeda".
- Fora do vocabulário, "bitcoin": ninguém acerta (o corpus diz "criptoativos").

### 9. ROI para uso corporativo (1:00)
Ver a seção [Argumento de ROI](#argumento-de-roi) abaixo.

### 10. Limitações e conclusão (1:00)
- Limitações: N ≤ 1.673; uma máquina; 1 relevante por pergunta; sem sinônimos nem stemming.
- **Conclusão:** para um FAQ como esse, o **BM25** é a melhor escolha: tem a melhor
  relevância e fica em ~2 ms por consulta. O próximo passo seria tratar sinônimos (expansão
  de consulta ou embeddings) e, em corpus grande, usar listas invertidas com poda.

---

## Roteiro da demo ao vivo

**Preparação (antes da aula):**
1. Na raiz do projeto, com o `.venv` ativo, rodar `python demo.py` uma vez e sair com
   `sair`. Na 1ª execução da máquina a indexação levou **13 s**; nas seguintes, **~1,2 s**
   (medido nesta máquina).
2. Deixar o terminal com fonte grande, já na raiz do projeto.
3. Deixar abertos numa aba: os 2 PNGs, o README (tabelas) e o notebook executado (plano B).

**Ao vivo:** `python demo.py`, depois **Enter** (modo Todos). As 3 consultas abaixo foram
executadas na demo em 29/09/2026 com estes resultados:

| # | Digitar | O que mostrar |
|---|---|---|
| 1 | `o banco pode cobrar` | Vetorial e BM25 trazem o mesmo Top-5; o 1º (a1173) diz "o banco pode cobrar tarifa pela emissão". O LSA põe cheque cruzado (a957) em 1º. Busca Linear e Booleano encontram documentos, mas com score 1,0 na ordem do corpus: não ranqueiam |
| 2 | `atendimento prioritário para idosos e gestantes` | Busca Linear e Booleano: nada (exigem todos os termos). Vetorial e BM25: a21 em 1º (BM25 17,4 contra 5,1 do 2º). O LSA não traz a21: fica no tema genérico "atendimento" |
| 3 | `como pagar a conta de luz` | Só o Vetorial põe a970 ("contas de serviços públicos (água, luz...)") em 1º; o BM25 a põe em 3º; o LSA só traz conta bancária e conta-salário |

Consulta extra, se sobrar tempo: `:modelo`, depois `2` (Booleano) e `pix AND limite`.
Mostra os operadores booleanos (retorna a486, a768...). Não usar AND no modo Todos: a Busca
Linear trata "AND" como palavra e não encontra nada.

**Plano B (se a demo falhar):** não tentar consertar ao vivo. Dizer "vou mostrar a saída já
gravada" e abrir:
1. o notebook `02_analise_relevancia.ipynb`, que já tem salvo o Top-5 lado a lado das mesmas
   consultas 1, 2 e 3;
2. `escalabilidade_log.png` e as tabelas do README.

---

## Argumento de ROI

**O que é fato (medido no projeto):**
- O BM25 encontra a resposta certa no Top-10 em **63,8%** das perguntas reais do FAQ, e no
  Top-1 em boa parte delas (MRR@10 = 0,404).
- Custo computacional desprezível: ~**1,8 ms** por consulta e ~**130 ms** para indexar o FAQ
  inteiro, numa máquina comum, sem GPU e com bibliotecas open source.

**Cenário de uso:** busca de autoatendimento no FAQ antes de abrir um chamado (atendimento ao
cliente), ou busca interna de normas para o time de compliance.

**Conta ilustrativa (todos os valores são hipóteses, não medições):**
- **[hipótese]** 10.000 perguntas por mês chegam ao atendimento.
- **[hipótese]** 20% delas deixam de virar chamado porque o cliente acha a resposta no Top-5.
- **[hipótese]** R$ 5,00 por chamado atendido por uma pessoa.
- Economia = 10.000 × 20% × R$ 5,00 = **R$ 10.000/mês** nesse cenário.

A fórmula é o que importa: *economia = volume × taxa de desvio × custo por atendimento*. Cada
empresa põe os próprios números; a taxa de desvio precisaria ser medida num piloto (teste
A/B). O custo de rodar o BM25 é praticamente zero perto disso.

---

## Perguntas prováveis (Q&A)

**1. Por que o BM25 também cresce linearmente, se tem índice?**
A biblioteca `rank_bm25` calcula o score de todos os documentos para cada termo da consulta
(`get_scores`) e só depois ordena. O índice evita reler o disco, mas não a varredura. Com
listas invertidas e poda WAND/MaxScore só seriam tocados os documentos que contêm os termos.

**2. Por que o LSA foi pior?**
Com 100 componentes, o SVD agrupa documentos por tema, mas termos raros e específicos
(ex.: "idosos", "gestantes", que aparecem num só documento) quase não pesam nas dimensões
latentes. Num FAQ, a pergunta e a resposta costumam compartilhar justamente esses termos.

**3. Por que o Booleano e a Busca Linear ficaram fora das métricas?**
Eles não ranqueiam: devolvem score fixo 1,0 na ordem do corpus e exigem todos os termos. O
Booleano só responde em 56 das 373 perguntas. MRR mede posição, e sem ranking a posição é
arbitrária. Calculado mesmo assim, o Booleano teria MRR@10 = 0,036.

**4. O que mudaria com um corpus maior?**
Não medimos acima de 1.673 (é o corpus inteiro). A expectativa é que os custos fixos do
Booleano e do Vetorial percam peso e eles passem a crescer linearmente, como o BM25. A Busca
Linear ficaria inviável. Aí compensaria usar listas invertidas com poda, ou um motor como
Lucene/Elasticsearch.

**5. Por que o Booleano é tão rápido?**
É um AND bit a bit do numpy sobre vetores de no máximo 1.673 posições. Nesse tamanho o custo
fixo do Python domina e o tempo fica constante (~0,04 ms).

**6. O que faz o parâmetro `b` do BM25?**
Normaliza pelo comprimento do documento. Com `b = 0`, o MRR@10 cai de 0,404 para 0,351.
Provável motivo: sem normalização, respostas longas acumulam ocorrências dos termos e sobem
no ranking.

**7. Como resolver sinônimos ("casa" × "imobiliário", "bitcoin" × "criptoativos")?**
Os modelos lexicais não resolvem. As saídas seriam expansão de consulta com um dicionário de
sinônimos do domínio, ou modelos de embeddings densos, trocando custo computacional por
cobertura semântica.

**8. O Recall@10 de 63,8% é bom?**
Cada pergunta tem um único relevante anotado, então um documento útil, mas não anotado,
conta como erro, e a métrica tende a ser pessimista. Para uma busca lexical sem stemming nem
sinônimos, significa que em quase 2 de cada 3 perguntas a resposta certa está na primeira
página.
