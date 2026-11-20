# Guia introdutório do projeto PRIORITY–DreaMS (20-11-2026)

**Leitura de apoio para a reunião com Joana e Daniel.** Este guia
distingue as hipóteses científicas, experimentos implementados, resultados
executados e limitações. As sugestões metodológicas não são decisões
aprovadas pela equipe.

## 1. O problema biológico

Actinobactérias como Streptomyces e Salinispora podem produzir moléculas de
interesse farmacológico. A genômica investiga os genes que poderiam produzir
essas moléculas; a metabolômica mede sinais químicos presentes nas amostras.
Um BGC (biosynthetic gene cluster) é um conjunto de genes associado à
biossíntese. Queremos descobrir se determinado BGC pode estar relacionado
a determinado metabólito, mas um espectro MS/MS não determina sozinho
o BGC produtor.

## 2. Como observar as moléculas?

A espectrometria de massas em tandem (MS/MS) seleciona íons, os fragmenta
e registra os fragmentos por relação massa/carga (m/z) e intensidade. Um
espectro é um conjunto de picos. O GNPS organiza espectros e constrói
redes de similaridade. O MassIVE hospeda experimentos; o PoDP cataloga
projetos com relações genoma–metaboloma.

## 3. Quais métodos são comparados?

- **DreaMS:** Transformer pré-treinado de modo auto-supervisionado para
  aprender representações de MS/MS. O projeto usa o checkpoint publicado,
  não reproduz o treinamento gigante da publicação. Embedding: 1.024 valores.
- **MS2DeepScore:** rede siamesa treinada para que similaridades entre
  representações espectrais reflitam similaridades de estruturas químicas.
- **Binned cosine / modified cosine:** referências espectrais matemáticas,
  sem uma nova rede neural aprendendo com o seu experimento.
- **Fingerprints RDKit (Morgan) e Tanimoto:** representação e medida de
  similaridade entre estruturas químicas anotadas. Biblioteca GNPS tem
  identificações **putativas**, não confirmação experimental independente.

Artigo DreaMS: https://www.nature.com/articles/s41587-025-02663-3
MS2DeepScore: https://doi.org/10.1186/s13321-021-00558-4

## 4. Histórico do experimento

O levantamento inicial `docs/01_initial_exploration.md` buscou datasets
actinobacterianos e modelos de representação. Seis jobs GNPS foram
analisados; em um deles (Crüsemann Streptomyces), nenhum espectro
passou o protocolo estrito DreaMS. Os resultados de A–D foram
originalmente reexecutados em 18 de novembro (veja `RESULTS.md`).

O tutorial de redes moleculares DreaMS foi reproduzido e registrado:
3.809 espectros, 6.858 arestas, similaridade para um par
0,9109245 contra 0,9109248 esperada.

**Experimento A:** compara vizinhos do DreaMS, MS2DeepScore e cosine
às famílias GNPS. É *concordância com GNPS*, não prova independente
de identidade química. Em Tahlan: recovery@5 DreaMS 0,618,
MS2DeepScore 0,597 e cosine 0,654.

**Experimento B:** compara similaridades de espectros aos fingerprints
químicos anotados. Em Tahlan, 176 estruturas distintas e 15.400
pares DEPENDENTES: Spearman DreaMS 0,266, MS2DeepScore 0,319,
modified cosine 0,292.

**Experimento C:** pequeno adaptador treinável 1024→256 sobre DreaMS
congelado; métodos centralizado, local, FedAvg e FedProx. Exploração,
não um novo DreaMS ou demonstração de vantagem federada, pois o
orçamento de treinamento não é integralmente comparável.

**Experimento D:** classifica origem de diferentes datasets a partir
dos embeddings. Alta separabilidade não prova efeito do laboratório,
pois os compostos medidos também diferem.

**Experimento E:** média fixa 50/50 dos escores DreaMS e MS2DeepScore,
SEM novo treinamento. Real em Tahlan: Spearman 0,340985 e analog
top-1 Tanimoto 0,522952. Ganho sobre MS2DeepScore 0,021909
com intervalo exploratório [−0,007597, 0,051414]; não é
melhoria comprovada.

**Experimento F:** validação externa na coleção Chevrette. Dos 21
compostos putativamente anotados e distintos, 5 compartilham
conectividade com Tahlan. Restaram 16; a avaliação corretamente
declarou resultado inconclusivo, sem reportar correlação indevida.

**Experimento G:** protótipo de ranking por coocorrência de BGCs e
features metabolômicas nas MESMAS linhagens. Calcula Fisher, efeito
phi e BH/FDR. Com dados sintéticos encontra exemplos deliberadamente
construídos; dados REAIS insuficientes para executar ranking confiável,
pois falta matriz validada de presença/ausência e links verificados.
Não é demonstração de superioridade sobre NPLinker.

## 5. Três exemplos REAIS para aprender a interpretar os métodos

Executados no GitHub Actions em 20-11-2026, usando spectra reais,
embeddings salvos e candidatos de biblioteca GNPS. Regra de seleção:
primeiros três InChIKey-connectivity blocos em ordem alfabética;
não seleção de melhores acertos.

**Gly-Trp**, GNPS scan 4764, m/z precursor 262,119:
picos mais fortes (m/z relativo) 188,071 (1,0),
146,060 (0,4111), 132,081 (0,3554).
Os modelos escolhem Ser-Trp como vizinho:
DreaMS 0,7568; MS2DeepScore 0,9760;
fusão 0,8664; Morgan Tanimoto 0,6875.
Mesmo grupo GNPS. Nomes são putativos.

**Candidatos de dois lipídios com cadeias trocadas:**
1-Docosahexaenoyl-2-stearoyl-sn-glycero-3-phosphocholine e
1-Stearoyl-2-docosahexaenoyl-sn-glycero-3-phosphocholine.
DreaMS 0,9095, MS2DeepScore 0,9968, fusão 0,9532,
Morgan Tanimoto 1,0. Demonstra que o fingerprint pode
não distinguir detalhes posicionais.

**Candidato de cromenona metoxilada**, GNPS scan 15729,
precursor 471,105: ambos os modelos recuperam um vizinho
cujo Tanimoto de Morgan é 0,2162. Isso levanta uma pergunta
sobre discordâncias entre semelhança espectral e química,
sem provar erro do modelo ou identidade molecular.

Resultado JSON completo e ferramenta:
`dreams_molecular_families/scripts/teaching_walkthrough.py`.

## 6. Ideias para discutir (ainda NÃO implementadas como estudos completos)

1. Desempenho por **classe química**: examinar quais grupos moleculares
   são melhor representados por qual embedding; requer classes
   definidas e amostras suficientes.
2. Estudar **discordâncias entre modelos**, com controles de aduto,
   precursor m/z, fragmentação e confiabilidade das anotações.
3. Ranking genoma–metaboloma **explicável** combinando coocorrência,
   classes biossintéticas (ex. antiSMASH) e classes químicas
   (ex. CANOPUS); requer curação linhagem–espectro–BGC.
4. **Viés de exclusão do DreaMS**: por que Duncan tem 2% de
   cobertura do protocolo e Crüsemann Streptomyces 0%?
   A exclusão depende de tipo de composto ou características
   instrumentais? Não mudar os limites sem validação.
5. Eventualmente, **modelo multimodal** com embeddings de BGCs e
   MS/MS, mas somente após obter pares positivos independentes
   suficientes e um conjunto de teste sem vazamento.

Preferência prática: estudar (1), (2) e (4) em paralelo à curadoria de
dados para (3). Não iniciar (5) nem federated learning sem alvo
biológico validado.

## 7. O que foi verificado

GitHub Actions
https://github.com/Victoria125/priority-wur/actions/runs/44698804477
executa 73 testes Python, realiza as análises E/F com embeddings
pré-calculados de Tahlan/Chevrette, imprime 3 exemplos reais e
um exemplo **sintético explicitamente marcado** para G, verifica
bloqueios de dados incompletos e compila 9 pequenas proposições
com Lean 4.

**Não se pode concluir:** que Lean provou a verdade biológica,
que a IA identifica automaticamente genes produtores, que a fusão
supera MS2DeepScore fora de Tahlan, ou que reproduzimos toda a
publicação do DreaMS. Os resultados de A–D vieram de execuções
registradas antes, não foram inteiramente reexecutados nessa
demonstração didática.
