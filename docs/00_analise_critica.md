# PRIORITY (WUR): o que descobri e análise crítica das instruções

*11/10/2026 (domingo). Para uso próprio, não enviar.*

## 1. Quem é quem (verificado)

**Joana Tirana**
- Postdoc no grupo de Bioinformática da WUR desde abril de 2026.
- No portal da WUR, aparece como PI do postdoc "PRIORITY" (01/04/2026 a 01/04/2029). Medema está como project leader, e van der Hooft e Gijs Selten como co-PIs.
- O CV dela lista Medema, van der Hooft e Probst como supervisores.
- **A formação dela é em computação, não em biologia.** Fez o PhD na UCD (Dublin) sobre split/federated learning, com defesa em fevereiro de 2026.
- Último paper: *"Data Heterogeneity and Forgotten Labels in Split Federated Learning"*, AAAI 2026.
- Os 16 repositórios públicos no GitHub (jtirana98) são todos de FL/split learning. Nenhum é de bioinformática.

**PRIORITY**
- Título completo: *"Prioritising novel genomic and chemical space via federated learning and preclinical development"*.
- Financiamento NWO + VWS (chamada KIC "Antimicrobials for the preclinical pipeline", anunciada em 18/03/2025).
- Líder: Gilles van Wezel (Leiden).
- Consórcio:
  - WUR: Medema, van der Hooft, **Probst**.
  - LUMC.
  - Leiden: Somayah Elsayed.
  - NAICONS (empresa; Donadio).
  - Fundación MEDINA (Genilloud).
  - GARDP.
  - KU Leuven/VIB (Masschelein).
  - UF Scripps (Ben Shen, do NPDC).
  - DOE/JGI (Nigel Mouncey).

**Por que tem FL.** O paper que explica o projeto é Medema & van Wezel, *PLOS Biol* 2025 ("Prioritizing microbial biosynthetic space using ecology and machine learning"). Ele diz que dados pareados de genoma, metaboloma e bioatividade ficam espalhados e às vezes presos por propriedade intelectual. Propõe FL para ligar dados públicos a coleções (semi)privadas e cita o MELLODDY como modelo. **É por isso que a Joana te mandou o MELLODDY.** Os "clientes" do FL seriam as coleções de linhagens dos parceiros (NAICONS e MEDINA têm coleções industriais enormes), não hospitais.

**Daniel Probst**
- Assistant professor na WUR desde setembro de 2024 (lab "Computer Science × Natural Sciences").
- Trabalha com cheminformática e ML eficiente: MAP4, MHFP, TMAP.
- Foi ele quem iniciou a colaboração, então provavelmente tem um ângulo próprio, possivelmente o lado químico (o "chemical space" do título). *Isso é hipótese minha; pergunte.*

**O que os papers da Joana sinalizam**

| Paper | O que é de fato | O que indica |
|---|---|---|
| NPLinker scoring + NPLinker 2 | Ferramenta *do próprio grupo* para ligar BGCs a espectros | Ela vai usar ou comparar com o NPLinker |
| Linking genomics and metabolomics (review) | Review do grupo | Contexto |
| GLUE | Método de integração multi-ômica com grafo (single-cell) | Ideia de integração |
| COMMBAT | Prevê **quais fatores de transcrição regulam um BGC** (motivos/TFBS). **Não é dado de transcriptômica** | Interesse em regulação e expressão de BGC |
| MELLODDY | FL entre farmacêuticas | O modelo do PRIORITY |
| Hannemann et al., ICCS 2024 | FL em transcriptômica (Flower vs TFF) | Ela pensa em FL também para transcriptômica |

## 2. Quais datasets a Joana está usando?

**Resposta honesta: não há evidência pública.** Não existe repositório nem output dela no portal, e no e-mail ela diz que *ainda está procurando* um dataset. Abaixo, minha inferência com nível de confiança:

| Fonte | Confiança | Por quê |
|---|---|---|
| Paired Omics Data Platform (PoDP), via NPLinker | Alta | É do grupo dela, o NPLinker 2 tem "modo PoDP", e ela mandou os papers |
| MIBiG 4.0 | Alta | Ground truth padrão do grupo (BGC → composto) |
| Genomas do NPDC (Ben Shen) | Média-alta | Ben Shen é parceiro; são ~18,7 mil genomas públicos de actinobactérias |
| MSV000094769 (UF + Dorrestein + Mohimani) | Média | Quem submeteu é da UF (instituição do Ben Shen), mas não há paper nem mapeamento |
| Dados de Leiden (van Wezel/Elsayed) | Média | São do líder do projeto; quase nada é público |
| Coleções NAICONS / MEDINA | Certa no futuro, **inacessível** | É a razão de existir FL; não peça esses dados |

**O que medi de fato** (scripts em `scripts/`, rodados hoje):
- **PoDP:** 81 projetos, dos quais **26 são majoritariamente de actinobactérias**. Juntos têm ~395 genomas, 1.297 links genoma↔metaboloma curados e 58 links BGC↔MS2 (a única "verdade" para linking).
- **MSV000094769:**
  - 161 GB no total (1.916 mzML duplicados em duas pastas, ~80 GB únicos) e 359 FASTA do RefSeq.
  - Os arquivos vêm copiados de **15 datasets anteriores**, um deles de isolados de *Populus* (nem tudo é actinobactéria).
  - **Não há nenhum arquivo que diga qual espectro pertence a qual genoma.**

## 3. As instruções: o que eu seguiria e o que mudaria

**Veredito geral:** a direção está certa. Agir já, de forma exploratória, primeiro os dados, sem codar FL e sem cobrar o Daniel. Mas a execução proposta tem erros factuais e uma lacuna grande: **ela ignora o que é o PRIORITY**. Ela trata o projeto como "integração multi-ômica genérica", quando ele é "priorizar descoberta de antibióticos com FL entre coleções".

| Instrução | Veredito | Comentário |
|---|---|---|
| Não mandar outro e-mail ao Daniel hoje | ✅ Seguiria | Correto. Na terça, mande *algo concreto*, não um "e aí?" |
| Agir já, em modo exploratório | ✅ Seguiria | A Joana te deu espaço para isso |
| Dados antes de FL; não codar FedAvg agora | ✅ com ressalva | Concordo em não codar. Mas FL é o coração do projeto e a especialidade da Joana. **Pense desde já em quem seriam os clientes**, porque isso muda quais dados importam (é preciso metadado de origem: laboratório, instrumento, meio) |
| Usar modelos pré-treinados, não treinar foundation model | ✅ Seguiria | Correto |
| "MSV000094769 é o dataset nº 1, com strain_001/genome.fasta + metabolome.mzML" | ❌ Mudaria | O dataset existe, mas *não é* isso. Não há mapeamento, há mistura de fontes, 161 GB e nenhum paper. Vira nível B: "promissor, depende de pedir o mapeamento" |
| Saccharomonospora XJ54 como prova de conceito tri-ômica | ⚠️ Parcial | Verifiquei: é real e está no PoDP. Mas é **1 linhagem com 15 amostras**. Serve para explorar e visualizar, não para treinar nada |
| GSE132487 para "testar um modelo de embedding transcriptômico" | ⚠️ Mudaria | São 27 amostras. Melhor usar o compêndio de iModulons de *S. coelicolor* (454 perfis). E sendo honesta: **não existe foundation model de transcriptômica bacteriana**. Isso é um achado útil para a Joana |
| "DNA foundation model" genérico para genoma | ⚠️ Mudaria | Actinobactérias têm ~70% de GC e BGCs de 10 a 100+ kb, maiores que o contexto de DNABERT-2/NT. Comece por modelos de proteína ou de contexto gênico (ESM-2, **gLM2**, Bacformer) e pelas ferramentas da casa (BiG-SCAPE/BiG-SLiCE). Os slides do Dimitris servem pela *avaliação* (gêneros fora do treino, hard negatives), não pelo modelo |
| Pergunta científica "embeddings podem representar atividade biossintética?" | ❌ Mudaria | Vaga, sem rótulo e sem métrica. Troque por tarefas com ground truth (ver `01_initial_exploration.md`, seção 3) |
| (ausente) Bioatividade | ❌ **Maior lacuna** | PRIORITY é sobre antibióticos. Sem a camada de atividade não há "priorização". Chevrette 2019 (120 genomas, com ensaios) e MIBiG resolvem parte disso |
| (ausente) Ferramentas do próprio grupo | ❌ Lacuna | NPLinker, antiSMASH, BiG-SCAPE, MIBiG (Medema), Spec2Vec/MS2DeepScore/MS2Query (van der Hooft), MAP4 (Probst). Propor essas primeiro mostra que você entendeu o grupo |
| Tabela `dataset_candidates.csv` + página "Initial exploration" | ✅ Seguiria, melhorado | Fiz, com coluna de verificação e scripts que reproduzem os números |
| Estrelinhas de classificação | ❌ Tiraria | Não dizem nada à Joana. Troque por "por que importa / ressalvas" |
| Classificar o GSE132487 como dado novo e "de agora" | ⚠️ | É de 2019 (SYSTERACT), mas tem o **van Wezel como contribuidor**. Use isso a seu favor |

**Seguiria totalmente diferente?** Não. Seguiria cerca de 60% e reenquadraria o resto. A mudança principal é o tipo de contribuição:

> **A melhor contribuição agora é: transformar dados públicos em um benchmark federado realista para o PRIORITY.**
> Os dados pareados de actinobactérias no PoDP já vêm, naturalmente, de laboratórios, instrumentos, meios e gêneros diferentes. Dividindo por laboratório de origem, você simula o consórcio (NAICONS, MEDINA, NPDC, Leiden…) com dados abertos.

Isso combina com:
- **o seu perfil**: dados, metadados e grafos de conhecimento, como no TACO-RDF. O PoDP é literalmente um problema de metadados e ligações;
- **o da Joana**: heterogeneidade e non-IID em FL, que é o paper AAAI dela;
- **e o que ela pediu**: dataset e representações.

Um ponto que vale levantar com ela: se no consórcio cada parceiro tiver *modalidades diferentes das mesmas linhagens*, isso é **FL vertical**, e split learning (a especialidade dela) encaixa muito bem. Coloquei como pergunta no documento.

## 4. Plano sugerido

| Quando | O quê |
|---|---|
| Seg 12/10 | Ler `01_initial_exploration.md` e ajustar à sua voz. Instalar o NPLinker 2 e rodá-lo em **um** dataset do PoDP (MSV000078839, que é pequeno). Isso vira "já rodei o pipeline do grupo" |
| Ter 13/10 | Mandar para a Joana o documento (curto) + as 5 perguntas. Se o Daniel continuar em silêncio, copie-o ou mande um e-mail curto com o mesmo material ("sharing what I explored"), em vez de cobrar de novo |
| Qua–Sex | Extrair a tabela de bioatividade do Chevrette 2019 (suplementar) e montar o baseline da tarefa 2 (BGC → atividade, MIBiG) ou da tarefa 1 (NPLinker), conforme a resposta da Joana |
| Depois | Simulação federada (Flower) com clientes = laboratórios |

## 5. Cuidados

- **Não diga que achou "o dataset que ela usa".** Diga que mapeou o que existe publicamente.
- **Não redesenhe o projeto por eles.** Apresente opções e pergunte; o Daniel pode ter outro plano.
- **MSV000094769 não tem publicação.** Antes de usar, pergunte à Joana se o grupo tem contato com os autores (UF/UCSD/CMU).
- **Números de terceiros que não conferi:** a data de publicação do paper do XJ54 (o PoDP aponta para um bioRxiv de fevereiro de 2026) e os detalhes do compêndio de iModulons.
- **E-mails:** só li os três threads (Joana/Daniel). Não enviei nada.

## 6. Rascunho de e-mail para terça (ajuste antes)

> Hi Joana,
>
> Following our discussion, I mapped the public paired-omics data for actinobacteria and candidate representation models for each layer. I wrote a short summary (attached/linked) with what I found, three possible first tasks, and a few questions about where I could be most useful.
>
> Two things stood out: the PoDP already has ~400 actinobacterial genomes with curated genome–metabolome links coming from different labs, which could make a realistic federated simulation; and the large MSV000094769 deposit has no genome-to-spectrum mapping, so it would need input from its authors.
>
> I'd be happy to go through it whenever suits you.
>
> Best,
> Vitória

## Fontes principais

- Projeto PRIORITY no portal da WUR: https://research.wur.nl/en/projects/priority/
- NWO, consórcios (PRIORITY): https://www.nwo.nl/en/news/five-public-private-consortia-launch-research-into-solutions-for-antimicrobial-resistance
- Medema & van Wezel 2025: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003058
- CV da Joana: https://jtirana98.github.io/ · GitHub: https://github.com/jtirana98
- Probst Lab: https://probstlab.science
- PoDP API: https://pairedomicsdata.bioinformatics.nl/api/projects
- MassIVE MSV000094769: https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000094769
- NPLinker 2 (JOSS 2026): https://joss.theoj.org/papers/10.21105/joss.09528
- COMMBAT (NAR 2026): https://www.commbat.uliege.be/
- Hannemann et al. ICCS 2024: https://arxiv.org/abs/2402.14527
- NPDC: https://npdc.rc.ufl.edu
- Chevrette et al. 2019: https://doi.org/10.1038/s41467-019-08438-0
- Walker & Clardy 2021 (BGC → atividade): https://pmc.ncbi.nlm.nih.gov/articles/PMC8243324
- gLM2: https://huggingface.co/tattabio/gLM2_650M · Bacformer: https://www.biorxiv.org/content/10.1101/2025.07.20.665723
- DreaMS: https://dreams-docs.readthedocs.io/
