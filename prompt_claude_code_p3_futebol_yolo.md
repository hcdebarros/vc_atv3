# PROMPT PARA O CLAUDE CODE — Prática 3: Futebol Analytics com YOLO

> Antes de colar: anexe os 3 PDFs e deixe a pasta do projeto vazia (ou com o clone de `github.com/eron93br/vc-20262-p3`, se tiver acesso).

---

## PAPEL

Você é um engenheiro sênior de visão computacional e ciência de dados esportiva. Vai me ajudar a desenvolver, do zero e de ponta a ponta, a **Prática 3 — Futebol Analytics com YOLO** (disciplina de Visão Computacional, CESAR School). Sou aluno de Ciência de Dados e IA, programo em Python e o projeto será entregue por **grupo**, com repositório GitHub, slides e apresentação de 10 min **em 09/11/2026**. O prazo é curto: priorize um pipeline funcional de ponta a ponta antes de refinar.

## 0. LEIA PRIMEIRO (obrigatório, antes de escrever código)

Os três PDFs anexados são a fonte da verdade:

1. **`P3 - Futebol Analytics com YOLO.pdf`** — enunciado: objetivos, regras, pipeline, saídas, critérios de avaliação e penalidades. **É o documento que manda.**
2. **Ahmad Yousef et al. (2025), Procedia CS** — YOLOv8x × YOLOv12x para jogador/goleiro/árbitro/bola; filtro de Kalman para a bola (F1 da bola 0,31 → 0,96); posse por distância jogador–bola.
3. **Gomes et al. (2026), REIC** — componentes de passe do EPV com SoccerMap (FCN); estado de jogo em 17 canais; P(sucesso | destino, estado) como superfície no campo; AUC, Brier e Log-Loss contra baseline de distância (SoccerMap: AUC 0,622 / Brier 0,138 / Log-Loss 0,444; baseline AUC 0,588); análise what-if.

Se existir pasta do repositório de apoio `eron93br/vc-20262-p3` (ou código de exemplo na pasta atual), **leia os códigos antes** e reaproveite nomes, formatos e convenções. A regra do enunciado diz "Usar o modelo da disciplina!": procure no repo/enunciado qual modelo/pesos YOLO a disciplina fornece e use esse. **Se não achar, pergunte-me; não invente.**

Depois de ler, **resuma em ≤15 linhas** o que entendeu (entregáveis, penalidades, pipeline) e liste dúvidas/lacunas. Só então comece.

## 1. DADOS DO MEU GRUPO

- URL do YouTube: `https://www.youtube.com/watch?v=fg3DmFxMWZ4`
- Intervalo do trecho (3 min, câmera principal, jogo corrido, sem replays/cortes): `18:20–21:20`
- **O vídeo JÁ FOI BAIXADO E CORTADO por mim** (não crie nenhum script de download/corte e não use yt-dlp). Caminho local do arquivo (**contém espaços**, trate com aspas/`pathlib`): `D:\programing\cesar\VC\vc-20262-p3\videos\Psg Bayern - camara tactica_clip.mp4` — fica fora do git.
- Nome do jogo (pasta `saidas/<jogo>/`): `psg_bayern`
- Hardware: GPU NVIDIA RTX 5060 Ti (8 GB VRAM), CPU AMD Ryzen 7 5700x, 16 GB RAM; SO: Windows
- **GPU Blackwell (RTX 50):** instale o PyTorch com CUDA 12.8+ (ver pytorch.org/get-started), confirme com `torch.cuda.is_available()` e imprima o nome da GPU antes de rodar o YOLO. Se cair para CPU, avise-me.
- **Memória:** processar o vídeo quadro a quadro (streaming), sem carregar tudo na RAM; salvar resultados em CSV incrementalmente.
- Integrantes e quem apresenta (1–2 pessoas): `[...]`

Se algum campo estiver vazio, pergunte antes de executar a etapa que depende dele.

## 2. REGRAS INEGOCIÁVEIS (violações custam nota)

| Penalidade | Como evitar |
|---|---|
| Vídeo versionado: **−10%** | `.gitignore` com `*.mp4`, `*.mkv`, `*.avi`, `videos/`, pesos grandes — **criar o `.gitignore` antes do primeiro commit**. Registrar só URL + intervalo no README. Rodar `git ls-files` antes de cada commit para conferir. |
| README ≠ arquivos gerados: **−20%** | Todo número do README deve vir dos arquivos em `saidas/` via script (`gerar_readme_numeros.py` ou template preenchido automaticamente). **Nunca digitar métrica à mão.** Ao final, rodar um script de consistência que compara README × arquivos. |
| Sem validação manual dos passes: **−15%** | Criar `passes_anotados.csv` e `validar_passes.py` (precisão, revocação, F1). A anotação manual **é feita por mim**; você constrói a ferramenta e o template. |

Outras regras:
- Uso de IA é permitido no código, mas **análise crítica e README devem ser autorais**: gere o esqueleto/estrutura e marque claramente `<!-- ESCREVER: ... -->` onde eu devo redigir a análise com minhas palavras. Inclua a **declaração de uso de IA** no README.
- Vídeo enviado por Google Drive (não pelo repo).
- Nunca fabrique resultados, métricas ou anotações. Se uma etapa não puder ser rodada (sem vídeo, sem GPU, sem calibração), diga e deixe o stub pronto.
- Reprodutibilidade: `requirements.txt` com versões fixadas, seeds fixas, e `README` com comandos exatos para rodar tudo (comandos para Windows/PowerShell).

## 3. ESTRUTURA DO REPOSITÓRIO

```
.
├── README.md                  # URL+intervalo, como rodar, respostas, análise, declaração de IA
├── requirements.txt
├── .gitignore                 # vídeos, pesos, __pycache__, etc.
├── extrair_frame0.py          # lê o vídeo local já cortado, valida (duração, fps, resolução) e salva frame0.jpg
├── detectar_rastrear.py       # YOLO (pessoa + bola) + ByteTrack -> tracks.csv, meta.json, rastreado.mp4
├── classificador_times.py     # K-Means na cor da camisa; árbitro/goleiros; atualiza meta.json
├── homografia.py              # calibração (≥4 marcos), ORB+RANSAC entre quadros, Kalman da bola
├── metricas.py                # posse, distância/velocidade, passes, mapas
├── xpass.py                   # eXP: baseline, heurístico, logístico (+ what-if)
├── soccermap_mini.py          # (EXTRA) mini SoccerMap
├── validar_passes.py          # precisão/revocação/F1 vs passes_anotados.csv
├── comparar_yolo.py           # n/s/m, imgsz, conf no mesmo subtrecho de 30 s
├── ferramentas/
│   ├── clicar_marcos.py       # UI (OpenCV) para eu clicar os marcos no frame0.jpg
│   └── anotar_passes.py       # UI/CLI para eu anotar passes manualmente
├── saidas/<jogo>/
│   ├── tracks.csv  meta.json  bola.csv  calibracao.json
│   ├── tracks_campo.csv  passes.csv  passes_anotados.csv
│   ├── metricas_times.json  metricas_jogadores.csv  resultados_xp.csv
├── figuras/                   # calor por time, mapa de passes, linha do tempo de posse, grade reprojetada, superfície eXP
└── slides/                    # esqueleto do PPT (sem vídeo)
```

## 4. PIPELINE E CRITÉRIOS DE ACEITE POR ETAPA

**Etapa 1 — Aquisição e preparação (10%)**
O vídeo já está baixado e cortado (18:20–21:20), então **não há etapa de download**. `extrair_frame0.py` abre o arquivo local, imprime duração/fps/resolução/nº de quadros (e avisa se a duração estiver fora de 3–5 min), e salva `frame0.jpg`. Opcional: detector simples de cortes de cena (diferença de histograma entre quadros) que lista timestamps suspeitos, para eu conferir que não há replay/troca de câmera. Registrar URL e intervalo no README, explicando que o vídeo foi obtido fora do repositório e está no Google Drive.

**Etapa 2 — Detecção, rastreamento e times (25%)**
- YOLO COCO (classes `person` e `sports ball`) + **ByteTrack** → `tracks.csv` (frame, id, classe, bbox, conf).
- `comparar_yolo.py`: ≥2 tamanhos (n/s/m) no mesmo subtrecho de 30 s; reportar **fps, jogadores/quadro e % de quadros com bola**; variar `imgsz` e `conf` da bola. Escolha final justificada com os números. (Com 8 GB de VRAM, evitar modelos x.)
- `classificador_times.py`: K-Means na cor da camisa (recorte do tronco, não do bbox inteiro) → times A/B, **árbitro e goleiros** tratados à parte; `meta.json` com as cores.
- Gerar `rastreado.mp4` (fora do git) para eu conferir cores e trocas de ID. Reportar trocas de ID observadas/estimadas.

**Etapa 3 — Do pixel ao campo (parte do 30%)**
- `ferramentas/clicar_marcos.py`: eu clico ≥4 marcos conhecidos (meio-campo, círculo central, quinas da área etc.) no `frame0.jpg`; o script calcula **H0** e grava `calibracao.json` com o **erro de reprojeção em metros** (meta do exemplo: ≤ 0,41 m).
- Câmera em movimento: ORB + RANSAC entre quadros, mascarando jogadores e placar → **Gₜ** (pan/zoom). Projeção: `[x,y,1]ᵀ ~ H0 · Gₜ · [u,v,1]ᵀ`, usando o ponto dos pés (centro da base do bbox).
- Bola: **filtro de Kalman** preenchendo lacunas + *gating* para descartar detecções incoerentes (como no paper do Ahmad Yousef). `bola.csv`.
- Figura: grade de 5 m reprojetada sobre o `frame0.jpg`. → `tracks_campo.csv` (x, y em metros; campo 105×68 m).

**Etapa 4 — Métricas de jogo (30%)**
- **Posse:** jogador mais próximo da bola a < 2 m; a posse segue até o rival tocar. → `metricas_times.json`.
- **Distância e velocidade** por jogador (soma de deslocamentos em m; descartar saltos > 12 m/s). → `metricas_jogadores.csv`.
- **Passes:** troca de dono, mesmo time = certo, outro time = errado. `passes.csv` com colunas: `frame_ini, passador, receptor, sucesso, x0, y0, x1, y1, distancia, duracao_s`.
- **Mapas:** calor por time, mapa de passes, linha do tempo de posse (`figuras/`).
- **Sensibilidade:** repetir posse e nº de passes com raio de 1, 2 e 3 m e comparar numa tabela.
- **Validação:** `ferramentas/anotar_passes.py` me ajuda a anotar à mão os passes do trecho (reproduzindo o vídeo/frames e registrando). `validar_passes.py` casa passes detectados × anotados (tolerância temporal configurável) e reporta precisão, revocação e F1.

**Etapa 5 — eXP: expected pass (parte do 35%)**
- `eXP = σ(β0 + β1·distância + β2·adversários na linha + β3·pressão + β4·espaço no destino + …)`.
- Comparar na mesma tabela (AUC ↑, Brier ↓, Log-Loss ↓): **Baseline (distância)**, **Heurístico (β fixos)**, **Logístico (treinado nos meus passes)**; incluir a linha de referência SoccerMap (Gomes et al.: 0,622 / 0,138 / 0,444) só como referência de literatura, deixando claro que a base e o tamanho amostral são diferentes.
- Dado pequeno: usar validação cruzada estratificada e avisar sobre intervalo de confiança/overfitting. Usar o `passes_anotados.csv` como rótulo quando disponível.
- Figura: superfície de eXP em cada ponto do campo no instante de um passe (como Fig. 1 de Gomes et al.). → `resultados_xp.csv`.
- **Extras (só se sobrar tempo):** what-if (destino deslocado 3–6 m); **PAx = passes certos − ΣeXP**; `soccermap_mini.py`.

**Etapa 6 — Análise crítica e validação (35%, a maior nota)**
Gere a estrutura do README com seções e tabelas preenchidas por script, e deixe `<!-- ESCREVER -->` para minha análise sobre:
- Como as métricas (posse, passes, eXP/PAx) poderiam ser usadas em estatísticas reais de futebol (ex.: avaliar jogadores, tática, scouting).
- Limitações: trocas de ID, oclusões, bola pequena, câmera única/broadcast, erro de homografia, fora-de-quadro, amostra pequena.
- Comparação honesta com os dois artigos e o que da validação manual sustenta (ou não) as conclusões.

## 5. FORMA DE TRABALHAR

1. Use lista de tarefas e trabalhe **por etapas**, rodando e validando cada uma com um teste rápido (ex.: 10–30 s de vídeo) antes de processar o trecho inteiro.
2. Primeiro entregue um **MVP de ponta a ponta** (detecção → tracks → homografia simples → posse → passes), depois refine.
3. Quando precisar de algo que só eu posso fazer (clicar marcos, anotar passes, escrever análise), **pare e me diga exatamente o que fazer e como rodar a ferramenta**.
4. Código limpo: type hints, docstrings curtas, funções pequenas, CLI com `argparse`, caminhos relativos (o caminho do vídeo vem por argumento de linha de comando ou `config.yaml`, nunca fixo no código), logs claros.
5. Faça commits pequenos e descritivos, mas **só commite/pushe quando eu pedir**. Antes de qualquer commit, verifique `git status` e `git ls-files` para garantir que nenhum vídeo/arquivo pesado entrou.
6. Se algo no enunciado e nos artigos conflitar, o **enunciado prevalece**; avise-me.

## 6. APRESENTAÇÃO (09/11/2026 — 10 min, 1–2 apresentadores)

Gere em `slides/` um roteiro/esqueleto de PPT (pode ser `.pptx` via python-pptx) com: (1) problema e objetivo, (2) **pipeline proposto** (diagrama), (3) detecção/rastreamento/times com a comparação de modelos, (4) homografia e erro de reprojeção, (5) **métrica escolhida** e resultados (tabelas/figuras reais de `saidas/` e `figuras/`), (6) validação (P/R/F1 dos passes, tabela eXP), (7) análise crítica e usos, (8) limitações e próximos passos. Prepare também um **clipe de 1–2 min** da extração (vídeo com overlay: caixas, IDs, times, posse e minimapa) — salvo fora do repo.

## 7. CHECKLIST FINAL (rodar e me mostrar antes de eu entregar)

- [ ] `git ls-files` sem `.mp4`/pesos; `.gitignore` ok
- [ ] README tem URL + intervalo do vídeo e declaração de uso de IA
- [ ] README × `saidas/` consistentes (script de verificação passou)
- [ ] `passes_anotados.csv` existe e `validar_passes.py` rodou
- [ ] `saidas/<jogo>/` com todos os CSV/JSON listados e `figuras/` preenchida
- [ ] Tabela de sensibilidade (1, 2, 3 m) e comparação de modelos YOLO presentes
- [ ] Tabela eXP (baseline / heurístico / logístico) presente
- [ ] Seções `<!-- ESCREVER -->` reportadas a mim como pendências
- [ ] Repo roda do zero seguindo só o README

Comece pela seção 0 (leitura + resumo + dúvidas).
