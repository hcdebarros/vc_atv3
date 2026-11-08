# Respostas Q1–Q6 — RASCUNHO GERADO POR IA

> **Atenção, grupo:** este arquivo foi escrito pelo assistente de IA (Claude Opus 5.5, Claude Code) a pedido do grupo, a partir dos arquivos em `saidas/psg_bayern/`. O enunciado pede análise e README **autorais**: revisem, corrijam e reescrevam com as palavras do grupo antes de colar nas seções `<!-- ESCREVER -->` do README, e mantenham o registro na Declaração de Uso de IA. Os números abaixo valem para a execução atual; se o pipeline for rodado de novo, confiram nas tabelas AUTO do README.
>
> Itens que dependem de **assistir ao vídeo** (Q3.1) foram estimados a partir dos dados e estão marcados como "conferir".

---

## Parte 1 — Aquisição

**Q1.1 — Escolha do jogo e do trecho.**
Escolhemos PSG × Bayern (Champions League) numa **câmera tática**: plano alto e aberto, que mostra quase toda a largura do campo, sem replays nem cortes (o detector de cortes por histograma não encontrou nenhum em 180 s) e sem placar sobreposto à imagem. Os uniformes contrastam bem — PSG azul-marinho, Bayern vermelho — e a iluminação noturna é artificial e uniforme. Pontos que atrapalham: (i) a câmera faz **pan de um gol ao outro, com zoom** (o centro da imagem se desloca até ~2.000 px em relação ao quadro 0), o que exigiu compensação de movimento; (ii) por ser alta e aberta, os jogadores são pequenos (~61 px de altura) e a bola menor ainda (~13 px); (iii) há sombras no gramado e a comissão técnica/gandulas aparecem no gramado junto à lateral.

**Q1.2 — Resolução, fps, quadros e custo.**
1920 × 1080, 25 fps, 4.500 quadros (180 s). No nosso hardware (RTX 5060 Ti 8 GB, PyTorch CUDA 12.8), a detecção + rastreamento com `yolov8m` a 1280 px e a predição extra da bola rodou a **13,2 fps** gravando o vídeo anotado (~5,7 min para o trecho); sem gravar vídeo, a comparação de modelos mediu 19,6 fps. As outras etapas são de CPU: classificação de times por track (~3 min), homografias entre quadros consecutivos (~5 min) e refino pelas linhas (~5 min). O pipeline inteiro roda em ~20 min — viável para trechos curtos, mas não em tempo real.

## Parte 2 — Detecção

**Q2.1 — Comparação dos modelos e escolha.**
No mesmo subtrecho de 30 s (tabela `comparar_yolo.csv`), a resolução pesou mais que o tamanho do modelo: a 640 px nenhum modelo passou de 13% dos quadros com bola (conf ≥ 0,10). A 1280 px, o **`yolov8m`** teve o melhor equilíbrio: 59,5% dos quadros com bola (conf ≥ 0,10) e 31,2% com conf ≥ 0,25 (o maior entre todos), 27,6 pessoas/quadro e **127 IDs únicos** — contra 221 do `yolov8n`, o que indica menos fragmentação de tracks. Os `yolo11m/l` não superaram o `yolov8m` na bola e foram mais lentos. Custo: 19,6 fps, suficiente para o trecho. Por isso escolhemos `yolov8m` @ 1280.

**Q2.2 — Por que a bola é mais difícil?**
Na câmera tática a bola mede ~13 × 12 px, contra ~61 px de altura de um jogador; a 640 px de entrada ela vira 4–5 px, abaixo do que o detector resolve. Ela também sofre borrão de movimento nos passes, fica oculta por pernas e corpos e se confunde com objetos brancos pequenos (cabeças, garrafas, letras das placas). Isso é o mesmo fenômeno de Ahmad Yousef et al. (2025, Tabela 2): F1 da bola ~0,31 sem filtro de Kalman, contra > 0,9 para jogadores. Um detalhe do nosso caso: o ByteTrack praticamente descarta a bola — só **4,1%** dos quadros tiveram bola vinda do `track()` (detecções de baixa confiança não viram track); por isso a bola é detectada por um `predict()` separado com confiança baixa.

**Q2.3 — Efeito de `imgsz` e `conf` da bola.**
Dobrar a resolução de 640 para 1280 px multiplicou a detecção da bola (yolov8m: 12,7% → 59,5% dos quadros com conf ≥ 0,10), ao custo de ~30% de velocidade. Baixar o limiar de 0,25 para 0,05 aumenta a cobertura (31% → 77%), mas traz falsos positivos: no trecho inteiro, **2.744 das 4.363 candidatas (63%)** foram projetadas **fora do campo** — placas de publicidade, torcida e as garrafas de água da área técnica. Por isso usamos conf 0,10 e deixamos o filtro de campo + o gating do Kalman escolherem a candidata coerente. Exemplos visuais: `figuras/comparar_yolo/`.

## Parte 3 — Rastreamento e times

**Q3.1 — Trocas de ID** *(estimativa pelos dados — conferir no `rastreado.mp4`)*.
Há ~24,6 pessoas visíveis por quadro, mas o ByteTrack gerou **365 IDs** em 3 min; a duração mediana de um ID é de só 2 s, 146 IDs duram menos de 1 s e apenas 14 IDs duram ≥ 60 s. Contando tracks que **nascem longe da borda da imagem** (logo, não são jogadores entrando no quadro), são ~83 novos IDs por minuto — um limite superior para as trocas. As situações típicas: disputas e aglomerações (um jogador oculto por outro), jogadores pequenos no lado oposto da câmera (detecção intermitente) e o pan da câmera (jogadores saem e voltam ao quadro com ID novo). `<!-- conferir: assistir 1 min do rastreado.mp4 e contar -->`

**Q3.2 — O K-Means de cor funcionou?**
O classificador online do código de apoio funcionou parcialmente: ao assistir o vídeo, jogadores distantes do Bayern apareciam como PSG e o árbitro caía no time do PSG, porque o time de cada ID era fixado com ~10 recortes ruidosos. Refizemos a classificação **por track**: um descritor de cor (frações de vermelho, laranja, azul, branco, escuro…) é calculado em todos os recortes de cada ID e agrupado com K-Means k = 2 entre tracks. Os centróides ficaram nítidos: PSG = 23% azul + 32% escuro; Bayern = 45% vermelho + 43% laranja. Sobre o espaço (S·cos H, S·sin H, V): o matiz é um **ângulo** e o vermelho fica nas duas pontas da escala do OpenCV (H ≈ 0 e H ≈ 179). A média simples de pixels vermelhos daria H ≈ 90 — **ciano** —, justamente a cor errada para o Bayern. Tratando H como ângulo (cos/sin ponderados pela saturação), o vermelho vira um único ponto.

**Q3.3 — k = 3.**
Não resolveu sozinho. No classificador online, o 3º cluster virou "pixels escuros/sombra" (centróide BGR ≈ (19, 36, 20)), não o árbitro. Na classificação por track, k = 3 separou o **técnico de preto** e k = 4 separou **jogadores do PSG bem iluminados** — ou seja, o cluster extra captura a maior variação de cor que existir, não necessariamente o árbitro. A solução foi uma regra explícita: o uniforme da arbitragem é o único com fração relevante de ciano (≥ 0,09; PSG ≤ 0,01), o que separou árbitro e assistentes (36 tracks em "outros"). Risco: a regra é específica deste jogo/uniforme e precisa ser revista em outro vídeo.

## Parte 4 — Bola e homografia

**Q4.1 — Cobertura da bola antes e depois do Kalman.**
O YOLO teve candidatas em 62,1% dos quadros, mas só 32,2% tinham candidata **dentro do campo**. Depois do gating (que rejeitou detecções incoerentes com a trajetória em 373 quadros), o Kalman aceitou 1.077 quadros e **preencheu 2.355** por predição, chegando a **76,3% de cobertura**. Onde ele acerta: lacunas curtas (≤ 15 quadros = 0,6 s) em passes rasteiros, em que a bola segue quase em linha reta. Onde ele "inventa": antes do filtro de campo, o Kalman chegou a travar numa detecção falsa nas placas de publicidade e manter uma "bola" parada fora do campo; e em desvios/rebotes ele continua a reta por até 0,6 s. Note que mais da metade da trajetória final vem de predição, não de detecção.

**Q4.2 — Erro de reprojeção por marco.**
No frame0 os 8 marcos tiveram erro de 0,09 a 0,32 m (média 0,16 m), melhor que o exemplo da disciplina (≤ 0,41 m). Os maiores erros foram nas **quinas das áreas do lado direito** (`ga_dir_sup` 0,32 m, `pa_dir_sup` 0,23 m), os pontos mais distantes da câmera: lá 1 pixel equivale a muito mais metros, então um erro de clique de 1–2 px vira dezenas de centímetros. Os marcos da linha do meio, perto da câmera e bem definidos, ficaram em ~0,1 m.

**Q4.3 — Composição H0 · Gₜ e deriva.**
H0 leva pixels do quadro 0 ao campo; Gₜ leva pixels do quadro t ao quadro 0 (estimada por ORB + RANSAC). Compor H0 · Gₜ projeta o quadro t no campo — mas Gₜ é um **produto** de muitas estimativas, e cada uma tem um pequeno erro que se acumula (**deriva**). Neste vídeo, o método do código de apoio (ORB só no gramado, com quadros-chave) derivou dezenas de metros em 60 s (`figuras/camera_deriva_metodo_apoio.jpg`); testamos ORB e SIFT com várias máscaras e todos acumularam 2,5–4 m de erro médio em só 20 s. Para **re-ancorar**: (i) a cada quadro, ajustamos a homografia às **linhas do campo** detectadas (distância ponto-à-reta, mínimos quadrados regularizados), o que impede o erro de se acumular; (ii) clicamos 10 quadros-âncora (a cada 20 s). Validação independente (inicializando só no quadro 0): erro médio de 0,14 a 1,27 m nas âncoras, com pior caso de 2,24 m aos 140 s. Na versão final as âncoras também reiniciam a estimativa.

**Q4.4 — Por que o pé? E a bola no ar?**
A homografia só vale para pontos **no plano do gramado**. O pé do jogador (centro da base do bbox) está no chão; o centro do bbox está ~1 m acima, e projetá-lo como se estivesse no chão o empurra para longe da câmera — alguns metros de erro nesta câmera alta. A bola no ar viola a mesma hipótese: num lançamento ela aparece mais "alta" na imagem e é projetada **mais longe da câmera** do que está, distorcendo distância e destino do passe. Nossos passes são majoritariamente rasteiros, mas lançamentos longos têm a posição projetada errada (e podem ser descartados pelo filtro de campo).

## Parte 5 — Métricas

**Q5.1 — Posse, distância e velocidade são plausíveis?**
Posse: PSG 58,9% × Bayern 41,1% (raio 2 m). Nos 13 jogadores rastreados por ≥ 60 s, a distância por minuto teve mediana de **159 m/min** e máximo de 175 m/min — acima dos ~110–120 m/min médios de um jogo inteiro. Parte é real (3 min de jogo corrido são mais intensos que a média de 90 min), mas parte é **ruído**: o "tremor" do bbox e o erro da homografia somam deslocamentos falsos (a mediana móvel atenua, não elimina). A velocidade máxima (mediana 26,4 km/h, máximo 32,3 km/h) é plausível para sprints de elite (~33–36 km/h). A distância **por time** é subestimada, porque cada ID cobre só parte do trecho (trocas de ID, jogadores fora do quadro com o pan).

**Q5.2 — Sensibilidade ao raio de posse.**
| raio | quadros com dono | posse PSG/Bayern | passes | % certos | F1 da validação |
|---|---|---|---|---|---|
| 1 m | 25% | 52,5 / 47,5 | 38 | 73,7% | 0,50 |
| 2 m | 55% | 58,9 / 41,1 | 68 | 54,4% | 0,44 |
| 3 m | 69% | 55,5 / 44,5 | 80 | 56,2% | 0,37 |

Raio maior "dá" a bola a mais jogadores, inclusive adversários que só passam perto dela: surgem mais passes e mais "passes errados" falsos, e o F1 cai. Com 1 m a precisão dos passes detectados sobe (0,55) e a % de certos fica mais realista, mas a revocação cai e só 25% dos quadros têm dono. A posse oscila ~6 pontos percentuais entre os raios — a métrica é sensível a um parâmetro arbitrário.

**Q5.3 — Validação: quais erros dominam?**
Contra 46 passes anotados à mão (tolerância de 1 s): **precisão 0,37, revocação 0,54, F1 0,44**; nos 25 casados, o rótulo certo/errado bate em 76%. Dominam os **falsos positivos (43)**: metade deles (23) são "passes errados" que na verdade são disputas, divididas ou a bola passando perto de um adversário — o detector registra uma "troca de dono" sem passe. Por isso ele marca só 54% dos passes como certos, contra **91%** na anotação. Os **falsos negativos (21)** são quase todos passes certos (19) e 14 deles do Bayern: passes curtos/rápidos em que a bola some (oclusão, lacuna maior que o Kalman cobre) ou em que receptor e passador estão a < 3 m. Há ainda 6 passes que casam no tempo mas com o **time errado** (posse atribuída ao time errado no instante do passe). Contradição relevante: a anotação mostra o **Bayern com mais passes** (27 × 19), enquanto a posse calculada favorece o PSG — o PSG tem mais IDs classificados (128 × 73, inclusive membros da comissão técnica de roupa escura), o que pode enviesar o "jogador mais próximo da bola" a favor dele.

**Q5.4 — Direção de ataque, trocas de ID e jogadores fora do quadro.**
A heurística estimou PSG → esquerda e conferimos pelos dados: o jogador mais avançado do PSG fica em média em x = 76 m (contra 70 m do Bayern) e o mais recuado do Bayern em 40,6 m (contra 43,3 m do PSG), ou seja, o PSG defende o gol da direita, onde também aparecem os goleiros detectados quando ele defende. Trocas de ID quebram a distância de um jogador em vários pedaços e criam "passes" falsos (o mesmo jogador com ID novo vira passador e receptor — o filtro de deslocamento ≥ 3 m atenua). Jogadores fora do quadro (pan) somem das métricas e da contagem de adversários na linha do passe. Correções propostas: re-identificação por aparência/número da camisa para juntar tracks do mesmo jogador; restrição física (um ID novo que aparece onde outro do mesmo time sumiu, < 1 s antes, é o mesmo jogador); e, idealmente, câmera fixa de campo inteiro ou dados de tracking.

## Parte 6 — eXP

**Q6.1 — Features × 17 canais de Gomes et al.**
| nossa feature | canal(is) análogo(s) na Tabela 1 |
|---|---|
| `distancia` | 3 (distância à bola), 17 (distância ao início do evento) |
| `adversarios_na_linha` | 11 (defensores entre bola e célula) |
| `pressao_passador` | 2 (ocupação do time defensor), localmente no passador |
| `espaco_destino` | 2 (ocupação defensiva) no destino |
| `progressao_x` | 5–7 (ângulos bola–gol) |
| `dist_destino_gol` | 4 (distância ao gol) |

Não conseguimos extrair com confiança: as **velocidades** (canais 8–9 e 12–16) — a da bola vem do Kalman em pixels, ruidosa e distorcida quando a bola sai do chão, e a dos jogadores é afetada por trocas de ID e tremor do bbox; a **ocupação completa** (canais 1–2), porque com o pan só parte dos 22 jogadores aparece; e o **instante exato do passe**, que no artigo vem de dados de evento e aqui é inferido pela troca de dono.

**Q6.2 — Comparação dos modelos e confiança.**
Com **rótulos manuais** sobram 25 passes, só 2 errados: não há como treinar baseline/logístico com validação cruzada, e a AUC do heurístico (0,74) com 2 negativos não tem significância nenhuma (um passe a mais muda tudo). Com **rótulos do detector** (68 passes, 31 "errados"): heurístico AUC 0,54, baseline de distância 0,64, logístico **0,79** (Brier 0,19, Log-Loss 0,59). Mas o logístico está **otimista por vazamento de rótulo**: "errado" significa "quem tocou depois é adversário", e `espaco_destino` mede a distância do adversário mais próximo ao ponto onde a bola chegou — o modelo lê parte da resposta (é a feature de maior peso, 1,07, contra 0,08 da distância). Com algumas dezenas de passes, os intervalos de confiança dessas métricas são enormes; a comparação com o SoccerMap (AUC 0,622 em ~7 mil passes, rastreamento óptico) é só de referência. Para melhorar: juntar passes de outros trechos/grupos (`--extra`) e medir o espaço no destino **antes** do passe (no quadro de saída), não onde a bola chegou.

**Q6.3 — Calibração.**
Com rótulos manuais, o heurístico prevê em média 0,78 contra 0,92 observado: **subestima** a chance de acerto (é conservador). Com rótulos do detector, o heurístico prevê 0,76 contra 0,54 observado — parece superestimar, mas é porque o rótulo do detector inventa passes errados; o logístico fica calibrado (0,55 × 0,54) por ter sido ajustado nesses mesmos rótulos. Ou seja, a calibração depende mais da qualidade dos rótulos que do modelo. No artigo, a média prevista (0,830) coincide com a observada (0,829).

**Q6.4 — A superfície de eXP faz sentido tático?**
Parcialmente. A superfície do heurístico é quente em espaços abertos e esfria perto dos adversários e atrás de linhas de marcação, como na Fig. 1 de Gomes et al. Já a do logístico com rótulos do detector é dominada por "buracos" frios em volta de cada adversário e quase não depende da distância (coeficiente 0,08) — passes de 40 m em espaço aberto aparecem tão seguros quanto passes curtos, o que é taticamente implausível: é o sintoma do vazamento de rótulo e da amostra pequena.

**Q6.5 — What-if, PAx e limitações.**
No PAx, o ID #16 (PSG) aparece no topo (+1,03), mas com **2 passes** — e um "jogador" aqui é um ID do rastreador, que se fragmenta; não dá para atribuir isso a uma pessoa nem tirar conclusão. No what-if (destino deslocado 3–6 m nos 2 passes errados rotulados à mão), um passe ganharia +0,44 de eXP com o receptor ~6 m mais aberto, e o outro só +0,04. Limitações de 3 min: poucas dezenas de passes, 1–2 por jogador, só 2 erros reais, IDs que não correspondem a jogadores, e métricas sensíveis a parâmetros (raio de posse). Servem para demonstrar o método, não para avaliar jogadores — para isso seriam necessários jogos inteiros, re-identificação e rótulos manuais em escala.
