#!/usr/bin/env python3
"""
Esqueleto do PPT da apresentação (10 min) com figuras e tabelas REAIS de saidas/.

Os textos de análise ficam como "[ESCREVER] ..." para o grupo redigir; os números
das tabelas vêm dos arquivos gerados (mesmo princípio do README).

Uso
---
python slides/gerar_slides.py --saida saidas/psg_bayern
→ slides/P3_futebol_yolo_psg_bayern.pptx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

AZUL = RGBColor(0x14, 0x21, 0x3D)
CINZA = RGBColor(0x55, 0x5F, 0x6D)
VERMELHO = RGBColor(0xC8, 0x10, 0x2E)


def _titulo(sl, texto: str, sub: str | None = None):
    tb = sl.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    p.text = texto
    p.font.size, p.font.bold, p.font.color.rgb = Pt(32), True, AZUL
    if sub:
        q = tb.text_frame.add_paragraph()
        q.text = sub
        q.font.size, q.font.color.rgb = Pt(16), CINZA


def _bullets(sl, itens: list[str], x, y, w, h, tam=16):
    tf = sl.shapes.add_textbox(x, y, w, h).text_frame
    tf.word_wrap = True
    for i, t in enumerate(itens):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "• " + t
        p.font.size = Pt(tam)
        p.font.color.rgb = VERMELHO if t.startswith("[ESCREVER]") else AZUL
        p.space_after = Pt(6)


def _imagem(sl, arq: Path, x, y, w=None, h=None):
    if arq.exists():
        sl.shapes.add_picture(str(arq), x, y, width=w, height=h)
    else:
        _bullets(sl, [f"[figura pendente: {arq.name}]"], x, y, Inches(4), Inches(0.5), 12)


def _tabela(sl, df: pd.DataFrame, x, y, w, tam=11):
    linhas, cols = df.shape[0] + 1, df.shape[1]
    t = sl.shapes.add_table(linhas, cols, x, y, w, Emu(int(Inches(0.32)) * linhas)).table
    for j, c in enumerate(df.columns):
        t.cell(0, j).text = str(c)
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        for j, v in enumerate(r):
            t.cell(i, j).text = "—" if pd.isna(v) else (f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v))
    for i in range(linhas):
        for j in range(cols):
            for p in t.cell(i, j).text_frame.paragraphs:
                p.font.size = Pt(tam)


def main():
    ap = argparse.ArgumentParser(description="Esqueleto de slides com resultados reais")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--arquivo", default="slides/P3_futebol_yolo_psg_bayern.pptx")
    a = ap.parse_args()

    S = Path(a.saida)
    F = S / "figuras"
    ler = lambda n: pd.read_csv(S / n) if (S / n).exists() else None  # noqa: E731
    lerj = lambda n: json.loads((S / n).read_text(encoding="utf-8")) if (S / n).exists() else None  # noqa: E731

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    branco = prs.slide_layouts[6]
    W = Inches(12.3)

    # 1 — Problema e objetivo
    sl = prs.slides.add_slide(branco)
    _titulo(sl, "Prática 3 — Futebol Analytics com YOLO", "PSG × Bayern · câmera tática · 18:20–21:20")
    _bullets(sl, ["Do vídeo de transmissão às métricas de jogo: posse, passes e eXP (expected pass)",
                  "Grupo: Helder · Lucas Rafael · José Kevin · Lucas Jerônimo · Felipe",
                  "[ESCREVER] Problema: por que medir posse/passes a partir de vídeo (sem dados de tracking)?"],
             Inches(0.5), Inches(1.6), Inches(6.2), Inches(4))
    _imagem(sl, S / "frame0.jpg", Inches(6.9), Inches(1.6), w=Inches(6))

    # 2 — Pipeline
    sl = prs.slides.add_slide(branco)
    _titulo(sl, "Pipeline proposto")
    etapas = ["Vídeo 3 min\n1080p25", "YOLOv8m\npessoa + bola", "ByteTrack\n(IDs)", "Times\ncor por track",
              "Kalman\nbola", "Homografia\nH0·Gₜ (m)", "Métricas\nposse · passes", "eXP\n+ validação"]
    for i, e in enumerate(etapas):
        x = Inches(0.4 + i * 1.6)
        caixa = sl.shapes.add_shape(1, x, Inches(2.6), Inches(1.4), Inches(1.3))
        caixa.fill.solid()
        caixa.fill.fore_color.rgb = VERMELHO if i == len(etapas) - 1 else AZUL
        caixa.text_frame.text = e
        for p in caixa.text_frame.paragraphs:
            p.font.size, p.font.color.rgb = Pt(13), RGBColor(255, 255, 255)
    _bullets(sl, ["Saídas em saidas/psg_bayern/: tracks.csv · bola.csv · calibracao.json · tracks_campo.csv · "
                  "passes.csv · metricas_times.json · resultados_xp.csv",
                  "[ESCREVER] Ajustes do grupo: filtro de gramado, times por track, regra do árbitro, bola fora do campo"],
             Inches(0.5), Inches(4.4), W, Inches(2))

    # 3 — Detecção / rastreamento / times
    sl = prs.slides.add_slide(branco)
    _titulo(sl, "Detecção, rastreamento e times", "Mesmo subtrecho de 30 s, 5 modelos × 2 resoluções")
    d = ler("comparar_yolo.csv")
    if d is not None:
        _tabela(sl, d[["modelo", "imgsz", "fps", "pessoas_por_quadro", "ids_unicos_pessoa",
                       "pct_quadros_bola_conf0.10"]].rename(columns={
                           "pessoas_por_quadro": "pessoas/q", "ids_unicos_pessoa": "IDs",
                           "pct_quadros_bola_conf0.10": "% bola (≥0,10)"}),
                Inches(0.5), Inches(1.6), Inches(6.4))
    _imagem(sl, F / "comparar_yolo" / "yolov8m_1280.jpg", Inches(7.2), Inches(1.6), w=Inches(5.7))
    _bullets(sl, ["[ESCREVER] Escolha do yolov8m@1280 e trocas de ID observadas"],
             Inches(7.2), Inches(5.2), Inches(5.7), Inches(1.5))

    # 4 — Homografia
    sl = prs.slides.add_slide(branco)
    _titulo(sl, "Do pixel ao campo — homografia H0 · Gₜ")
    _imagem(sl, S / "calibracao_grade.jpg", Inches(0.5), Inches(1.5), w=Inches(7.4))
    cal = lerj("calibracao.json")
    if cal and "erro_reprojecao_m" in cal:
        _tabela(sl, pd.DataFrame({"marco": cal["marcos"], "erro (m)": [round(e, 2) for e in cal["erro_reprojecao_m"]]}),
                Inches(8.2), Inches(1.5), Inches(4.6))
    _bullets(sl, ["[ESCREVER] ORB + RANSAC no gramado (sem jogadores) para o pan; deriva; bola no ar"],
             Inches(8.2), Inches(5.4), Inches(4.6), Inches(1.5))

    # 5 — Métrica escolhida e resultados
    sl = prs.slides.add_slide(branco)
    _titulo(sl, "Métricas de jogo", "[ESCREVER] qual métrica o grupo destaca e por quê")
    m = lerj("metricas_times.json")
    nomes = lerj("times_nomes.json") or {"0": "time 0", "1": "time 1"}
    if m:
        _tabela(sl, pd.DataFrame([{"time": nomes[t], "posse %": m[f"time{t}"]["posse_pct"],
                                   "passes": m[f"time{t}"]["passes"], "certos": m[f"time{t}"]["passes_certos"],
                                   "dist. total (m)": m[f"time{t}"]["distancia_total"]} for t in ("0", "1")]),
                Inches(0.5), Inches(1.6), Inches(6))
    _imagem(sl, F / "mapa_passes.png", Inches(0.5), Inches(3.0), h=Inches(4.2))
    _imagem(sl, F / "mapa_calor_times.png", Inches(6.8), Inches(1.6), w=Inches(6.2))
    _imagem(sl, F / "posse_linha_tempo.png", Inches(6.8), Inches(4.4), w=Inches(6.2))

    # 6 — Validação
    sl = prs.slides.add_slide(branco)
    _titulo(sl, "Validação manual e eXP")
    v = lerj("validacao_passes.json")
    if v:
        _tabela(sl, pd.DataFrame([{k: v[k] for k in ("detectados", "anotados", "casados", "precisao",
                                                     "revocacao", "f1")}]), Inches(0.5), Inches(1.5), Inches(6.3))
    xp = ler("resultados_xp_rotulos_detector.csv")
    if xp is None:
        xp = ler("resultados_xp.csv")
    if xp is not None:
        xp = xp.rename(columns={xp.columns[0]: "modelo"})[["modelo", "AUC-ROC (↑)", "Brier (↓)", "Log-Loss (↓)"]]
        xp["modelo"] = xp["modelo"] + " (rót. detector)"
        xp.loc[len(xp)] = ["SoccerMap (Gomes et al.) — ref.", 0.622, 0.138, 0.444]
        _tabela(sl, xp, Inches(0.5), Inches(2.9), Inches(6.3))
    _imagem(sl, F / "superficie_xp.png", Inches(7.0), Inches(1.5), w=Inches(6))
    _bullets(sl, ["[ESCREVER] O que a validação sustenta (e o que não)"], Inches(0.5), Inches(5.6),
             Inches(6.3), Inches(1.2))

    # 7 — Sensibilidade
    sl = prs.slides.add_slide(branco)
    _titulo(sl, "Sensibilidade ao raio de posse (1, 2, 3 m)")
    s = ler("sensibilidade_raio.csv")
    if s is not None:
        _tabela(sl, s, Inches(0.5), Inches(1.5), W)
    _imagem(sl, F / "sensibilidade_raio.png", Inches(2.5), Inches(3.2), w=Inches(8))

    # 8 — Análise crítica / limitações
    sl = prs.slides.add_slide(branco)
    _titulo(sl, "Análise crítica, usos e limitações")
    _bullets(sl, ["[ESCREVER] Usos: avaliação de passadores (PAx), tática (mapas), scouting a partir de vídeo",
                  "[ESCREVER] Limitações: trocas de ID, bola pequena/no ar, câmera com pan (jogadores fora do quadro), "
                  "erro de homografia, só 3 min de jogo",
                  "[ESCREVER] Comparação com Ahmad Yousef et al. (Kalman) e Gomes et al. (eXP/SoccerMap)",
                  "[ESCREVER] Próximos passos"], Inches(0.5), Inches(1.6), W, Inches(5))

    # 9 — Vídeo
    sl = prs.slides.add_slide(branco)
    _titulo(sl, "Demonstração (1–2 min)", "clipe_apresentacao.mp4 — caixas, IDs, times, posse e minimapa")
    _bullets(sl, ["[ESCREVER] Inserir o vídeo (Inserir → Vídeo) ou link do Google Drive"],
             Inches(0.5), Inches(1.8), W, Inches(1))

    Path(a.arquivo).parent.mkdir(parents=True, exist_ok=True)
    prs.save(a.arquivo)
    print(f"Slides salvos em {a.arquivo} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
