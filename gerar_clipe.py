#!/usr/bin/env python3
"""
Clipe de apresentação (1–2 min) com overlay da extração: caixas + IDs + times,
bola (detectada × Kalman), dono da bola, posse acumulada, passes e minimapa.

O vídeo gerado fica FORA do git (*.mp4 no .gitignore) — envie pelo Google Drive.

Uso
---
python gerar_clipe.py --video "videos/Psg Bayern - camara tactica_clip.mp4" --saida saidas/psg_bayern \
    --inicio-s 0 --fim-s 90
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from campo import COMPRIMENTO, LARGURA

COR = {0: (216, 110, 30), 1: (40, 40, 214), -2: (0, 215, 255), -1: (150, 150, 150)}  # BGR: 0=PSG azul, 1=Bayern vermelho


def minimapa(larg: int) -> tuple[np.ndarray, float]:
    """Desenho do campo (vista de cima) e escala px/m."""
    esc = larg / (COMPRIMENTO + 6)
    alt = int((LARGURA + 6) * esc)
    img = np.full((alt, larg, 3), (60, 125, 58), np.uint8)
    p = lambda x, y: (int((x + 3) * esc), int((y + 3) * esc))  # noqa: E731
    cv2.rectangle(img, p(0, 0), p(COMPRIMENTO, LARGURA), (255, 255, 255), 1)
    cv2.line(img, p(COMPRIMENTO / 2, 0), p(COMPRIMENTO / 2, LARGURA), (255, 255, 255), 1)
    cv2.circle(img, p(COMPRIMENTO / 2, LARGURA / 2), int(9.15 * esc), (255, 255, 255), 1)
    for x0, s in ((0, 1), (COMPRIMENTO, -1)):
        cv2.rectangle(img, p(x0, LARGURA / 2 - 20.16), p(x0 + s * 16.5, LARGURA / 2 + 20.16), (255, 255, 255), 1)
    return img, esc


def main():
    ap = argparse.ArgumentParser(description="Clipe com overlay para a apresentação")
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--inicio-s", type=float, default=0.0)
    ap.add_argument("--fim-s", type=float, default=90.0)
    ap.add_argument("--largura", type=int, default=1280, help="Largura do vídeo de saída")
    ap.add_argument("--arquivo", default="clipe_apresentacao.mp4")
    a = ap.parse_args()

    pasta = Path(a.saida)
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    fps = meta["fps"]
    tr = pd.read_csv(pasta / "tracks.csv")
    jog = tr[tr["classe"] != "bola"].copy()
    moda = jog[jog["time"] != -1].groupby("track_id")["time"].agg(lambda s: s.value_counts().idxmax())
    jog["time_f"] = jog["track_id"].map(moda).fillna(-1).astype(int)
    jog_f = {f: g for f, g in jog.groupby("frame")}
    bola = pd.read_csv(pasta / "bola.csv").set_index("frame")
    campo = pd.read_csv(pasta / "tracks_campo.csv") if (pasta / "tracks_campo.csv").exists() else None
    campo_f = {f: g for f, g in campo.groupby("frame")} if campo is not None else {}
    bola_c = (pd.read_csv(pasta / "bola_campo.csv").set_index("frame")
              if (pasta / "bola_campo.csv").exists() else None)
    posse = (pd.read_csv(pasta / "posse_por_quadro.csv").set_index("frame")
             if (pasta / "posse_por_quadro.csv").exists() else None)
    passes = pd.read_csv(pasta / "passes.csv") if (pasta / "passes.csv").exists() else pd.DataFrame()
    nomes = json.loads((pasta / "times_nomes.json").read_text(encoding="utf-8")) \
        if (pasta / "times_nomes.json").exists() else {"0": "time 0", "1": "time 1"}

    f0, f1 = int(a.inicio_s * fps), int(a.fim_s * fps)
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    W, H = meta["largura"], meta["altura"]
    esc = a.largura / W
    out = cv2.VideoWriter(str(pasta / a.arquivo), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (a.largura, int(H * esc)))
    mm_base, mm_esc = minimapa(int(a.largura * 0.30))
    cont = {0: 0, 1: 0}

    for f in tqdm(range(f0, f1), desc="Clipe"):
        ok, frame = cap.read()
        if not ok:
            break
        dono = None
        if posse is not None and f in posse.index:
            r = posse.loc[f]
            dono = r["dono"] if pd.notna(r["dono"]) else None
            if pd.notna(r["time_posse"]):
                cont[int(r["time_posse"])] += 1
        for r in jog_f.get(f, pd.DataFrame()).itertuples():
            c = COR.get(r.time_f, COR[-1])
            esp = 4 if dono is not None and r.track_id == dono else 2
            cv2.rectangle(frame, (r.x1, r.y1), (r.x2, r.y2), c, esp)
            cv2.putText(frame, f"#{r.track_id}", (r.x1, r.y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        if f in bola.index:
            b = bola.loc[f]
            cor_b = (255, 255, 255) if b["fonte"] == "detectada" else (0, 255, 255)
            cv2.circle(frame, (int(b["px"]), int(b["py"])), 12, cor_b, 3)

        vis = cv2.resize(frame, None, fx=esc, fy=esc)
        # Painel de posse + passe em andamento
        tot = max(cont[0] + cont[1], 1)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (20, 20, 20), -1)
        txt = (f"t={f / fps:5.1f}s   posse: {nomes['0']} {100 * cont[0] / tot:4.1f}%  |  "
               f"{nomes['1']} {100 * cont[1] / tot:4.1f}%")
        cv2.putText(vis, txt, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if len(passes):
            ativos = passes[(passes["frame_ini"] <= f) & (f <= passes["frame_fim"] + int(fps * 0.6))]
            for p in ativos.itertuples():
                ok_ = bool(p.sucesso)
                cv2.putText(vis, f"PASSE {'CERTO' if ok_ else 'ERRADO'} #{p.passador}->#{p.tocou_em}",
                            (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0) if ok_ else (0, 0, 255), 2)
        # Minimapa
        mm = mm_base.copy()
        for r in campo_f.get(f, pd.DataFrame()).itertuples():
            if pd.isna(r.x_m):
                continue
            t = int(moda.get(r.track_id, -1))
            cv2.circle(mm, (int((r.x_m + 3) * mm_esc), int((r.y_m + 3) * mm_esc)), 4, COR.get(t, COR[-1]), -1)
        if bola_c is not None and f in bola_c.index and pd.notna(bola_c.loc[f, "x_m"]):
            bx, by = bola_c.loc[f, "x_m"], bola_c.loc[f, "y_m"]
            cv2.circle(mm, (int((bx + 3) * mm_esc), int((by + 3) * mm_esc)), 4, (255, 255, 255), -1)
        y0 = vis.shape[0] - mm.shape[0] - 10
        x0 = vis.shape[1] - mm.shape[1] - 10
        vis[y0:y0 + mm.shape[0], x0:x0 + mm.shape[1]] = mm
        out.write(vis)
    cap.release()
    out.release()
    print(f"Clipe salvo em {pasta / a.arquivo} (fora do git — enviar pelo Google Drive)")


if __name__ == "__main__":
    main()
