#!/usr/bin/env python3
"""
Etapa 2 — compara tamanhos/versões de YOLO (COCO) no MESMO subtrecho de 30 s.

Para cada combinação (modelo, imgsz) mede, em streaming (quadro a quadro):
  - fps de inferência (YOLO + ByteTrack, como no pipeline real);
  - nº médio de pessoas por quadro (conf ≥ --conf);
  - % de quadros com bola detectada, para vários limiares de confiança da bola.
    A bola é detectada por um predict SEPARADO (classe 32, conf baixo), como no
    detectar_rastrear.py: o ByteTrack descarta detecções de baixa confiança e
    quase nunca devolve a bola (medimos isso também: coluna pct_bola_via_track);
  - nº médio de candidatas a bola por quadro (proxy de falsos positivos).
Salva também um quadro anotado por combinação (figuras/comparar_yolo/) para
inspecionar visualmente falsos positivos de bola.

Uso
---
python comparar_yolo.py --video "videos/Psg Bayern - camara tactica_clip.mp4" --saida saidas/psg_bayern \
    --modelos yolov8n.pt yolov8s.pt yolov8m.pt yolo11m.pt --imgsz 640 1280 --inicio-s 0 --dur-s 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

PESSOA, BOLA = 0, 32  # ids COCO
LIMIARES_BOLA = (0.05, 0.10, 0.25)


def avaliar(modelo_nome: str, imgsz: int, video: Path, f0: int, n: int, conf: float,
            device: str, dir_fig: Path, quadro_fig: int) -> dict:
    """Roda YOLO + ByteTrack em n quadros a partir de f0 e devolve as estatísticas."""
    modelo = YOLO(modelo_nome)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    pessoas, bola_max, n_cand, bola_track, ids = [], [], [], [], set()
    # aquecimento (não entra no tempo)
    ok, frame = cap.read()
    modelo.predict(frame, imgsz=imgsz, device=device, verbose=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in tqdm(range(n), desc=f"{modelo_nome}@{imgsz}", leave=False):
        ok, frame = cap.read()
        if not ok:
            break
        r = modelo.track(frame, persist=True, classes=[PESSOA, BOLA], conf=conf,
                         imgsz=imgsz, device=device, tracker="bytetrack.yaml", verbose=False)[0]
        b = r.boxes
        cls = b.cls.cpu().numpy().astype(int) if b is not None else np.array([])
        mp = cls == PESSOA
        pessoas.append(int(mp.sum()))
        bola_track.append(bool((cls == BOLA).any()))
        if b is not None and b.id is not None:
            ids.update(b.id.cpu().numpy().astype(int)[mp].tolist())
        rb = modelo.predict(frame, classes=[BOLA], conf=min(LIMIARES_BOLA), imgsz=imgsz,
                            device=device, verbose=False)[0]
        cf = rb.boxes.conf.cpu().numpy()
        bola_max.append(float(cf.max()) if len(cf) else 0.0)
        n_cand.append(int((cf >= 0.10).sum()))
        if i == quadro_fig:
            vis = r.plot(line_width=2, font_size=10)
            for (x1, y1, x2, y2), c in zip(rb.boxes.xyxy.cpu().numpy().astype(int), cf):
                cv2.circle(vis, ((x1 + x2) // 2, (y1 + y2) // 2), 14, (0, 255, 255), 2)
                cv2.putText(vis, f"bola {c:.2f}", (x2 + 6, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imwrite(str(dir_fig / f"{Path(modelo_nome).stem}_{imgsz}.jpg"), vis)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    cap.release()
    bola_max = np.array(bola_max)
    linha = {"modelo": modelo_nome, "imgsz": imgsz, "quadros": len(pessoas),
             "fps": round(len(pessoas) / dt, 1),
             "pessoas_por_quadro": round(float(np.mean(pessoas)), 2),
             "ids_unicos_pessoa": len(ids),
             "candidatas_bola_por_quadro": round(float(np.mean(n_cand)), 2),
             "pct_bola_via_track": round(100 * float(np.mean(bola_track)), 1)}
    for t in LIMIARES_BOLA:
        linha[f"pct_quadros_bola_conf{t:.2f}"] = round(100 * float((bola_max >= t).mean()), 1)
    return linha


def main():
    ap = argparse.ArgumentParser(description="Compara modelos YOLO no mesmo subtrecho")
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--modelos", nargs="+", default=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640, 1280])
    ap.add_argument("--inicio-s", type=float, default=0.0)
    ap.add_argument("--dur-s", type=float, default=30.0)
    ap.add_argument("--conf", type=float, default=0.30, help="Confiança mínima para pessoas")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.device == "cpu":
        print("AVISO: rodando em CPU — os fps não representam o hardware com GPU.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    pasta = Path(a.saida)
    dir_fig = pasta / "figuras" / "comparar_yolo"
    dir_fig.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    f0, n = int(a.inicio_s * fps), int(a.dur_s * fps)

    linhas = []
    for m in a.modelos:
        for s in a.imgsz:
            linhas.append(avaliar(m, s, Path(a.video), f0, n, a.conf, a.device, dir_fig, n // 2))
            print(linhas[-1])
    df = pd.DataFrame(linhas)
    df.insert(0, "subtrecho", f"{a.inicio_s:.0f}-{a.inicio_s + a.dur_s:.0f}s")
    df.to_csv(pasta / "comparar_yolo.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nTabela salva em {pasta / 'comparar_yolo.csv'}; quadros anotados em {dir_fig}/")


if __name__ == "__main__":
    main()
