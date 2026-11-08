#!/usr/bin/env python3
"""
Figuras de validação da compensação de câmera: o modelo do campo (linhas)
reprojetado sobre quadros ENTRE as âncoras manuais.

  - camera_deriva_metodo_apoio.jpg : H0 · G_t do código de apoio (ORB só no gramado,
                                     camera_gramado.npy) — mostra a deriva;
  - camera_validacao.jpg            : método final (camera_refinada.npy).

Uso
---
python figuras_camera.py --video "videos/Psg Bayern - camara tactica_clip.mp4" --saida saidas/psg_bayern
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from homografia import aplicar_h, modelo_linhas

QUADROS = (750, 1250, 1750, 2250, 3250, 4250)   # 30, 50, 70, 90, 130, 170 s — longe das âncoras


def painel(video: str, Hs: dict[int, np.ndarray], titulo: str, arq: Path):
    cap = cv2.VideoCapture(video)
    pts, _ = modelo_linhas(0.3)
    out = []
    for f, H in Hs.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        pix = aplicar_h(np.linalg.inv(H), pts)
        for x, y in pix[np.isfinite(pix).all(1)]:
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), -1)
        cv2.putText(img, f"t = {f / 25:.0f} s", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 255), 5)
        out.append(cv2.resize(img, (800, 450)))
    cap.release()
    grade = np.vstack([np.hstack(out[i:i + 3]) for i in range(0, len(out), 3)])
    faixa = np.zeros((60, grade.shape[1], 3), np.uint8)
    cv2.putText(faixa, titulo, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.imwrite(str(arq), np.vstack([faixa, grade]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"→ {arq}")


def main():
    ap = argparse.ArgumentParser(description="Figuras de validação da câmera")
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    S = Path(a.saida)
    fig = S / "figuras"
    fig.mkdir(exist_ok=True)
    H0 = np.array(json.loads((S / "calibracao.json").read_text(encoding="utf-8"))["H0"])
    if (S / "camera_gramado.npy").exists():
        G = np.load(S / "camera_gramado.npy")
        painel(a.video, {f: H0 @ G[f] for f in QUADROS},
               "Metodo do codigo de apoio (ORB so no gramado + quadros-chave): deriva",
               fig / "camera_deriva_metodo_apoio.jpg")
    Hr = np.load(S / "camera_refinada.npy")
    painel(a.video, {f: Hr[f] for f in QUADROS},
           "Metodo final (ORB consecutivo + refino pelas linhas do campo)", fig / "camera_validacao.jpg")


if __name__ == "__main__":
    main()
