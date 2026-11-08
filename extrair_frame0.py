#!/usr/bin/env python3
"""
Etapa 1 — valida o vídeo local (já cortado fora do repositório) e salva frame0.jpg.

O vídeo NÃO é baixado aqui: ele foi obtido previamente (ver README) e fica fora do git.
Este script:
  1. abre o arquivo, imprime duração, fps, resolução e nº de quadros;
  2. avisa se a duração estiver fora da faixa 3–5 min exigida no enunciado;
  3. salva o primeiro quadro (frame0.jpg), usado na calibração da homografia;
  4. (opcional, --cortes) varre o vídeo em streaming e lista instantes com queda
     brusca na correlação de histogramas HSV entre quadros consecutivos —
     candidatos a corte de câmera/replay para conferência manual.

Uso
---
python extrair_frame0.py --video "videos/Psg Bayern - camara tactica_clip.mp4" --saida saidas/psg_bayern --cortes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm


def info_video(cap: cv2.VideoCapture) -> dict:
    """Propriedades básicas do vídeo aberto."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fps": fps, "n_frames": n, "duracao_s": n / fps,
            "largura": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "altura": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}


def histograma(frame) -> "cv2.Mat":
    hsv = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


def detectar_cortes(cap: cv2.VideoCapture, fps: float, limiar: float, n_total: int) -> list[dict]:
    """Varre o vídeo quadro a quadro (sem guardar na RAM) e lista quedas de correlação."""
    cortes, ant = [], None
    for f in tqdm(range(n_total), desc="Cortes", unit="quadro"):
        ok, frame = cap.read()
        if not ok:
            break
        h = histograma(frame)
        if ant is not None:
            corr = cv2.compareHist(ant, h, cv2.HISTCMP_CORREL)
            if corr < limiar:
                cortes.append({"frame": f, "t_s": round(f / fps, 2), "correlacao": round(float(corr), 3)})
        ant = h
    return cortes


def main():
    ap = argparse.ArgumentParser(description="Valida o vídeo local e salva frame0.jpg")
    ap.add_argument("--video", required=True, help="Caminho do vídeo (use aspas se tiver espaços)")
    ap.add_argument("--saida", required=True, help="Pasta de saída (ex.: saidas/psg_bayern)")
    ap.add_argument("--cortes", action="store_true", help="Procurar cortes de câmera")
    ap.add_argument("--limiar-corte", type=float, default=0.6, help="Correlação mínima entre quadros")
    a = ap.parse_args()

    video, pasta = Path(a.video), Path(a.saida)
    pasta.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir: {video}")

    info = info_video(cap)
    print(f"Vídeo     : {video.name}")
    print(f"Resolução : {info['largura']}x{info['altura']}")
    print(f"FPS       : {info['fps']:.2f}")
    print(f"Quadros   : {info['n_frames']}")
    print(f"Duração   : {info['duracao_s']:.1f} s ({info['duracao_s']/60:.2f} min)")
    if not 180 - 1 <= info["duracao_s"] <= 300 + 1:
        print("AVISO: duração fora da faixa de 3–5 min pedida no enunciado.")

    ok, frame0 = cap.read()
    if not ok:
        raise SystemExit("Não foi possível ler o primeiro quadro.")
    cv2.imwrite(str(pasta / "frame0.jpg"), frame0)
    print(f"frame0.jpg salvo em {pasta / 'frame0.jpg'}")

    if a.cortes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        info["cortes"] = detectar_cortes(cap, info["fps"], a.limiar_corte, info["n_frames"])
        info["limiar_corte"] = a.limiar_corte
        print(f"{len(info['cortes'])} possível(is) corte(s): "
              f"{[c['t_s'] for c in info['cortes']] or 'nenhum'}")
    cap.release()

    info["video_nome"] = video.name
    (pasta / "video_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Informações em {pasta / 'video_info.json'}")


if __name__ == "__main__":
    main()
