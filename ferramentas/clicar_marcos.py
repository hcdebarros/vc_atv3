#!/usr/bin/env python3
"""
Ferramenta manual — clicar marcos do campo no frame0.jpg e calcular H0.

Abre o frame0.jpg numa janela OpenCV e pede, NA ORDEM, cada marco da lista.
Teclas: clique esquerdo = marcar · u = desfazer · s = PULAR o marco atual
(se ele não estiver visível) · q = terminar.
Depois de clicar, calcula H0 (homografia.py), grava calibracao.json com o
erro de reprojeção (em metros) por marco e a figura calibracao_grade.jpg
(grade de 5 m reprojetada sobre o frame0).

Marcos sugeridos para o vídeo PSG × Bayern (câmera tática, lado direito do campo):
  meio_sup, circulo_sup, centro, circulo_inf, meio_inf,   (linha do meio + círculo)
  canto_sup_dir,                                           (bandeirinha de escanteio, alto à direita)
  ga_dir_sup, pa_dir_sup                                   (quinas das áreas, alto à direita)
Nomes e coordenadas de todos os marcos: campo.MARCOS.

Uso
---
python ferramentas/clicar_marcos.py --saida saidas/psg_bayern
python ferramentas/clicar_marcos.py --saida saidas/psg_bayern --pontos meio_sup,centro,meio_inf,ga_dir_sup
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from campo import MARCOS  # noqa: E402
from homografia import calibrar  # noqa: E402

PADRAO = "meio_sup,circulo_sup,centro,circulo_inf,meio_inf,canto_sup_dir,ga_dir_sup,pa_dir_sup"


def clicar(img, marcos: list[str]) -> tuple[list[list[float]], list[str]]:
    """Janela de cliques; devolve pixels e os nomes efetivamente marcados."""
    pts: list[list[float] | None] = []
    janela = "Marcos: clique | u=desfazer | s=pular | q=terminar"
    escala = min(1.0, 1600 / img.shape[1])

    def ao_clicar(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < len(marcos):
            pts.append([x / escala, y / escala])

    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, int(img.shape[1] * escala), int(img.shape[0] * escala))
    cv2.setMouseCallback(janela, ao_clicar)
    while True:
        vis = cv2.resize(img, None, fx=escala, fy=escala)
        for p, nome in zip(pts, marcos):
            if p is None:
                continue
            q = (int(p[0] * escala), int(p[1] * escala))
            cv2.drawMarker(vis, q, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(vis, nome, (q[0] + 8, q[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        prox = marcos[len(pts)] if len(pts) < len(marcos) else "pronto! tecle q"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(vis, f"Proximo: {prox}   ({len(pts)}/{len(marcos)})", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow(janela, vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k == ord("s") and len(pts) < len(marcos):
            pts.append(None)
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()
    pares = [(p, m) for p, m in zip(pts, marcos) if p is not None]
    return [p for p, _ in pares], [m for _, m in pares]


def main():
    ap = argparse.ArgumentParser(description="Clicar marcos no frame0.jpg → calibracao.json")
    ap.add_argument("--saida", required=True, help="Pasta do jogo (contém frame0.jpg)")
    ap.add_argument("--pontos", default=PADRAO, help="Marcos, na ordem do clique (campo.MARCOS)")
    a = ap.parse_args()

    pasta = Path(a.saida)
    marcos = [m for m in a.pontos.split(",") if m]
    faltando = [m for m in marcos if m not in MARCOS]
    if faltando:
        raise SystemExit(f"Marcos desconhecidos: {faltando}. Opções: {list(MARCOS)}")
    img = cv2.imread(str(pasta / "frame0.jpg"))
    if img is None:
        raise SystemExit(f"frame0.jpg não encontrado em {pasta} (rode extrair_frame0.py).")

    pixels, nomes = clicar(img, marcos)
    if len(pixels) < 4:
        raise SystemExit("São necessários pelo menos 4 marcos (não colineares).")
    (pasta / "calibracao.json").write_text(json.dumps({"pixels": pixels, "marcos": nomes}, indent=2))
    calibrar(pasta, None, sem_janela=True)   # calcula H0, erro de reprojeção e grade


if __name__ == "__main__":
    main()
