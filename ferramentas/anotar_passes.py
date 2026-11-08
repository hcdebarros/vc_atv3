#!/usr/bin/env python3
"""
Ferramenta manual — anotar passes assistindo ao vídeo (ground truth da validação).

Mostra o vídeo (o clipe ORIGINAL, sem as caixas do detector, para não enviesar a
anotação) e grava cada passe em passes_anotados.csv a cada tecla (salvamento
incremental — pode fechar e continuar depois: as anotações existentes são lidas).

Anote no instante em que a bola SAI do pé do passador.

Teclas
------
espaço   pausar / continuar            a / d   voltar / avançar 1 quadro
j / l    voltar / avançar 1 s           , / .   voltar / avançar 5 s
0 / 1    escolher o time do PASSADOR (cores em meta.json → times_bgr; aparecem no topo)
c        passe CERTO (chegou a um companheiro) no quadro atual
e        passe ERRADO (interceptado / saiu) no quadro atual
z        desfazer a última anotação        q   salvar e sair

Formato de saída (lido por validar_passes.py):
    t_s,frame,time,sucesso
    3.40,85,0,1

Uso
---
python ferramentas/anotar_passes.py --video "videos/Psg Bayern - camara tactica_clip.mp4" --saida saidas/psg_bayern
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

COLUNAS = ["t_s", "frame", "time", "sucesso"]


def salvar(anot: list[dict], arq: Path):
    pd.DataFrame(anot, columns=COLUNAS).sort_values("frame").to_csv(arq, index=False)


def cores_times(pasta: Path) -> dict[int, tuple[int, int, int]]:
    # Cores de exibição fixas (as mesmas das figuras): 0 = azul, 1 = vermelho
    return {0: (230, 120, 30), 1: (40, 40, 220)}


def nomes_times(pasta: Path) -> dict[int, str]:
    try:
        n = json.loads((pasta / "times_nomes.json").read_text(encoding="utf-8"))
        return {0: n.get("0", "time 0"), 1: n.get("1", "time 1")}
    except (OSError, ValueError):
        return {0: "time 0", 1: "time 1"}


def main():
    ap = argparse.ArgumentParser(description="Anotação manual de passes → passes_anotados.csv")
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", required=True, help="Pasta do jogo (saidas/<jogo>)")
    ap.add_argument("--largura", type=int, default=1440, help="Largura da janela")
    a = ap.parse_args()

    pasta = Path(a.saida)
    arq = pasta / "passes_anotados.csv"
    anot = pd.read_csv(arq).to_dict("records") if arq.exists() else []
    cores = cores_times(pasta)
    nomes = nomes_times(pasta)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f, pausado, time_sel = 0, True, 0
    janela = "Anotar passes (espaco, a/d, j/l, ,/., 0/1, c/e, z, q)"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    frame = None

    def ir_para(alvo: int):
        nonlocal f, frame
        f = max(0, min(n - 1, alvo))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if ok:
            frame = fr

    ir_para(0)
    while True:
        if not pausado:
            ok, fr = cap.read()
            if not ok:
                pausado = True
            else:
                frame, f = fr, f + 1
        esc = a.largura / frame.shape[1]
        vis = cv2.resize(frame, None, fx=esc, fy=esc)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 70), (0, 0, 0), -1)
        cor = cores.get(time_sel, (255, 255, 255))
        cv2.putText(vis, f"t={f / fps:7.2f}s  quadro {f}/{n}  {'PAUSADO' if pausado else ''}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"time do passador: {time_sel} = {nomes[time_sel]}   (0 = {nomes[0]}, 1 = {nomes[1]})", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
        cv2.putText(vis, f"anotados: {len(anot)}", (760, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        recentes = [r for r in anot if 0 <= f - r["frame"] < fps * 1.5]
        for r in recentes:
            txt = "CERTO" if r["sucesso"] else "ERRADO"
            cv2.putText(vis, f"passe {txt} (time {r['time']})", (vis.shape[1] - 380, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if r["sucesso"] else (0, 0, 255), 2)
        cv2.imshow(janela, vis)

        k = cv2.waitKey(int(1000 / fps) if not pausado else 30) & 0xFF
        if k == ord("q"):
            break
        elif k == ord(" "):
            pausado = not pausado
        elif k in (ord("a"), ord("d"), ord("j"), ord("l"), ord(","), ord(".")):
            passo = {"a": -1, "d": 1, "j": -int(fps), "l": int(fps), ",": -5 * int(fps), ".": 5 * int(fps)}
            pausado = True
            ir_para(f + passo[chr(k)])
        elif k in (ord("0"), ord("1")):
            time_sel = k - ord("0")
        elif k in (ord("c"), ord("e")):
            anot.append({"t_s": round(f / fps, 2), "frame": f, "time": time_sel, "sucesso": int(k == ord("c"))})
            salvar(anot, arq)
        elif k == ord("z") and anot:
            anot.pop()
            salvar(anot, arq)

    cap.release()
    cv2.destroyAllWindows()
    salvar(anot, arq)
    df = pd.DataFrame(anot, columns=COLUNAS)
    print(f"{len(df)} passes anotados ({int(df['sucesso'].sum()) if len(df) else 0} certos) → {arq}")


if __name__ == "__main__":
    main()
