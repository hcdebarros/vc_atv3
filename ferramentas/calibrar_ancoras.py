#!/usr/bin/env python3
"""
Ferramenta manual — calibrar QUADROS-ÂNCORA ao longo do clipe (re-ancoragem da homografia).

Por quê: a câmera faz pan/zoom grandes; encadear homografias entre quadros
deriva (Q4.3). Calibrando à mão um quadro a cada ~20 s, a projeção fica
"presa" às âncoras (ver homografia.py ancorar).

Como usar (para cada quadro mostrado):
  1. clique um ponto de cruzamento de linhas na IMAGEM (à esquerda) — use a lupa;
  2. clique o MARCO correspondente no DIAGRAMA do campo (à direita) — ele "gruda"
     no marco mais próximo;
  3. repita para ≥ 4 marcos bem espalhados (5–6 é melhor);
  4. ENTER = salvar e ir ao próximo quadro · u = desfazer o último par ·
     s = pular este quadro · q = salvar e sair.
O quadro 0 é importado de calibracao.json (já clicado com clicar_marcos.py).
Tudo é salvo em calibracao_ancoras.json a cada quadro (pode parar e continuar).

Uso
---
python ferramentas/calibrar_ancoras.py --video "videos/Psg Bayern - camara tactica_clip.mp4" --saida saidas/psg_bayern
python ferramentas/calibrar_ancoras.py ... --quadros 500,1000,1500      # só alguns
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from campo import COMPRIMENTO, LARGURA, MARCOS  # noqa: E402

IW, IH = 1200, 675          # imagem exibida
DW = 620                    # largura do painel do diagrama
DESC = DW / (COMPRIMENTO + 10)   # px por metro no diagrama
LUPA, ZOOM = 220, 4


def dpx(x: float, y: float) -> tuple[int, int]:
    return int((x + 5) * DESC), int((y + 5) * DESC)


def diagrama() -> np.ndarray:
    d = np.full((int((LARGURA + 10) * DESC), DW, 3), (60, 120, 55), np.uint8)
    c = (235, 235, 235)
    cv2.rectangle(d, dpx(0, 0), dpx(COMPRIMENTO, LARGURA), c, 1)
    cv2.line(d, dpx(COMPRIMENTO / 2, 0), dpx(COMPRIMENTO / 2, LARGURA), c, 1)
    cv2.circle(d, dpx(COMPRIMENTO / 2, LARGURA / 2), int(9.15 * DESC), c, 1)
    for x0, s in ((0, 1), (COMPRIMENTO, -1)):
        cv2.rectangle(d, dpx(x0, 13.84), dpx(x0 + s * 16.5, 54.16), c, 1)
        cv2.rectangle(d, dpx(x0, 24.84), dpx(x0 + s * 5.5, 43.16), c, 1)
    for nome, (x, y) in MARCOS.items():
        cv2.circle(d, dpx(x, y), 4, (0, 255, 255), -1)
    return d


def problema(pixels, marcos) -> str | None:
    """Motivo para não aceitar os pares (None = ok)."""
    rep = sorted({m for m in marcos if marcos.count(m) > 1})
    if rep:
        return f"marco(s) repetido(s): {rep} — cada marco só pode ser usado uma vez"
    if len(set(marcos)) < 4:
        return "são necessários ≥ 4 marcos diferentes"
    dst = np.array([MARCOS[m] for m in marcos], np.float64)
    # Colinearidade: menor valor singular dos pontos centrados (em metros)
    sv = np.linalg.svd(dst - dst.mean(0), compute_uv=False)
    if sv[-1] < 1.0:
        return "marcos (quase) todos numa mesma linha — inclua pontos fora dessa linha"
    H, _ = cv2.findHomography(np.array(pixels, np.float64), dst, 0)
    if H is None:
        return "homografia não pôde ser calculada com esses pontos"
    return None


def calcular(pixels, marcos) -> tuple[np.ndarray, np.ndarray]:
    src = np.array(pixels, np.float64)
    dst = np.array([MARCOS[m] for m in marcos], np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return H, np.linalg.norm(proj - dst, axis=1)


def sessao(frame: np.ndarray, f: int, base: np.ndarray, guia: dict | None):
    """Janela de cliques para um quadro. Retorna (pixels, marcos) ou None (pular) / 'sair'."""
    esc = IW / frame.shape[1]
    img = cv2.resize(frame, (IW, IH))
    pares, pend, mouse = [], None, [0, 0]
    jan = f"Ancora quadro {f} (t={f / 25:.0f}s) - imagem -> diagrama | ENTER prox | u desfaz | s pula | q sai"

    def cb(ev, x, y, *_):
        nonlocal pend
        mouse[:] = [x, y]
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        if x < IW and y < IH:
            pend = [x / esc, y / esc]
        elif x >= IW and pend is not None and y < base.shape[0]:
            mx, my = (x - IW) / DESC - 5, y / DESC - 5
            nome = min(MARCOS, key=lambda m: np.hypot(MARCOS[m][0] - mx, MARCOS[m][1] - my))
            pares.append((pend, nome))
            pend = None

    cv2.namedWindow(jan, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(jan, IW + DW, IH)
    cv2.setMouseCallback(jan, cb)
    while True:
        canvas = np.zeros((IH, IW + DW, 3), np.uint8)
        vis = img.copy()
        if guia is not None:   # linhas previstas pela âncora anterior (só orientação)
            for seg in guia["segs"]:
                cv2.line(vis, *[tuple(int(v * esc) for v in pt) for pt in seg], (0, 200, 255), 1)
        d = base.copy()
        for (u, v), nome in pares:
            q = (int(u * esc), int(v * esc))
            cv2.drawMarker(vis, q, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(vis, nome, (q[0] + 6, q[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(d, dpx(*MARCOS[nome]), 7, (0, 0, 255), 2)
        if pend is not None:
            cv2.drawMarker(vis, (int(pend[0] * esc), int(pend[1] * esc)), (255, 0, 255), cv2.MARKER_CROSS, 22, 2)
        canvas[:, :IW] = vis
        canvas[:d.shape[0], IW:] = d
        # Lupa em torno do mouse (resolução original)
        mx, my = mouse
        if mx < IW and my < IH:
            cx, cy = int(mx / esc), int(my / esc)
            r = LUPA // (2 * ZOOM)
            pad = cv2.copyMakeBorder(frame, r, r, r, r, cv2.BORDER_CONSTANT)
            rec = cv2.resize(pad[cy:cy + 2 * r, cx:cx + 2 * r], (LUPA, LUPA), interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(rec, (LUPA // 2, LUPA // 2), (0, 0, 255), cv2.MARKER_CROSS, 30, 1)
            y0 = d.shape[0] + 10
            if y0 + LUPA <= IH:
                canvas[y0:y0 + LUPA, IW + 10:IW + 10 + LUPA] = rec
        msg = (f"pares: {len(pares)}  " + ("-> agora clique o MARCO no diagrama" if pend is not None
                                             else "-> clique um cruzamento de linhas na imagem"))
        cv2.putText(canvas, msg, (IW + 10, IH - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(jan, canvas)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u"):
            if pend is not None:
                pend = None
            elif pares:
                pares.pop()
        elif k == ord("s"):
            cv2.destroyWindow(jan)
            return None
        elif k == ord("q"):
            cv2.destroyWindow(jan)
            return "sair" if len(pares) < 4 else ([p for p, _ in pares], [m for _, m in pares], "sair")
        elif k in (13, 10):
            if len(pares) >= 4:
                cv2.destroyWindow(jan)
                return [p for p, _ in pares], [m for _, m in pares]


def main():
    ap = argparse.ArgumentParser(description="Calibração manual de quadros-âncora")
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--quadros", default=None, help="Lista separada por vírgula (padrão: a cada 20 s)")
    ap.add_argument("--intervalo-s", type=float, default=20.0)
    a = ap.parse_args()

    pasta = Path(a.saida)
    arq = pasta / "calibracao_ancoras.json"
    anc = json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else {}
    cal0 = json.loads((pasta / "calibracao.json").read_text(encoding="utf-8"))
    H0, e0 = calcular(cal0["pixels"], cal0["marcos"])
    anc["0"] = {"pixels": cal0["pixels"], "marcos": cal0["marcos"], "H": H0.tolist(),
                "erro_reprojecao_m": e0.tolist()}

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if a.quadros:
        quadros = [int(q) for q in a.quadros.split(",")]
    else:
        passo = int(a.intervalo_s * fps)
        quadros = list(range(passo, n, passo)) + [n - 1]
    base = diagrama()
    pend = [q for q in quadros if str(q) not in anc]
    print(f"Quadros a calibrar: {pend} (já feitos: {sorted(int(k) for k in anc)})")

    for f in pend:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        while True:
            r = sessao(frame, f, base, None)
            if r is None or r == "sair":
                break
            motivo = problema(r[0], r[1])
            if motivo is None:
                break
            print(f"quadro {f}: NÃO aceito — {motivo}. Refaça este quadro.")
        if r is None:
            print(f"quadro {f}: pulado")
            continue
        if r == "sair":
            break
        pixels, marcos = r[0], r[1]
        H, e = calcular(pixels, marcos)
        anc[str(f)] = {"pixels": pixels, "marcos": marcos, "H": H.tolist(), "erro_reprojecao_m": e.tolist()}
        arq.write_text(json.dumps(anc, indent=2), encoding="utf-8")
        print(f"quadro {f}: {len(marcos)} marcos, erro médio {e.mean():.2f} m, máx {e.max():.2f} m "
              f"({', '.join(f'{m}={x:.2f}' for m, x in zip(marcos, e))})")
        if len(r) == 3:
            break
    arq.write_text(json.dumps(anc, indent=2), encoding="utf-8")
    cap.release()
    cv2.destroyAllWindows()
    print(f"{len(anc)} âncoras em {arq}. Próximo: python homografia.py ancorar --saida {pasta}")


if __name__ == "__main__":
    main()
