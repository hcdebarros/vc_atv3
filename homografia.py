#!/usr/bin/env python3
"""
Do pixel ao campo: homografia + compensação do movimento da câmera.

Por que isto é necessário? (Ahmad Yousef et al., 2025, Fig. 8)
A câmera lateral introduz distorção de perspectiva: 1 pixel perto da câmera vale
poucos centímetros, 1 pixel do outro lado do campo vale quase meio metro.
Distâncias, velocidades e o eXP precisam ser calculados em METROS.

Ideia
-----
1. Calibração (uma vez, no frame0.jpg): clicamos ≥ 4 marcos conhecidos do campo
   (ver campo.MARCOS) e estimamos H0 tal que  [x_m, y_m, 1]ᵀ ~ H0 · [u, v, 1]ᵀ.
2. Câmera: a transmissão faz pan/zoom. Estimamos a homografia G_t que leva o
   quadro t ao quadro 0 (ORB + RANSAC só no gramado, sem jogadores e placar),
   casando cada quadro com um quadro-chave para reduzir o acúmulo de erro.
3. Projeção: ponto do quadro t vai ao campo por  H0 · G_t · [u, v, 1]ᵀ.

Subcomandos
-----------
python homografia.py calibrar --saida ../saidas/jogo1 --pontos meio_sup,centro,meio_inf,ga_dir_sup,ga_dir_inf
python homografia.py camera   --saida ../saidas/jogo1
python homografia.py projetar --saida ../saidas/jogo1

Sem interface gráfica (ex.: Colab)? Crie <saida>/calibracao.json à mão com
{"pixels": [[u, v], ...], "marcos": ["meio_sup", ...]} e rode "calibrar --sem-janela".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from campo import COMPRIMENTO, LARGURA, MARCOS


# ── Utilitário ─────────────────────────────────────────────────────────────

def aplicar_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Aplica a homografia H (3×3) a pontos (N×2) em coordenadas cartesianas."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


# ── 1. Calibração ──────────────────────────────────────────────────────────

def _clicar_pontos(img: np.ndarray, marcos: list[str]) -> list[list[float]]:
    """Janela OpenCV: clique nos marcos na ordem pedida. 'u' desfaz, 'q' termina."""
    pts: list[list[float]] = []
    janela = "Calibracao - clique nos marcos (u = desfazer, q = sair)"

    def ao_clicar(evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN and len(pts) < len(marcos):
            pts.append([float(x), float(y)])

    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(janela, ao_clicar)
    while True:
        vis = img.copy()
        for (x, y), nome in zip(pts, marcos):
            cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(vis, nome, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        prox = marcos[len(pts)] if len(pts) < len(marcos) else "pronto! tecle q"
        cv2.putText(vis, f"Proximo: {prox}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow(janela, vis)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla == ord("u") and pts:
            pts.pop()
        elif tecla == ord("q"):
            break
    cv2.destroyAllWindows()
    return pts


def calibrar(pasta: Path, marcos: list[str] | None, sem_janela: bool):
    arq = pasta / "calibracao.json"
    if sem_janela:
        cal = json.loads(arq.read_text(encoding="utf-8"))
        pixels, marcos = cal["pixels"], cal["marcos"]
    else:
        desconhecidos = [m for m in marcos if m not in MARCOS]
        if desconhecidos:
            raise SystemExit(f"Marcos desconhecidos: {desconhecidos}. Opções: {list(MARCOS)}")
        pixels = _clicar_pontos(cv2.imread(str(pasta / "frame0.jpg")), marcos)
        marcos = marcos[:len(pixels)]

    if len(pixels) < 4:
        raise SystemExit("São necessários pelo menos 4 pontos (não colineares).")

    src = np.array(pixels, dtype=np.float64)
    dst = np.array([MARCOS[m] for m in marcos], dtype=np.float64)
    H0, _ = cv2.findHomography(src, dst, cv2.RANSAC if len(src) > 4 else 0, 1.0)

    # Erro de reprojeção: quão bem H0 leva cada clique ao seu marco (em metros)
    erro = np.linalg.norm(aplicar_h(H0, src) - dst, axis=1)
    json.dump({"pixels": pixels, "marcos": marcos, "H0": H0.tolist(),
               "erro_reprojecao_m": erro.tolist()}, open(arq, "w"), indent=2)
    print(f"H0 salva em {arq}")
    for m, e in zip(marcos, erro):
        print(f"  {m:18s} erro = {e:.2f} m")

    # Imagem de conferência: grade do campo reprojetada sobre o frame0
    img = cv2.imread(str(pasta / "frame0.jpg"))
    Hinv = np.linalg.inv(H0)
    linhas = [[[x, 0], [x, LARGURA]] for x in np.arange(0, COMPRIMENTO + 0.1, 5)]
    linhas += [[[0, y], [COMPRIMENTO, y]] for y in np.arange(0, LARGURA + 0.1, 5)]
    for seg in linhas:
        # clip evita overflow quando a linha projetada cai muito fora da imagem
        a, b = np.clip(aplicar_h(Hinv, seg), -1e4, 1e4).astype(int)
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 255, 255), 1)
    cv2.imwrite(str(pasta / "calibracao_grade.jpg"), img)
    print(f"Confira a grade de 5 m em {pasta/'calibracao_grade.jpg'}")


# ── 2. Movimento da câmera ─────────────────────────────────────────────────

def _mascara(frame: np.ndarray, caixas: np.ndarray, margem_placar: float) -> np.ndarray:
    """Máscara dos pixels úteis para estimar o movimento da câmera.

    Mantém só o GRAMADO (com as linhas): a torcida se mexe e os painéis de LED
    são animados, o que engana o RANSAC e faz a homografia "derivar".
    Remove também os jogadores (se movem por conta própria) e o placar.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    # Fechamento morfológico: inclui as linhas brancas que cortam o gramado
    verde = cv2.morphologyEx(verde, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    verde = cv2.erode(verde, np.ones((15, 15), np.uint8))   # afasta da borda com as placas
    verde[: int(h * margem_placar), :] = 0
    for x1, y1, x2, y2 in caixas.astype(int):
        verde[max(y1 - 5, 0):y2 + 5, max(x1 - 5, 0):x2 + 5] = 0
    return verde


def _mascara_tudo(frame: np.ndarray, caixas: np.ndarray, margem_placar: float) -> np.ndarray:
    """Ajuste do grupo: imagem INTEIRA menos jogadores (e menos a faixa do placar).

    Uma câmera de transmissão gira em torno de um centro óptico fixo (pan/tilt/zoom).
    Nesse caso, a relação entre dois quadros é uma homografia para TODOS os pontos
    da cena — inclusive arquibancada e placas, que não estão no plano do gramado.
    A arquibancada tem muito mais textura que a grama (listras repetitivas), então
    o ORB acha correspondências mais confiáveis; torcedores que se mexem e LEDs
    animados são minoria e viram outliers do RANSAC. Testado: com a máscara só de
    gramado, a projeção derivava dezenas de metros em 60 s neste vídeo.
    """
    h, w = frame.shape[:2]
    m = np.full((h, w), 255, np.uint8)
    m[: int(h * margem_placar), :] = 0
    for x1, y1, x2, y2 in caixas.astype(int):
        m[max(y1 - 8, 0):y2 + 8, max(x1 - 8, 0):x2 + 8] = 0
    return m


def movimento_camera(pasta: Path, limiar_corte: float, margem_placar: float, min_inliers: int = 150,
                     mascara: str = "tudo", n_features: int = 5000, max_quadros: int | None = None):
    """Estima G_t (quadro t → quadro 0) casando cada quadro com um QUADRO-CHAVE.

    Encadear quadro a quadro (G_t = G_{t-1}·M_t) soma o erro de centenas de
    estimativas. Casando com um quadro-chave, o erro só se acumula quando o
    quadro-chave é trocado (quando a câmera andou tanto que sobram poucos inliers).
    """
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    tracks = pd.read_csv(pasta / "tracks.csv")
    caixas_por_frame = {f: g[["x1", "y1", "x2", "y2"]].to_numpy()
                        for f, g in tracks.groupby("frame")}

    cap = cv2.VideoCapture(meta["video"])
    cap.set(cv2.CAP_PROP_POS_MSEC, meta["inicio_s"] * 1000)
    orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=10)
    fazer_mascara = _mascara_tudo if mascara == "tudo" else _mascara
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    n = meta["n_frames"] if max_quadros is None else min(max_quadros, meta["n_frames"])
    G = np.zeros((n, 3, 3))
    cortes, falhas, n_chaves = [], 0, 1
    chave = None          # (keypoints, descritores, G do quadro-chave)
    ant = None            # (keypoints, descritores, G) do quadro anterior
    ant_hist = None

    for f in tqdm(range(n), desc="Câmera"):
        ok, frame = cap.read()
        if not ok:
            G = G[:f]
            break
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = fazer_mascara(frame, caixas_por_frame.get(f, np.empty((0, 4))), margem_placar)
        kp, des = orb.detectAndCompute(cinza, mask)

        # Detecção de corte de câmera: queda brusca na correlação dos histogramas
        hist = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if f == 0:
            G[0] = np.eye(3)
            chave = ant = (kp, des, G[0])
            ant_hist = hist
            continue
        if cv2.compareHist(ant_hist, hist, cv2.HISTCMP_CORREL) < limiar_corte:
            cortes.append(f)
        ant_hist = hist

        M, inliers = _casar(kp, des, chave[0], chave[1], bf)
        if M is not None and inliers >= min_inliers:
            G[f] = chave[2] @ M
        else:
            # Poucos inliers com o quadro-chave: casa com o quadro anterior
            # e transforma o anterior no novo quadro-chave
            M, inliers = _casar(kp, des, ant[0], ant[1], bf)
            if M is None:
                falhas += 1
                M = np.eye(3)
            G[f] = ant[2] @ M
            chave = ant
            n_chaves += 1
        ant = (kp, des, G[f])
    cap.release()

    np.save(pasta / "camera.npy", G)
    print(f"Homografias salvas em {pasta/'camera.npy'} ({len(G)} quadros, "
          f"{n_chaves} quadros-chave, {falhas} falhas)")
    if cortes:
        t = [f"{c / meta['fps']:.1f}s" for c in cortes]
        print(f"ATENÇÃO: {len(cortes)} possível(is) corte(s) de câmera em {t}.")
        print("  Após um corte a projeção deixa de valer: use um trecho sem cortes ou")
        print("  divida o vídeo em segmentos e calibre cada um separadamente.")
    json.dump({"cortes_frames": cortes}, open(pasta / "camera_cortes.json", "w"))
    # Resumo para o README: quanto a câmera andou (centro da imagem levado ao quadro 0)
    c = np.array([[meta["largura"] / 2, meta["altura"] / 2]])
    desl = [float(np.linalg.norm(aplicar_h(Gf, c) - c)) for Gf in G]
    json.dump({"quadros": int(len(G)), "quadros_chave": n_chaves, "falhas": falhas,
               "cortes_frames": cortes, "mascara": mascara, "desloc_max_px": max(desl) if desl else 0.0,
               "desloc_medio_px": float(np.mean(desl)) if desl else 0.0},
              open(pasta / "camera_resumo.json", "w"), indent=2)


def _casar(kp, des, kp_ref, des_ref, bf):
    """Homografia (quadro atual → referência) por ORB + teste da razão de Lowe + RANSAC."""
    if des is None or des_ref is None or len(kp) < 10 or len(kp_ref) < 10:
        return None, 0
    pares = bf.knnMatch(des, des_ref, k=2)
    bons = [m for m, *o in pares if o and m.distance < 0.75 * o[0].distance]
    if len(bons) < 10:
        return None, 0
    p = np.float32([kp[m.queryIdx].pt for m in bons])
    q = np.float32([kp_ref[m.trainIdx].pt for m in bons])
    M, ok = cv2.findHomography(p, q, cv2.RANSAC, 2.0)
    return M, int(ok.sum()) if ok is not None else 0


# ── 2b. Re-ancoragem (ajuste do grupo) ────────────────────────────────────
# Mesmo com a máscara "tudo", encadear homografias por 3 min de pan/zoom deriva
# (testado: bom até ~20 s, dezenas de metros de erro aos 60 s). Solução (Q4.3):
# calibrar À MÃO alguns quadros-âncora (a cada ~20 s) e, para um quadro t entre as
# âncoras a < t < b, projetar cada ponto pelas DUAS âncoras —
#     x_a = H_a · M_{a←t} · p     x_b = H_b · M_{b←t} · p
# — e fazer a média ponderada pela distância temporal (w = (t − a)/(b − a)).
# M_{a←t} é o produto das homografias entre quadros CONSECUTIVOS (mais robustas
# que o casamento com quadro-chave distante). O erro fica limitado ao meio
# intervalo entre âncoras e a projeção é exata nas âncoras.

def pares_consecutivos(pasta: Path, n_features: int = 5000):
    """M_t: homografia quadro t → quadro t−1 (imagem toda menos jogadores)."""
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    tracks = pd.read_csv(pasta / "tracks.csv")
    caixas = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in tracks.groupby("frame")}
    cap = cv2.VideoCapture(meta["video"])
    orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=10)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    n = meta["n_frames"]
    M = np.tile(np.eye(3), (n, 1, 1))
    inl = np.zeros(n, int)
    ant = None
    for f in tqdm(range(n), desc="Pares consecutivos"):
        ok, frame = cap.read()
        if not ok:
            break
        kp, des = orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                       _mascara_tudo(frame, caixas.get(f, np.empty((0, 4))), 0.0))
        if ant is not None:
            Mf, k = _casar(kp, des, ant[0], ant[1], bf)
            if Mf is not None:
                M[f], inl[f] = Mf, k
        ant = (kp, des)
    cap.release()
    np.save(pasta / "camera_pares.npy", M)
    np.save(pasta / "camera_pares_inliers.npy", inl)
    print(f"Pares consecutivos salvos ({n} quadros; inliers medianos {int(np.median(inl[1:]))}, "
          f"mínimo {int(inl[1:].min())})")


def ancorar(pasta: Path):
    """Combina âncoras manuais (calibracao_ancoras.json) com os pares consecutivos."""
    M = np.load(pasta / "camera_pares.npy")
    anc = json.loads((pasta / "calibracao_ancoras.json").read_text(encoding="utf-8"))
    quadros = sorted(int(f) for f in anc)
    Hs = {int(f): np.array(v["H"]) for f, v in anc.items()}
    n = len(M)
    HA, HB, W = np.zeros((n, 3, 3)), np.zeros((n, 3, 3)), np.zeros(n)
    for f in range(n):
        a = max([q for q in quadros if q <= f], default=quadros[0])
        b = min([q for q in quadros if q >= f], default=quadros[-1])
        # M_{a←f} = M_{a+1} · … · M_f  (para f ≥ a);  para f < a usa as inversas
        def cadeia(ref):
            T = np.eye(3)
            if f >= ref:
                for k in range(f, ref, -1):
                    T = M[k] @ T
            else:
                for k in range(f + 1, ref + 1):
                    T = np.linalg.inv(M[k]) @ T
            return T
        HA[f] = Hs[a] @ cadeia(a)
        HB[f] = Hs[b] @ cadeia(b)
        W[f] = 0.0 if b == a else (f - a) / (b - a)
    np.savez(pasta / "camera_ancoras.npz", HA=HA, HB=HB, W=W, quadros=np.array(quadros))
    print(f"Projeção re-ancorada com {len(quadros)} âncoras: {quadros}")


# ── 2c. Refinamento pelas LINHAS do campo (ajuste do grupo) ─────────────────
# Medimos que qualquer encadeamento de homografias (ORB ou SIFT, com máscara de
# gramado ou da imagem toda) acumula 2,5–4 m de erro médio em 20 s neste vídeo.
# Solução: a cada quadro, projetar o MODELO do campo (linhas + círculo + meias-
# luas) com a estimativa atual, casar cada ponto do modelo com o pixel de linha
# branca mais próximo (máscara top-hat no gramado) e reajustar a homografia por
# RANSAC — um ICP de linhas, em 3 iterações com raio de busca decrescente.
# Inicialização: só o quadro 0 (calibracao.json); depois H_t ← H_{t−1} · M_t
# (par consecutivo) e refino. As âncoras manuais dos outros quadros NÃO entram
# no cálculo: servem de validação independente (camera_validacao_ancoras.csv).

def modelo_linhas(passo: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Pontos (m) sobre todas as linhas do campo e a TANGENTE unitária em cada ponto."""
    pts, tans = [], []

    def seg(a, b):
        n = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) / passo), 2)
        pts.append(np.linspace(a, b, n))
        d = np.subtract(b, a, dtype=float)
        tans.append(np.tile(d / np.linalg.norm(d), (n, 1)))

    def arco(cx, cy, r, filtro=None):
        th = np.linspace(0, 2 * np.pi, int(2 * np.pi * r / passo), endpoint=False)
        P = np.c_[cx + r * np.cos(th), cy + r * np.sin(th)]
        T = np.c_[-np.sin(th), np.cos(th)]
        k = filtro(P) if filtro else np.ones(len(P), bool)
        pts.append(P[k])
        tans.append(T[k])

    L, C = COMPRIMENTO, LARGURA
    seg((0, 0), (L, 0)); seg((0, C), (L, C)); seg((0, 0), (0, C)); seg((L, 0), (L, C))
    seg((L / 2, 0), (L / 2, C))
    for x0, sg in ((0, 1), (L, -1)):
        for prof, meia in ((16.5, 20.16), (5.5, 9.16)):
            y0, y1 = C / 2 - meia, C / 2 + meia
            seg((x0, y0), (x0 + sg * prof, y0)); seg((x0, y1), (x0 + sg * prof, y1))
            seg((x0 + sg * prof, y0), (x0 + sg * prof, y1))
        arco(x0 + sg * 11, C / 2, 9.15, lambda P, x0=x0, sg=sg: sg * (P[:, 0] - x0) > 16.5)
    arco(L / 2, C / 2, 9.15)
    return np.vstack(pts), np.vstack(tans)


def mascara_linhas(frame: np.ndarray, caixas: np.ndarray) -> np.ndarray:
    """Pixels de linha branca sobre o gramado (top-hat), sem jogadores."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    verde = cv2.morphologyEx(verde, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    th = cv2.morphologyEx(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.MORPH_TOPHAT, np.ones((15, 15), np.uint8))
    lin = ((th > 25) & (verde > 0) & (hsv[..., 1] < 90)).astype(np.uint8)
    for x1, y1, x2, y2 in caixas.astype(int):
        lin[max(y1 - 4, 0):y2 + 4, max(x1 - 4, 0):x2 + 4] = 0
    return lin


def refinar_quadro(H: np.ndarray, lin: np.ndarray, modelo: tuple[np.ndarray, np.ndarray],
                   raios=(15, 10, 6, 4, 4), min_pontos: int = 60, lam: float = 50.0,
                   salto_max_m: float = 1.5) -> tuple[np.ndarray, int]:
    """Refino ponto-a-RETA (Gauss-Newton regularizado) de uma homografia pixel→metros.

    Para cada ponto do modelo projetado na imagem, o resíduo é a distância
    PERPENDICULAR à linha (normal = perpendicular à tangente projetada) até o
    pixel de linha branca mais próximo. Resolvemos uma correção Hc ≈ I + P na
    imagem por mínimos quadrados com penalidade λ‖p‖² — direções que as linhas
    visíveis não determinam (ex.: só linhas paralelas) ficam como a predição.
    Retorna (H, nº de pontos usados; 0 = não refinou).
    """
    pts_m, tan_m = modelo
    h, w = lin.shape
    if lin.sum() < 200:
        return H, 0
    dist, lab = cv2.distanceTransformWithLabels(1 - lin, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(lin)
    coord = np.zeros((lab.max() + 1, 2), np.float64)
    coord[lab[ys, xs]] = np.c_[xs, ys]
    Hinv = np.linalg.inv(H)
    img = aplicar_h(Hinv, pts_m)
    img2 = aplicar_h(Hinv, pts_m + 0.25 * tan_m)
    ok = (img[:, 0] >= 1) & (img[:, 0] < w - 2) & (img[:, 1] >= 1) & (img[:, 1] < h - 2) & \
         np.isfinite(img).all(1) & np.isfinite(img2).all(1)
    img, t_img = img[ok], img2[ok] - img[ok]
    nrm = np.c_[-t_img[:, 1], t_img[:, 0]]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    E = 1000.0                                   # escala (coordenadas normalizadas)
    Hc = np.eye(3)
    usados = 0
    for r in raios:
        x = aplicar_h(Hc, img)
        xi, yi = np.round(x[:, 0]).astype(int), np.round(x[:, 1]).astype(int)
        dentro = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xi, yi = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
        perto = dentro & (dist[yi, xi] < r)
        if perto.sum() < min_pontos:
            return H, 0
        alvo = coord[lab[yi[perto], xi[perto]]]
        xp, n_ = x[perto] / E, nrm[perto]
        res = np.sum(n_ * (xp - alvo / E), axis=1)              # distância perpendicular (normalizada)
        X, Y = xp[:, 0], xp[:, 1]
        z, o = np.zeros_like(X), np.ones_like(X)
        Jx = np.c_[X, Y, o, z, z, z, -X * X, -X * Y]
        Jy = np.c_[z, z, z, X, Y, o, -X * Y, -Y * Y]
        A = n_[:, :1] * Jx + n_[:, 1:] * Jy
        p_ = np.linalg.solve(A.T @ A + lam * np.eye(8) * (len(A) / 1000.0), -A.T @ res)
        P = np.array([[1 + p_[0], p_[1], p_[2] * E], [p_[3], 1 + p_[4], p_[5] * E],
                      [p_[6] / E, p_[7] / E, 1.0]])
        Hc = P @ Hc
        usados = int(perto.sum())
    Hn = H @ np.linalg.inv(Hc)
    c = np.array([[w * a, h * b] for a in (0.25, 0.5, 0.75) for b in (0.4, 0.6, 0.8)])
    if np.linalg.norm(aplicar_h(Hn, c) - aplicar_h(H, c), axis=1).mean() > salto_max_m:
        return H, 0
    return Hn, usados


def refinar(pasta: Path, usar_ancoras: bool = False, max_quadros: int | None = None):
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    tracks = pd.read_csv(pasta / "tracks.csv")
    caixas = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in tracks.groupby("frame")}
    M = np.load(pasta / "camera_pares.npy")
    anc = (json.loads((pasta / "calibracao_ancoras.json").read_text(encoding="utf-8"))
           if (pasta / "calibracao_ancoras.json").exists() else {})
    Hanc = {int(f): np.array(v["H"]) for f, v in anc.items()}
    H0 = np.array(json.loads((pasta / "calibracao.json").read_text(encoding="utf-8"))["H0"])
    modelo = modelo_linhas()
    n = meta["n_frames"] if max_quadros is None else min(max_quadros, meta["n_frames"])
    Ht = np.zeros((n, 3, 3))
    inl = np.zeros(n, int)
    cap = cv2.VideoCapture(meta["video"])
    for f in tqdm(range(n), desc="Refino por linhas"):
        ok, frame = cap.read()
        if not ok:
            break
        if f == 0:
            H = H0
        elif usar_ancoras and f in Hanc:
            H = Hanc[f]
        else:
            H = Ht[f - 1] @ M[f]
        Ht[f], inl[f] = refinar_quadro(H, mascara_linhas(frame, caixas.get(f, np.empty((0, 4)))), modelo)
    cap.release()
    if max_quadros is None:
        np.save(pasta / "camera_refinada.npy", Ht)
        np.save(pasta / "camera_refino_inliers.npy", inl)

    # Validação independente nas âncoras manuais (grade visível de 5 × 4 m)
    grade = np.array([[x, y] for x in range(0, 106, 5) for y in range(0, 69, 4)], float)
    linhas = []
    for f, Ha in sorted(Hanc.items()):
        if f >= n:
            continue
        pix = aplicar_h(np.linalg.inv(Ha), grade)
        vis = (pix[:, 0] > 0) & (pix[:, 0] < meta["largura"]) & (pix[:, 1] > 0) & (pix[:, 1] < meta["altura"])
        e = np.linalg.norm(aplicar_h(Ht[f], pix[vis]) - grade[vis], axis=1)
        linhas.append({"quadro": f, "t_s": round(f / meta["fps"], 1), "pontos_grade": int(vis.sum()),
                       "erro_medio_m": round(float(e.mean()), 2), "erro_mediano_m": round(float(np.median(e)), 2),
                       "erro_max_m": round(float(e.max()), 2), "usada_no_calculo": bool(f == 0 or usar_ancoras)})
    val = pd.DataFrame(linhas)
    ref = 100 * float((inl[1:] > 0).mean())
    if max_quadros is None:
        val.to_csv(pasta / "camera_validacao_ancoras.csv", index=False)
        json.dump({"metodo": "pares consecutivos ORB + ICP de linhas do campo",
                   "quadros_refinados_pct": round(ref, 1),
                   "inliers_medianos": int(np.median(inl[inl > 0])) if (inl > 0).any() else 0,
                   "usar_ancoras": usar_ancoras},
                  open(pasta / "camera_refino_resumo.json", "w"), indent=2)
    print(f"Refinados {ref:.1f}% dos quadros (inliers medianos "
          f"{int(np.median(inl[inl > 0])) if (inl > 0).any() else 0}).")
    print("Validação nas âncoras manuais (erro da grade visível):")
    print(val.to_string(index=False))


class Projetor:
    """Pixel (u, v) do quadro f → metros. Usa as âncoras se existirem; senão H0 · G_t."""

    def __init__(self, pasta: Path, sem_camera: bool = False):
        ref = pasta / "camera_refinada.npy"
        self.Hr = np.load(ref) if ref.exists() and not sem_camera else None
        arq = pasta / "camera_ancoras.npz"
        self.anc = np.load(arq) if arq.exists() and not sem_camera else None
        self.H0 = np.array(json.loads((pasta / "calibracao.json").read_text(encoding="utf-8"))["H0"])
        self.G = None if sem_camera or self.anc is not None else np.load(pasta / "camera.npy")
        self.modo = "ancoras" if self.anc is not None else ("H0" if sem_camera else "H0·G_t")
        if self.Hr is not None:
            self.modo = "refinada por linhas"

    def n(self) -> int:
        if self.Hr is not None:
            return len(self.Hr)
        return len(self.anc["W"]) if self.anc is not None else (len(self.G) if self.G is not None else 10**9)

    def __call__(self, f: int, pts: np.ndarray) -> np.ndarray:
        if self.Hr is not None:
            return aplicar_h(self.Hr[f], pts)
        if self.anc is not None:
            w = float(self.anc["W"][f])
            return (1 - w) * aplicar_h(self.anc["HA"][f], pts) + w * aplicar_h(self.anc["HB"][f], pts)
        return aplicar_h(self.H0 if self.G is None else self.H0 @ self.G[f], pts)


# ── 3. Projeção ────────────────────────────────────────────────────────────

def projetar(pasta: Path, sem_camera: bool):
    proj = Projetor(pasta, sem_camera)
    print(f"Projeção: modo {proj.modo}")

    def para_campo(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        xy = np.full((len(df), 2), np.nan)
        for f, idx in df.groupby("frame").indices.items():
            if f >= proj.n():
                continue
            xy[idx] = proj(int(f), df.iloc[idx][["px", "py"]].to_numpy())
        df["x_m"], df["y_m"] = xy[:, 0], xy[:, 1]
        # Descarta pontos muito fora do campo (erros de projeção/detecções na arquibancada)
        fora = (df["x_m"] < -5) | (df["x_m"] > COMPRIMENTO + 5) | (df["y_m"] < -5) | (df["y_m"] > LARGURA + 5)
        df.loc[fora, ["x_m", "y_m"]] = np.nan
        return df

    tracks = pd.read_csv(pasta / "tracks.csv")
    jog = para_campo(tracks[tracks["classe"] != "bola"])
    jog.to_csv(pasta / "tracks_campo.csv", index=False)
    print(f"Jogadores projetados: {jog['x_m'].notna().mean()*100:.1f}% válidos → tracks_campo.csv")

    if (pasta / "bola.csv").exists():
        bola = para_campo(pd.read_csv(pasta / "bola.csv"))
        bola.to_csv(pasta / "bola_campo.csv", index=False)
        print(f"Bola projetada: {bola['x_m'].notna().mean()*100:.1f}% válidos → bola_campo.csv")


def main():
    p = argparse.ArgumentParser(description="Homografia pixel → campo (metros)")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrar")
    c.add_argument("--saida", required=True)
    c.add_argument("--pontos", default="", help="Marcos separados por vírgula (ver campo.MARCOS)")
    c.add_argument("--sem-janela", action="store_true", help="Lê os pixels de calibracao.json")
    m = sub.add_parser("camera")
    m.add_argument("--saida", required=True)
    m.add_argument("--limiar-corte", type=float, default=0.5, help="Correlação mínima entre quadros")
    m.add_argument("--margem-placar", type=float, default=0.0,
                   help="Fração superior ignorada (este vídeo não tem placar na tela)")
    m.add_argument("--mascara", choices=["tudo", "gramado"], default="tudo",
                   help="tudo = imagem menos jogadores (padrão do grupo); gramado = código de apoio")
    m.add_argument("--n-features", type=int, default=5000)
    m.add_argument("--max-quadros", type=int, default=None, help="Só para testes rápidos")
    sub.add_parser("pares").add_argument("--saida", required=True)
    sub.add_parser("ancorar").add_argument("--saida", required=True)
    r_ = sub.add_parser("refinar")
    r_.add_argument("--saida", required=True)
    r_.add_argument("--usar-ancoras", action="store_true",
                    help="Reinicia nas âncoras manuais (padrão: âncoras só para validação)")
    r_.add_argument("--max-quadros", type=int, default=None, help="Só para testes (não salva)")
    j = sub.add_parser("projetar")
    j.add_argument("--saida", required=True)
    j.add_argument("--sem-camera", action="store_true", help="Assume câmera fixa (usa só H0)")
    a = p.parse_args()

    pasta = Path(a.saida)
    if a.cmd == "calibrar":
        calibrar(pasta, [s for s in a.pontos.split(",") if s], a.sem_janela)
    elif a.cmd == "camera":
        movimento_camera(pasta, a.limiar_corte, a.margem_placar, mascara=a.mascara,
                         n_features=a.n_features, max_quadros=a.max_quadros)
    elif a.cmd == "pares":
        pares_consecutivos(pasta)
    elif a.cmd == "ancorar":
        ancorar(pasta)
    elif a.cmd == "refinar":
        refinar(pasta, a.usar_ancoras, a.max_quadros)
    else:
        projetar(pasta, a.sem_camera)


if __name__ == "__main__":
    main()
