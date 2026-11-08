"""
Classificador de times por cor da camisa (K-Means no espaço HSV).

Versão comentada em PT-BR de implementacoes/football_track/team_classifier.py,
com duas extensões:
  1. k configurável: com k=3, o menor cluster vira "outros" (árbitro/goleiros);
  2. votação: cada track_id recebe o time mais votado nas primeiras observações,
     em vez de ficar preso à primeira predição (que pode vir de um recorte ruim).

Estratégia
----------
1. Calibração: recortes de jogadores nos primeiros N quadros.
2. Para cada recorte, cor média da região da camisa (ignorando pixels de grama)
   num espaço HSV "circular": o matiz H é um ângulo, e o vermelho fica nas duas
   pontas (H≈0 e H≈179). A média simples de H daria ≈90 (ciano)! Por isso usamos
   (S·cos H, S·sin H, V), que trata o matiz como ângulo.
3. K-Means sobre esses vetores de cor → centróides dos times.
4. Inferência: cada jogador rastreado vai para o centróide mais próximo.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Faixa vertical do bounding box usada como "camisa" (evita cabeça, calção e meião)
_CAMISA_TOPO = 0.10
_CAMISA_BASE = 0.55

# Faixa HSV do verde da grama (OpenCV: H ∈ [0, 179])
_GRAMA_H = (35, 85)
_GRAMA_S_MIN = 40
_GRAMA_V_MIN = 40

OUTROS = -2        # árbitro, goleiros ou cluster minoritário
DESCONHECIDO = -1  # ainda não calibrado / recorte inválido


def cor_camisa(recorte_bgr: np.ndarray) -> np.ndarray | None:
    """Cor média (S·cos H, S·sin H, V) da região da camisa, sem os pixels de grama."""
    h, w = recorte_bgr.shape[:2]
    if h < 20 or w < 10:
        return None

    camisa = recorte_bgr[int(h * _CAMISA_TOPO): int(h * _CAMISA_BASE), :]
    hsv = cv2.cvtColor(camisa, cv2.COLOR_BGR2HSV)

    grama = (
        (hsv[:, :, 0] >= _GRAMA_H[0]) & (hsv[:, :, 0] <= _GRAMA_H[1]) &
        (hsv[:, :, 1] >= _GRAMA_S_MIN) & (hsv[:, :, 2] >= _GRAMA_V_MIN)
    )
    manter = ~grama
    if manter.sum() < 30:
        return None
    px = hsv[manter].astype(np.float32)
    ang = px[:, 0] * (2 * np.pi / 180.0)          # H do OpenCV ∈ [0, 180) → radianos
    return np.array([(px[:, 1] * np.cos(ang)).mean(),
                     (px[:, 1] * np.sin(ang)).mean(),
                     px[:, 2].mean()], dtype=np.float32)


def para_bgr(cor: np.ndarray) -> tuple[int, int, int]:
    """Converte (S·cos H, S·sin H, V) de volta para BGR (para desenhar/conferir)."""
    h = (np.arctan2(cor[1], cor[0]) % (2 * np.pi)) * 180.0 / (2 * np.pi)
    s = np.clip(np.hypot(cor[0], cor[1]), 0, 255)
    hsv = np.uint8([[[h, s, np.clip(cor[2], 0, 255)]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


class ClassificadorTimes:
    """K-Means de cor de camisa, calibrado nos quadros iniciais do clipe."""

    def __init__(self, k: int = 2, votos_necessarios: int = 10):
        self.k = k
        self.votos_necessarios = votos_necessarios
        self.calibrado = False
        self._kmeans: KMeans | None = None
        self._cluster_para_time: dict[int, int] = {}
        self._amostras: list[np.ndarray] = []
        self._votos: dict[int, Counter] = defaultdict(Counter)
        self._time_fixo: dict[int, int] = {}

    # ── Calibração ─────────────────────────────────────────────────────
    def adicionar_amostra(self, recorte_bgr: np.ndarray):
        cor = cor_camisa(recorte_bgr)
        if cor is not None:
            self._amostras.append(cor)

    def ajustar(self) -> bool:
        """Ajusta o K-Means; retorna False se ainda há poucas amostras."""
        if len(self._amostras) < 20 * self.k:
            return False
        X = np.stack(self._amostras)
        self._kmeans = KMeans(n_clusters=self.k, n_init=10, random_state=42).fit(X)

        # Os 2 maiores clusters são os times 0 e 1; os demais viram OUTROS
        tamanhos = np.bincount(self._kmeans.labels_, minlength=self.k)
        ordem = np.argsort(-tamanhos)
        self._cluster_para_time = {int(c): (i if i < 2 else OUTROS) for i, c in enumerate(ordem)}

        self.calibrado = True
        self._amostras.clear()
        return True

    def centroides_bgr(self) -> dict[int, tuple[int, int, int]]:
        """Cor média de cada time em BGR (útil para desenhar e conferir)."""
        return {time: para_bgr(self._kmeans.cluster_centers_[c])
                for c, time in self._cluster_para_time.items()}

    # ── Inferência ─────────────────────────────────────────────────────
    def prever(self, track_id: int, recorte_bgr: np.ndarray) -> int:
        """Time (0, 1), OUTROS ou DESCONHECIDO para este jogador."""
        if not self.calibrado:
            return DESCONHECIDO
        if track_id in self._time_fixo:
            return self._time_fixo[track_id]

        cor = cor_camisa(recorte_bgr)
        if cor is None:
            return self._mais_votado(track_id)

        cluster = int(self._kmeans.predict(cor.reshape(1, -1))[0])
        self._votos[track_id][self._cluster_para_time[cluster]] += 1

        # Depois de votos suficientes, a decisão fica fixa (estabilidade temporal)
        if sum(self._votos[track_id].values()) >= self.votos_necessarios:
            self._time_fixo[track_id] = self._mais_votado(track_id)
        return self._mais_votado(track_id)

    def _mais_votado(self, track_id: int) -> int:
        votos = self._votos.get(track_id)
        return votos.most_common(1)[0][0] if votos else DESCONHECIDO

    def esquecer(self, track_id: int):
        """Chamar quando um track some, para reclassificá-lo se voltar."""
        self._votos.pop(track_id, None)
        self._time_fixo.pop(track_id, None)


# ══════════════════════════════════════════════════════════════════════════
# Pós-processamento por TRACK (ajuste do grupo)
# ══════════════════════════════════════════════════════════════════════════
# O classificador online acima decide o time de um ID com ~10 recortes, logo
# que ele aparece. Na câmera tática os jogadores do outro lado do campo têm
# ~40 px de altura: alguns recortes ruins bastam para errar o time (observado
# no rastreado.mp4: jogadores do Bayern marcados como PSG; árbitro no time 0).
# Aqui, depois do rastreamento, relemos o vídeo e calculamos um descritor de
# cor para TODOS os recortes de cada track; o time sai da MÉDIA do track
# (centenas de recortes), com K-Means k=2 entre tracks e detecção de outliers
# (árbitro, goleiros) pela distância ao centróide.

# Faixas de matiz (OpenCV, H ∈ [0, 180)) para pixels saturados
_FAIXAS_H = {"vermelho": [(0, 8), (165, 180)], "laranja_amarelo": [(8, 35)], "verde_ciano": [(35, 95)],
             "azul": [(95, 130)], "roxo_rosa": [(130, 165)]}
NOMES_DESCRITOR = list(_FAIXAS_H) + ["branco", "escuro"]


def descritor_cor(recorte_bgr: np.ndarray) -> np.ndarray | None:
    """Frações de pixels da camisa (sem grama) em 5 faixas de matiz + branco + escuro."""
    h, w = recorte_bgr.shape[:2]
    if h < 16 or w < 6:
        return None
    camisa = recorte_bgr[int(h * 0.15): int(h * 0.50), int(w * 0.15): int(w * 0.85)]
    hsv = cv2.cvtColor(camisa, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    grama = (H >= _GRAMA_H[0]) & (H <= _GRAMA_H[1]) & (S >= _GRAMA_S_MIN) & (V >= _GRAMA_V_MIN)
    H, S, V = H[~grama], S[~grama], V[~grama]
    if len(H) < 15:
        return None
    escuro = V < 60
    branco = (S < 50) & (V >= 150)
    sat = ~escuro & ~branco & (S >= 60)
    d = [sum(((H >= a) & (H < b) & sat).sum() for a, b in faixas) for faixas in _FAIXAS_H.values()]
    d += [branco.sum(), escuro.sum()]
    return np.array(d, np.float32) / len(H)


def _cli():
    import argparse
    import json
    from pathlib import Path

    import pandas as pd
    from tqdm import tqdm

    ap = argparse.ArgumentParser(description="Reclassifica os times por track (pós-processamento)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--passo", type=int, default=2, help="Usa 1 a cada N quadros")
    ap.add_argument("--min-amostras", type=int, default=8, help="Recortes mínimos para decidir o time")
    ap.add_argument("--k", type=int, default=2,
                    help="Clusters entre tracks (2 = os dois times; extras viram OUTROS)")
    ap.add_argument("--arbitro-ciano", type=float, default=0.08,
                    help="Regra do árbitro: fração de verde/ciano ≥ este valor e pouco escuro → OUTROS")
    ap.add_argument("--arbitro-escuro", type=float, default=0.15)
    ap.add_argument("--outlier", type=float, default=2.5,
                    help="Track vira OUTROS se a distância ao centróide > outlier × mediana")
    a = ap.parse_args()

    pasta = Path(a.saida)
    tr = pd.read_csv(pasta / "tracks.csv")
    pes = tr[tr["classe"] != "bola"]
    por_frame = {f: g for f, g in pes.groupby("frame")}
    soma: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(len(NOMES_DESCRITOR), np.float64))
    bgr: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3, np.float64))
    n: Counter = Counter()

    cap = cv2.VideoCapture(a.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f in tqdm(range(total), desc="Cores por track"):
        ok, frame = cap.read()
        if not ok:
            break
        if f % a.passo or f not in por_frame:
            continue
        for r in por_frame[f].itertuples():
            rec = frame[max(r.y1, 0):r.y2, max(r.x1, 0):r.x2]
            d = descritor_cor(rec)
            if d is None:
                continue
            soma[r.track_id] += d
            h = rec.shape[0]
            bgr[r.track_id] += rec[int(h * 0.15):int(h * 0.5)].reshape(-1, 3).mean(0)
            n[r.track_id] += 1
    cap.release()

    ids = [t for t in n if n[t] >= a.min_amostras]
    X = np.stack([soma[t] / n[t] for t in ids])
    peso = np.array([n[t] for t in ids], float)
    km = KMeans(n_clusters=a.k, n_init=20, random_state=42).fit(X, sample_weight=peso)
    lab = km.labels_
    dist = np.linalg.norm(X - km.cluster_centers_[lab], axis=1)
    lim = a.outlier * np.median(dist)
    # Convenção estável: time 1 = cluster mais "quente" (vermelho+laranja);
    # time 0 = cluster de maior massa entre os demais; o resto vira OUTROS (árbitro).
    c = km.cluster_centers_
    quente = int(np.argmax(c[:, 0] + c[:, 1]))
    resto = [i for i in range(a.k) if i != quente]
    massa = {i: peso[lab == i].sum() for i in resto}
    frio = max(resto, key=massa.get)
    mapa = {i: OUTROS for i in range(a.k)}
    mapa[quente], mapa[frio] = 1, 0
    time_track = {t: (mapa[int(l)] if d_ <= lim else OUTROS) for t, l, d_ in zip(ids, lab, dist)}
    # Regra explícita do árbitro (testada: com k=3 o 3º cluster virou o técnico de preto; com
    # k=4, jogadores do PSG bem iluminados). O uniforme da arbitragem (azul-claro/turquesa) é o
    # único com fração de verde-ciano relevante (≥ 0,09) e quase sem pixels escuros; os tracks
    # do PSG ficam com ciano ≤ 0,01 e escuro ≥ 0,14 (ver times_por_track.csv).
    i_c, i_e = NOMES_DESCRITOR.index("verde_ciano"), NOMES_DESCRITOR.index("escuro")
    arbitros = [t for t, x in zip(ids, X) if x[i_c] >= a.arbitro_ciano and x[i_e] < a.arbitro_escuro]
    for t in arbitros:
        time_track[t] = OUTROS

    df = pd.DataFrame(X, columns=NOMES_DESCRITOR)
    df.insert(0, "track_id", ids)
    df.insert(1, "amostras", [n[t] for t in ids])
    df["dist_centroide"] = dist
    df["time"] = [time_track[t] for t in ids]
    df.to_csv(pasta / "times_por_track.csv", index=False)

    if "time_online" not in tr:
        tr["time_online"] = tr["time"]
    tr["time"] = [("" if c == "bola" else time_track.get(t, DESCONHECIDO))
                  for c, t in zip(tr["classe"], tr["track_id"])]
    tr.to_csv(pasta / "tracks.csv", index=False)

    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    meta["times_bgr_online"] = meta.get("times_bgr_online", meta.get("times_bgr", {}))
    cores = {}
    for tm in (0, 1, OUTROS):
        sel = [t for t in ids if time_track[t] == tm]
        if sel:
            cores[str(tm)] = [int(v) for v in (sum(bgr[t] for t in sel) / sum(n[t] for t in sel))]
    meta["times_bgr"] = cores
    meta["classificacao_times"] = {
        "metodo": "pós-processamento por track: descritor de cor (frações de matiz/branco/escuro) "
                  "médio do track + K-Means k=2 + outliers", "passo": a.passo,
        "min_amostras": a.min_amostras, "outlier": a.outlier,
        "regra_arbitro": {"verde_ciano_min": a.arbitro_ciano, "escuro_max": a.arbitro_escuro,
                          "tracks": [int(t) for t in arbitros]},
        "k": a.k,
        "centroides": {str(mapa[c]): dict(zip(NOMES_DESCRITOR, np.round(km.cluster_centers_[c], 3).tolist()))
                       for c in range(a.k)}}
    (pasta / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    cont = Counter(time_track.values())
    print(f"Tracks classificados: {len(ids)} (time 0: {cont[0]}, time 1: {cont[1]}, outros: {cont[OUTROS]}); "
          f"{len(n) - len(ids)} com poucas amostras ficaram como desconhecidos.")
    for c in range(a.k):
        print(f"  centróide time {mapa[c]} (massa {peso[lab == c].sum():.0f}):",
              {k_: float(v) for k_, v in zip(NOMES_DESCRITOR, np.round(km.cluster_centers_[c], 2))})
    print(f"  cores médias (BGR): {cores}")


if __name__ == "__main__":
    _cli()
