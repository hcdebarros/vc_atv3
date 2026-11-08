#!/usr/bin/env python3
"""
Etapa 4 — sensibilidade da posse e do nº de passes ao raio de posse (1, 2, 3 m).

Reusa exatamente as funções de metricas.py (mesmo protocolo), variando só o raio.

Uso
---
python sensibilidade_posse.py --saida saidas/psg_bayern --raios 1 2 3

Saídas: sensibilidade_raio.csv e figuras/sensibilidade_raio.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from metricas import carregar_bola, carregar_jogadores, detectar_passes, posse_por_quadro
from validar_passes import casar


def main():
    ap = argparse.ArgumentParser(description="Sensibilidade ao raio de posse")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--raios", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    ap.add_argument("--min-quadros", type=int, default=3)
    ap.add_argument("--dist-min-passe", type=float, default=3.0)
    ap.add_argument("--dur-max-passe", type=float, default=4.0)
    ap.add_argument("--tolerancia", type=float, default=1.0, help="Validação: tolerância (s)")
    a = ap.parse_args()

    pasta = Path(a.saida)
    fps = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))["fps"]
    jog, bola = carregar_jogadores(pasta), carregar_bola(pasta)
    arq_gt = pasta / "passes_anotados.csv"
    gt = pd.read_csv(arq_gt) if arq_gt.exists() else None

    linhas = []
    for r in a.raios:
        p = posse_por_quadro(jog, bola, r, a.min_quadros)
        passes = detectar_passes(p, jog, fps, a.dist_min_passe, a.dur_max_passe)
        posse = p["time_posse"].value_counts(normalize=True)
        linhas.append({
            "raio_m": r,
            "quadros_com_dono_pct": round(100 * float(p["dono"].notna().mean()), 1),
            "posse_time0_pct": round(100 * float(posse.get(0.0, 0.0)), 1),
            "posse_time1_pct": round(100 * float(posse.get(1.0, 0.0)), 1),
            "passes": int(len(passes)),
            "passes_certos": int(passes["sucesso"].sum()) if len(passes) else 0,
            "precisao_passe_pct": round(100 * float(passes["sucesso"].mean()), 1) if len(passes) else None,
        })
        if gt is not None and len(passes):
            vp = len(casar(passes, gt, a.tolerancia))
            pr, rv = vp / len(passes), vp / len(gt)
            linhas[-1].update({"validacao_precisao": round(pr, 2), "validacao_revocacao": round(rv, 2),
                               "validacao_f1": round(2 * pr * rv / (pr + rv), 2) if vp else 0.0})
    df = pd.DataFrame(linhas)
    df.to_csv(pasta / "sensibilidade_raio.csv", index=False)
    print(df.to_string(index=False))

    (pasta / "figuras").mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(df["raio_m"], df["posse_time0_pct"], "o-", color="#1d4ed8", label="time 0")
    ax[0].plot(df["raio_m"], df["posse_time1_pct"], "o-", color="#d62828", label="time 1")
    ax[0].set(xlabel="raio de posse (m)", ylabel="posse (%)", title="Posse × raio")
    ax[0].legend()
    ax[1].bar(df["raio_m"].astype(str), df["passes"], color="#64748b", label="detectados")
    ax[1].bar(df["raio_m"].astype(str), df["passes_certos"], color="#22c55e", label="certos")
    ax[1].set(xlabel="raio de posse (m)", ylabel="nº de passes", title="Passes × raio")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(pasta / "figuras" / "sensibilidade_raio.png", dpi=130)
    print(f"→ {pasta / 'sensibilidade_raio.csv'} e figuras/sensibilidade_raio.png")


if __name__ == "__main__":
    main()
