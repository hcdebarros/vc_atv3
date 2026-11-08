#!/usr/bin/env python3
"""
Validação dos passes detectados contra uma anotação manual (ground truth).

Assista ao trecho e anote os passes em um CSV (passes_anotados.csv):

    t_s,time,sucesso
    3.4,0,1
    7.9,0,0
    ...

t_s = instante do passe em segundos, contado a partir do INÍCIO DO CLIPE
processado (o mesmo relógio do vídeo rastreado.mp4); time = 0/1 conforme as
cores do rastreado.mp4 (T0 = vermelho, T1 = azul); sucesso = 1 se completado.

Um passe detectado "casa" com um anotado se estiver a até --tolerancia segundos
e for do mesmo time. Reporta precisão, revocação e F1 da detecção de passes.

Uso
---
python validar_passes.py --saida ../saidas/jogo1 --anotado passes_anotados.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def casar(det: pd.DataFrame, gt: pd.DataFrame, tol: float):
    """Casamento guloso por proximidade temporal (cada anotação casa no máx. uma vez)."""
    usados, pares = set(), []
    for i, d in det.sort_values("t_ini_s").iterrows():
        cand = gt[(gt["time"] == d["time"]) & ((gt["t_s"] - d["t_ini_s"]).abs() <= tol)]
        cand = cand[~cand.index.isin(usados)]
        if len(cand):
            j = (cand["t_s"] - d["t_ini_s"]).abs().idxmin()
            usados.add(j)
            pares.append((i, j))
    return pares


def main():
    ap = argparse.ArgumentParser(description="Precisão/revocação da detecção de passes")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--anotado", default=None, help="Padrão: <saida>/passes_anotados.csv")
    ap.add_argument("--tolerancia", type=float, default=1.0, help="Segundos")
    a = ap.parse_args()

    pasta = Path(a.saida)
    det = pd.read_csv(pasta / "passes.csv")
    gt = pd.read_csv(a.anotado or pasta / "passes_anotados.csv")
    if len(gt) == 0:
        raise SystemExit("passes_anotados.csv está vazio — anote os passes com ferramentas/anotar_passes.py.")
    pares = casar(det, gt, a.tolerancia)

    vp = len(pares)
    precisao = vp / len(det) if len(det) else 0.0
    revocacao = vp / len(gt) if len(gt) else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if vp else 0.0
    acordo = sum(det.loc[i, "sucesso"] == gt.loc[j, "sucesso"] for i, j in pares) / vp if vp else 0.0

    print(f"Detectados: {len(det)} | Anotados: {len(gt)} | Casados: {vp}")
    print(f"Precisão  : {precisao:.2f}")
    print(f"Revocação : {revocacao:.2f}")
    print(f"F1        : {f1:.2f}")
    print(f"Acordo certo/errado nos casados: {acordo:.2f}")

    # Erros: falsos positivos (detectados sem par) e falsos negativos (anotados sem par)
    fp = det.drop(index=[i for i, _ in pares])
    fn = gt.drop(index=[j for _, j in pares])
    res = {"tolerancia_s": a.tolerancia, "detectados": int(len(det)), "anotados": int(len(gt)),
           "casados": vp, "falsos_positivos": int(len(fp)), "falsos_negativos": int(len(fn)),
           "precisao": round(precisao, 3), "revocacao": round(revocacao, 3), "f1": round(f1, 3),
           "acordo_sucesso": round(float(acordo), 3),
           "anotados_certos_pct": round(100 * float(gt["sucesso"].mean()), 1),
           "detectados_certos_pct": round(100 * float(det["sucesso"].mean()), 1) if len(det) else None}
    (pasta / "validacao_passes.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    fp.to_csv(pasta / "validacao_falsos_positivos.csv", index=False)
    fn.to_csv(pasta / "validacao_falsos_negativos.csv", index=False)
    print(f"Resultados → {pasta / 'validacao_passes.json'} (+ listas de FP/FN em CSV)")


if __name__ == "__main__":
    main()
