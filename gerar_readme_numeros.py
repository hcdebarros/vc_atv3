#!/usr/bin/env python3
"""
Preenche (e verifica) os números do README a partir dos arquivos em saidas/<jogo>/.

Regra do enunciado: README ≠ arquivos gerados = −20%. Por isso NENHUM número do
README é digitado à mão: cada tabela fica entre marcadores

    <!-- AUTO:nome -->
    ...conteúdo gerado...
    <!-- /AUTO:nome -->

e este script reescreve o conteúdo a partir dos CSV/JSON. Também copia as
figuras de saidas/<jogo>/figuras/ para figuras/ (pasta exigida na entrega).

Uso
---
python gerar_readme_numeros.py --saida saidas/psg_bayern            # preenche README.md
python gerar_readme_numeros.py --saida saidas/psg_bayern --verificar  # só confere (exit 1 se divergir)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PADRAO = re.compile(r"(<!-- AUTO:(\w+) -->\n)(.*?)(<!-- /AUTO:\2 -->)", re.S)


def _json(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _csv(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None


def _pendente(arq: str) -> str:
    return f"_(pendente: `{arq}` ainda não foi gerado)_\n"


def _tabela(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    """DataFrame → tabela Markdown (sem dependência do pacote tabulate)."""
    fmt = fmt or {}
    cab = "| " + " | ".join(map(str, df.columns)) + " |\n"
    sep = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in df.columns) + "|\n"
    linhas = ""
    for _, r in df.iterrows():
        cel = []
        for c in df.columns:
            v = r[c]
            if pd.isna(v):
                cel.append("—")
            elif c in fmt:
                cel.append(format(v, fmt[c]))
            elif isinstance(v, float):
                cel.append(f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:.0f}")
            else:
                cel.append(str(v))
        linhas += "| " + " | ".join(cel) + " |\n"
    return cab + sep + linhas


# ── Blocos ─────────────────────────────────────────────────────────────────

def b_video(p: Path) -> str:
    v = _json(p / "video_info.json")
    if v is None:
        return _pendente("video_info.json")
    cortes = v.get("cortes")
    txt = "nenhum" if cortes == [] else (f"{len(cortes)} (ver video_info.json)" if cortes else "não verificado")
    return (f"| Resolução | FPS | Quadros | Duração | Cortes de câmera detectados |\n|---|---:|---:|---:|---|\n"
            f"| {v['largura']}×{v['altura']} | {v['fps']:.0f} | {v['n_frames']} | "
            f"{v['duracao_s']:.1f} s | {txt} |\n")


def b_yolo(p: Path) -> str:
    d = _csv(p / "comparar_yolo.csv")
    if d is None:
        return _pendente("comparar_yolo.csv")
    cols = ["modelo", "imgsz", "fps", "pessoas_por_quadro", "ids_unicos_pessoa",
            "pct_quadros_bola_conf0.05", "pct_quadros_bola_conf0.10", "pct_quadros_bola_conf0.25"]
    extra = ["pct_bola_via_track"] if "pct_bola_via_track" in d else []
    d = d[cols + extra].rename(columns={
        "pessoas_por_quadro": "pessoas/quadro", "ids_unicos_pessoa": "IDs únicos",
        "pct_quadros_bola_conf0.05": "% bola (conf≥0,05)", "pct_quadros_bola_conf0.10": "% bola (conf≥0,10)",
        "pct_quadros_bola_conf0.25": "% bola (conf≥0,25)", "pct_bola_via_track": "% bola só via track"})
    sub = _csv(p / "comparar_yolo.csv")["subtrecho"].iloc[0]
    return f"Subtrecho: **{sub}** do clipe (750 quadros).\n\n" + _tabela(d)


def b_deteccao(p: Path) -> str:
    m = _json(p / "meta.json")
    t = _csv(p / "tracks.csv")
    if m is None or t is None:
        return _pendente("meta.json / tracks.csv")
    j = t[t["classe"] != "bola"]
    por_q = j.groupby("frame").size()
    moda = j[j["time"] != -1].groupby("track_id")["time"].agg(lambda s: s.value_counts().idxmax())
    cont = moda.value_counts()
    cores = m.get("times_bgr", {})
    nomes = _json(p / "times_nomes.json") or {}
    linhas = [
        ("Modelo / imgsz", f"{m['modelo']} / {m['imgsz']}"),
        ("Device", m["device"]),
        ("Rastreador", m.get("tracker", "bytetrack.yaml")),
        ("Quadros processados", m["n_frames"]),
        ("Velocidade de processamento", f"{m['fps_processamento']:.1f} fps"),
        ("Pessoas por quadro (média)", f"{por_q.mean():.1f}"),
        ("IDs únicos de pessoa", j["track_id"].nunique()),
        ("IDs por time (0 / 1 / outros)",
         f"{cont.get(0, 0)} / {cont.get(1, 0)} / {cont.get(-2, 0)}"),
    ]
    for k, v in sorted(cores.items()):
        rot = nomes.get(k, "árbitro/goleiros" if k == "-2" else f"time {k}")
        linhas.append((f"Cor do cluster {k} (BGR) — {rot}", str(tuple(v))))
    return "| Item | Valor |\n|---|---|\n" + "".join(f"| {a} | {b} |\n" for a, b in linhas)


def b_bola(p: Path) -> str:
    b = _json(p / "bola_resumo.json")
    if b is None:
        return _pendente("bola_resumo.json")
    no_campo = b.get("quadros_com_deteccao_no_campo", "—")
    return ("| Quadros | Com detecção YOLO | Candidatas (fora do campo) | Quadros com bola no campo | "
            "Aceitos após gating | Preenchidos pelo Kalman | Cobertura final |\n"
            "|---:|---:|---:|---:|---:|---:|---:|\n"
            f"| {b['quadros']} | {b['quadros_com_deteccao_yolo']} ({b['pct_quadros_com_deteccao_yolo']}%) | "
            f"{b['candidatas']} ({b.get('candidatas_fora_do_campo', 0)}) | {no_campo} | "
            f"{b['aceitas_apos_gating']} | {b['preenchidas_kalman']} | {b['cobertura_final_pct']}% |\n")


def b_calibracao(p: Path) -> str:
    c = _json(p / "calibracao.json")
    if c is None or "erro_reprojecao_m" not in c:
        return _pendente("calibracao.json (rode ferramentas/clicar_marcos.py)")
    d = pd.DataFrame({"marco": c["marcos"], "pixel (u, v)": [f"({u:.0f}, {v:.0f})" for u, v in c["pixels"]],
                      "erro de reprojeção (m)": c["erro_reprojecao_m"]})
    e = pd.Series(c["erro_reprojecao_m"])
    return (_tabela(d, {"erro de reprojeção (m)": ".2f"}) +
            f"\nErro médio **{e.mean():.2f} m**, máximo **{e.max():.2f} m** ({len(e)} marcos no frame0).\n")


def b_camera(p: Path) -> str:
    """Âncoras manuais e validação independente da compensação de câmera."""
    anc = _json(p / "calibracao_ancoras.json")
    ind = _csv(p / "camera_validacao_independente.csv")
    if anc is None or ind is None:
        return _pendente("calibracao_ancoras.json / camera_validacao_independente.csv")
    a_ = pd.DataFrame([{"quadro": int(f), "t (s)": int(f) / 25, "marcos clicados": len(v["marcos"]),
                        "erro reproj. médio (m)": float(np.mean(v["erro_reprojecao_m"]))}
                       for f, v in sorted(anc.items(), key=lambda kv: int(kv[0]))])
    v = ind.rename(columns={"t_s": "t (s)", "pontos_grade": "pontos da grade", "erro_medio_m": "erro médio (m)",
                            "erro_mediano_m": "erro mediano (m)", "erro_max_m": "erro máx (m)",
                            "usada_no_calculo": "âncora usada no cálculo?"})
    cam = _json(p / "camera_resumo_gramado.json") or _json(p / "camera_resumo.json")
    ref = _json(p / "camera_refino_resumo.json") or {}
    txt = ""
    if cam:
        txt = (f"Câmera: {cam['quadros']} quadros, {len(cam['cortes_frames'])} cortes detectados; o pan desloca o "
               f"centro da imagem até {cam['desloc_max_px']:.0f} px em relação ao quadro 0.\n\n")
    return (txt + "**Âncoras manuais** (`ferramentas/calibrar_ancoras.py`):\n\n" +
            _tabela(a_, {"t (s)": ".0f", "erro reproj. médio (m)": ".2f"}) +
            "\n**Validação independente** do método final (pares ORB consecutivos + refino pelas linhas do campo), "
            "inicializado **só** no quadro 0: erro da grade de 5 × 4 m visível em cada âncora manual, que NÃO "
            "entrou no cálculo:\n\n" + _tabela(v, {"t (s)": ".0f"}) +
            f"\nNa versão final as âncoras também reiniciam a estimativa; "
            f"{ref.get('quadros_refinados_pct', '?')}% dos quadros foram refinados pelas linhas "
            f"(mediana de {ref.get('inliers_medianos', '?')} pontos de linha por quadro).\n")


def b_metricas(p: Path) -> str:
    m = _json(p / "metricas_times.json")
    if m is None:
        return _pendente("metricas_times.json")
    nomes = _json(p / "times_nomes.json") or {}
    linhas = []
    for t in ("0", "1"):
        r = m[f"time{t}"]
        linhas.append({"time": f"{t} ({nomes.get(t, '?')})", "posse (%)": r["posse_pct"], "passes": r["passes"],
                       "passes certos": r["passes_certos"], "precisão (%)": r["precisao_pct"],
                       "distância total (m)": r["distancia_total"]})
    return (f"Raio de posse: **{m['raio_posse']} {m['unidade']}** · quadros com bola: "
            f"{m['quadros_com_bola']}/{m['quadros_total']} · direção de ataque: "
            f"time 0 → {m['direcao_ataque']['0']}, time 1 → {m['direcao_ataque']['1']}\n\n"
            + _tabela(pd.DataFrame(linhas)))


def b_jogadores(p: Path) -> str:
    d = _csv(p / "metricas_jogadores.csv")
    if d is None:
        return _pendente("metricas_jogadores.csv")
    d = d[d["tempo_s"] >= 60].sort_values("distancia", ascending=False).head(8)
    cols = ["track_id", "time", "tempo_s", "distancia", "vel_media", "vel_max_kmh"]
    d = d[[c for c in cols if c in d]].copy()
    d["m_por_min"] = d["distancia"] / (d["tempo_s"] / 60)
    return "Jogadores rastreados por ≥ 60 s (top 8 em distância):\n\n" + _tabela(
        d, {"tempo_s": ".0f", "distancia": ".0f", "vel_media": ".2f", "vel_max_kmh": ".1f", "m_por_min": ".0f"})


def b_sensibilidade(p: Path) -> str:
    d = _csv(p / "sensibilidade_raio.csv")
    return _pendente("sensibilidade_raio.csv") if d is None else _tabela(d, {"raio_m": ".0f"})


def b_validacao(p: Path) -> str:
    v = _json(p / "validacao_passes.json")
    if v is None:
        return _pendente("validacao_passes.json (anote os passes e rode validar_passes.py)")
    return ("| Detectados | Anotados | Casados (VP) | FP | FN | Precisão | Revocação | F1 | "
            "Acordo certo/errado |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            f"| {v['detectados']} | {v['anotados']} | {v['casados']} | {v['falsos_positivos']} | "
            f"{v['falsos_negativos']} | {v['precisao']:.2f} | {v['revocacao']:.2f} | {v['f1']:.2f} | "
            f"{v['acordo_sucesso']:.2f} |\n\nTolerância temporal: ±{v['tolerancia_s']} s. "
            f"Passes certos — anotados: {v['anotados_certos_pct']}%, detectados: {v['detectados_certos_pct']}%.\n")


def _tabela_xp(p: Path, sufixo: str) -> str | None:
    d = _csv(p / f"resultados_xp{sufixo}.csv")
    feats = _csv(p / f"passes_features{sufixo}.csv")
    info = _json(p / f"xp_info{sufixo}.json") or {}
    if d is None:
        return None
    d = d.rename(columns={d.columns[0]: "modelo"})
    n = f"{len(feats)} passes ({100 * feats['sucesso'].mean():.1f}% certos, " \
        f"{int((feats['sucesso'] == 0).sum())} errados)" if feats is not None else "?"
    return f"Base: **{n}** · rótulos: {info.get('rotulos', '?')}.\n\n" + _tabela(d)


def b_xp(p: Path) -> str:
    a = _tabela_xp(p, "")
    b = _tabela_xp(p, "_rotulos_detector")
    if a is None and b is None:
        return _pendente("resultados_xp.csv")
    ref = ("**Referência de literatura** (outra base, ~7 mil passes de teste — não comparável diretamente):\n\n"
           "| modelo | AUC-ROC (↑) | Brier (↓) | Log-Loss (↓) |\n|---|---:|---:|---:|\n"
           "| SoccerMap · Gomes et al. (2026) | 0.622 | 0.138 | 0.444 |\n"
           "| Baseline de distância · Gomes et al. (2026) | 0.588 | 0.141 | 0.456 |\n")
    out = ""
    if a is not None:
        out += "**A — rótulos da anotação manual** (passes detectados que casaram com `passes_anotados.csv`):\n\n" + a
        if "logistico" not in a:
            out += ("\n_Com tão poucos passes errados não há como ajustar baseline/logístico com validação "
                    "cruzada; as métricas acima (só 2 negativos) não têm significância estatística._\n")
    if b is not None:
        out += ("\n**B — rótulos do próprio detector** (troca de dono; todos os passes detectados, "
                "validação cruzada estratificada):\n\n" + b)
    return out + "\n" + ref


def b_pax(p: Path) -> str:
    d = _csv(p / "jogadores_xp.csv")
    if d is None:
        return _pendente("jogadores_xp.csv")
    w = _csv(p / "what_if.csv")
    txt = ""
    if w is not None and len(w):
        txt = (f"\nWhat-if (destino deslocado 3–6 m, {len(w)} passes errados): ganho médio de eXP "
               f"**{w['ganho'].mean():.3f}**; {int((w['ganho'] > 0.10).sum())} passes com ganho > 0,10.\n")
    return "Top 5 PAx (passes certos − ΣeXP):\n\n" + _tabela(d.head(5), {"eXP": ".2f", "PAx": ".2f"}) + txt


BLOCOS = {"video": b_video, "yolo": b_yolo, "deteccao": b_deteccao, "bola": b_bola,
          "calibracao": b_calibracao, "camera": b_camera, "metricas": b_metricas, "jogadores": b_jogadores,
          "sensibilidade": b_sensibilidade, "validacao": b_validacao, "xp": b_xp, "pax": b_pax}


def preencher(texto: str, pasta: Path) -> str:
    def sub(m):
        nome = m.group(2)
        conteudo = BLOCOS[nome](pasta) if nome in BLOCOS else f"_(bloco desconhecido: {nome})_\n"
        return m.group(1) + conteudo + m.group(4)
    return PADRAO.sub(sub, texto)


def copiar_figuras(pasta: Path, destino: Path) -> list[str]:
    destino.mkdir(exist_ok=True)
    copiadas = []
    for f in sorted(list((pasta / "figuras").glob("*.png")) + list((pasta / "figuras").glob("*.jpg"))):
        shutil.copy2(f, destino / f.name)
        copiadas.append(f.name)
    sub = pasta / "figuras" / "comparar_yolo"
    if sub.exists():
        (destino / "comparar_yolo").mkdir(exist_ok=True)
        for f in sorted(sub.glob("*.jpg")):
            shutil.copy2(f, destino / "comparar_yolo" / f.name)
            copiadas.append(f"comparar_yolo/{f.name}")
    for f in ("calibracao_grade.jpg", "frame0.jpg"):
        if (pasta / f).exists():
            shutil.copy2(pasta / f, destino / f)
            copiadas.append(f)
    return copiadas


def main():
    ap = argparse.ArgumentParser(description="Números do README gerados a partir de saidas/")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--verificar", action="store_true", help="Só confere; exit 1 se divergir")
    a = ap.parse_args()

    pasta, readme = Path(a.saida), Path(a.readme)
    atual = readme.read_text(encoding="utf-8")
    novo = preencher(atual, pasta)
    blocos = [m.group(2) for m in PADRAO.finditer(atual)]

    if a.verificar:
        dif = [m.group(2) for m, n in zip(PADRAO.finditer(atual), PADRAO.finditer(novo))
               if m.group(3) != n.group(3)]
        pend = [m.group(2) for m in PADRAO.finditer(atual) if "_(pendente" in m.group(3)]
        faltam = [f for f in re.findall(r"\]\((figuras/[^)]+)\)", atual) if not Path(f).exists()]
        print(f"Blocos AUTO: {len(blocos)} | divergentes: {dif or 'nenhum'} | pendentes: {pend or 'nenhum'}")
        print(f"Figuras referenciadas inexistentes: {faltam or 'nenhuma'}")
        sys.exit(1 if dif or faltam else 0)

    readme.write_text(novo, encoding="utf-8")
    figs = copiar_figuras(pasta, Path("figuras"))
    print(f"README atualizado ({len(blocos)} blocos AUTO); {len(figs)} figuras copiadas para figuras/")


if __name__ == "__main__":
    main()
