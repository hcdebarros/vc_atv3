#!/usr/bin/env python3
"""
Baixa um TRECHO de um vídeo do YouTube (720p por padrão; use --altura 1080) para a prática.

Adaptado de implementacoes/download_youtube_videos.py. A diferença principal é
que aqui o yt-dlp baixa apenas o intervalo pedido (download_ranges), em vez de
baixar o jogo inteiro e cortar depois — um jogo completo em 720p passa de 1 GB.

Uso
---
# Trecho de 3 minutos (do minuto 12 ao 15)
python baixar_video.py "https://www.youtube.com/watch?v=VIDEO_ID" --inicio 12 --fim 15

# Em 1080p, usando MM:SS
python baixar_video.py "URL" --inicio 18:20 --fim 21:20 --altura 1080

# Escolhendo a pasta de saída
python baixar_video.py "URL" --inicio 12 --fim 15 --saida ../dados/videos

Dicas para escolher o trecho
----------------------------
- Prefira a câmera principal (plano aberto, lateral), SEM replays e cortes de câmera.
- 3 a 5 minutos são suficientes; jogo corrido rende mais passes que bola parada.
- Não versione o vídeo no GitHub: registre apenas a URL e o intervalo no README.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yt_dlp
from yt_dlp.utils import download_range_func


def _parse_args():
    p = argparse.ArgumentParser(description="Baixa um trecho de vídeo do YouTube.")
    p.add_argument("url", help="URL do vídeo no YouTube")
    p.add_argument("--inicio", type=_minutos, default=None, metavar="MIN",
                   help="Início do trecho em minutos (ex.: 12.5) ou MM:SS (ex.: 12:30)")
    p.add_argument("--fim", type=_minutos, default=None, metavar="MIN",
                   help="Fim do trecho em minutos ou MM:SS")
    p.add_argument("--altura", type=int, default=720, metavar="PX",
                   help="Resolução máxima (altura em px), ex.: 720, 1080 (padrão: 720)")
    p.add_argument("--saida", default="videos", help="Pasta de saída (padrão: videos/)")
    return p.parse_args()


def _minutos(valor: str) -> float:
    """Converte '12.5', '12:30' ou '1:02:30' (H:MM:SS) para minutos."""
    if ":" not in valor:
        return float(valor)
    segundos = 0.0
    for parte in valor.split(":"):
        segundos = segundos * 60 + float(parte)
    return segundos / 60


def _nome_seguro(nome: str) -> str:
    """Remove caracteres problemáticos para nomes de arquivo."""
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in nome).strip()


def _localizar_ffmpeg() -> str | None:
    """ffmpeg do PATH ou, na falta dele, o binário do pacote imageio-ffmpeg."""
    caminho = shutil.which("ffmpeg")
    if caminho:
        return caminho
    try:
        import imageio_ffmpeg
        origem = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None
    # O yt-dlp só reconhece o executável se ele se chamar "ffmpeg(.exe)".
    destino = Path.home() / ".cache" / "baixar_video" / ("ffmpeg" + origem.suffix)
    if not destino.exists():
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origem, destino)
    return str(destino)


def baixar(url: str, inicio_min: float | None, fim_min: float | None, saida: str,
           altura: int = 720) -> Path:
    pasta = Path(saida)
    pasta.mkdir(parents=True, exist_ok=True)

    trecho = inicio_min is not None or fim_min is not None
    ffmpeg = _localizar_ffmpeg()
    if (trecho or altura > 720) and ffmpeg is None:
        sys.exit("Erro: o ffmpeg é necessário (trechos/1080p) e não foi encontrado. "
                 "Instale-o no PATH ou rode: pip install imageio-ffmpeg")

    # Formato: melhor vídeo MP4 até a altura pedida + melhor áudio M4A
    h = altura
    opcoes = {
        "format": (f"bestvideo[height<={h}][ext=mp4]+bestaudio[ext=m4a]/"
                   f"bestvideo[height<={h}]+bestaudio/best[height<={h}]"),
        "merge_output_format": "mp4",
        "outtmpl": str(pasta / "%(title)s.%(ext)s"),
    }
    if ffmpeg:
        opcoes["ffmpeg_location"] = ffmpeg
        # Algumas checagens do yt-dlp ignoram ffmpeg_location e olham só o PATH.
        os.environ["PATH"] = str(Path(ffmpeg).parent) + os.pathsep + os.environ.get("PATH", "")

    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    titulo = _nome_seguro(info.get("title", "video"))
    duracao = float(info.get("duration") or 0)

    if trecho:
        ini_s = (inicio_min or 0.0) * 60
        fim_s = fim_min * 60 if fim_min is not None else duracao
        if fim_s <= ini_s:
            sys.exit("Erro: --fim deve ser maior que --inicio.")
        # Baixa somente o intervalo [ini_s, fim_s]; cortar nos keyframes evita
        # quadros pretos no início do clipe.
        opcoes["download_ranges"] = download_range_func(None, [(ini_s, fim_s)])
        opcoes["force_keyframes_at_cuts"] = True
        opcoes["outtmpl"] = str(pasta / f"{titulo}_clip.%(ext)s")
        print(f"Baixando trecho {int(ini_s//60)}:{int(ini_s%60):02d}–{int(fim_s//60)}:{int(fim_s%60):02d} "
              f"(≤{h}p) de: {info.get('title')}")
    else:
        opcoes["outtmpl"] = str(pasta / f"{titulo}.%(ext)s")
        print(f"Baixando vídeo completo ({duracao/60:.0f} min): {info.get('title')}")

    with yt_dlp.YoutubeDL(opcoes) as ydl:
        ydl.download([url])

    arquivo = pasta / (f"{titulo}_clip.mp4" if trecho else f"{titulo}.mp4")
    print(f"Pronto. Arquivo: {arquivo}")
    return arquivo


if __name__ == "__main__":
    a = _parse_args()
    baixar(a.url, a.inicio, a.fim, a.saida, a.altura)
