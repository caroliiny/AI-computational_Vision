"""
Visualiza os resultados de avaliar_previsao.py.

Uso: python visualizar_previsao.py pasta_de_trabalho [--sem-video]
Grava em <pasta>:
  grafico_horizonte.png    erro médio por horizonte (5/10/20 quadros) e método
  grafico_categoria.png    erro por tipo de oclusão (parcial, grande, quase_total, total) e método
  grafico_trajetorias.png  exemplos de oclusão: referência x previsão de cada método
  video_previsoes.mp4      vídeo com oclusão (trecho de teste); verde = referência do vídeo original,
                           pontos coloridos = previsão de cada método para a junta oculta
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from protocolo import H_MAX, carregar_dados  # deve vir primeiro: ajusta o sys.path (coco17, EKF)
from coco17 import NOMES

ORDEM_CATEGORIAS = ["parcial", "grande", "quase_total", "total", "natural"]
CORES = {"Último valor": "#888888", "EKF": "#ff8c00", "LSTM": "#1f77b4", "GRU": "#d62728",
         "Transformer": "#e0c000", "T-GNN": "#17becf"}


def bgr(hexa):
    return tuple(int(hexa[i:i + 2], 16) for i in (5, 3, 1))


def grafico_horizonte(pasta, metodos):
    df = pd.read_csv(pasta / "metricas_horizonte.csv")
    hs = sorted(df["horizonte"].unique())
    larg = 0.8 / len(metodos)
    plt.figure(figsize=(9, 5))
    for i, m in enumerate(metodos):
        v = [df[(df.metodo == m) & (df.horizonte == h)]["erro_medio_px"].mean() for h in hs]
        plt.bar(np.arange(len(hs)) + i * larg, v, larg, label=m, color=CORES[m])
    plt.xticks(np.arange(len(hs)) + 0.4 - larg / 2, [f"{h} quadros" for h in hs])
    plt.ylabel("Erro médio (px)")
    plt.title("Previsão sob oclusão — erro por horizonte (menor = melhor)")
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(pasta / "grafico_horizonte.png", dpi=150); plt.close()


def grafico_categoria(pasta, metodos):
    df = pd.read_csv(pasta / "metricas_categoria.csv")
    cats = [c for c in ORDEM_CATEGORIAS if c in set(df["categoria"])]
    hs = sorted(df["horizonte"].unique())
    fig, eixos = plt.subplots(1, len(hs), figsize=(5 * len(hs), 4.5), sharey=True, squeeze=False)
    larg = 0.8 / len(metodos)
    for ax, h in zip(eixos[0], hs):
        for i, m in enumerate(metodos):
            v = [df[(df.metodo == m) & (df.horizonte == h) & (df.categoria == c)]["erro_medio_px"].mean()
                 for c in cats]
            ax.bar(np.arange(len(cats)) + i * larg, v, larg, label=m, color=CORES[m])
        ax.set_xticks(np.arange(len(cats)) + 0.4 - larg / 2)
        ax.set_xticklabels(cats, rotation=20)
        ax.set_title(f"{h} quadros à frente"); ax.grid(axis="y", alpha=0.3)
    eixos[0][0].set_ylabel("Erro médio (px)")
    eixos[0][0].legend(fontsize=8)
    fig.suptitle("Erro por tipo de oclusão (categorias sem previsão avaliável ficam em branco)")
    fig.tight_layout(); fig.savefig(pasta / "grafico_categoria.png", dpi=150); plt.close(fig)


def escolher_eventos(eventos, n=4):
    """Um evento de cada cenário (o mais longo, com preferência por >= 20 quadros), até n."""
    melhor = {}
    for e in eventos:
        if e["cenario"] not in melhor or e["duracao"] > melhor[e["cenario"]]["duracao"]:
            melhor[e["cenario"]] = e
    return sorted(melhor.values(), key=lambda e: -e["duracao"])[:n]


def grafico_trajetorias(pasta, d, metodos, prev, indice, eventos):
    esc = escolher_eventos(eventos)
    fig, eixos = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    for ax, e in zip(eixos.ravel(), esc):
        t0, j = e["t0"], e["junta"]
        k = min(e["duracao"], H_MAX)
        qs = np.arange(t0 - 15, t0 + k + 1)
        ax.plot(qs, d["ref"][qs, j, 0], "k-", lw=2, label="Referência (original)")
        visiveis = d["obs_vis"][qs, j] == 1
        ax.plot(qs[visiveis], d["obs"][qs[visiveis], j, 0], "g.", ms=6, label="Visível no vídeo com oclusão")
        for i, m in enumerate(metodos):
            ax.plot(np.arange(t0 + 1, t0 + k + 1), prev[i, indice[t0], :k, j, 0], "--", color=CORES[m], label=m)
        ax.axvspan(t0 + 0.5, t0 + k + 0.5, color="gray", alpha=0.15)
        ax.set_title(f"{e['cenario']} — {NOMES[j]} (oclusão a partir do quadro {t0 + 1})")
        ax.set_xlabel("Quadro"); ax.set_ylabel("x (px)"); ax.grid(alpha=0.3)
    for ax in eixos.ravel()[len(esc):]:
        ax.axis("off")
    eixos[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(pasta / "grafico_trajetorias.png", dpi=150); plt.close(fig)


def video(pasta, d, metodos, prev, indice, eventos):
    cap = cv2.VideoCapture(str(pasta / "video_oclusao.mp4"))
    out = cv2.VideoWriter(str(pasta / "video_previsoes.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                          d["fps"], (d["W"], d["H"]))
    por_quadro = {}                                   # quadro -> [(evento, passo)]
    for e in eventos:
        for k in range(1, min(e["duracao"], H_MAX) + 1):
            por_quadro.setdefault(e["t0"] + k, []).append((e, k))
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if t >= d["corte"]:
            for e, k in por_quadro.get(t, []):
                j = e["junta"]
                if d["ref_vis"][t, j]:
                    cv2.circle(frame, tuple(int(v) for v in d["ref"][t, j]), 9, (0, 255, 0), 2)
                for i, m in enumerate(metodos):
                    p = prev[i, indice[e["t0"]], k - 1, j]
                    if not np.isnan(p).any():
                        cv2.circle(frame, (int(p[0]), int(p[1])), 4, bgr(CORES[m]), -1)
            cv2.putText(frame, f"Quadro {t}  (teste)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "anel verde = referencia (video original)", (15, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            for i, m in enumerate(metodos):
                cv2.putText(frame, m, (15, 82 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr(CORES[m]), 2)
            out.write(frame)
        t += 1
    cap.release(); out.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta")
    ap.add_argument("--sem-video", action="store_true")
    a = ap.parse_args()
    pasta = Path(a.pasta)

    with open(pasta / "treino.json", encoding="utf-8") as f:
        frac = json.load(f)["frac_treino"]
    d = carregar_dados(pasta, frac)
    z = np.load(pasta / "previsoes.npz")
    metodos, prev = [str(m) for m in z["metodos"]], z["previsoes"]
    with open(pasta / "eventos.json", encoding="utf-8") as f:
        ev = json.load(f)
    indice = {t0: i for i, t0 in enumerate(ev["origens"])}

    grafico_horizonte(pasta, metodos)
    grafico_categoria(pasta, metodos)
    grafico_trajetorias(pasta, d, metodos, prev, indice, ev["eventos"])
    if not a.sem_video:
        video(pasta, d, metodos, prev, indice, ev["eventos"])
    print(f"Gráficos e vídeo gravados em {pasta}")
