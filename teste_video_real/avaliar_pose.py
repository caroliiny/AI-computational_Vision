"""
Avaliação com erro em pixels dos cinco métodos no padrão COCO-17 (vale para vídeo real e sintético).

  - Oclusão ARTIFICIAL (sempre): esconde grupos de juntas bem detectadas e compara a previsão com a
    detecção original. É a forma de ter RMSE num vídeo real, que não tem gabarito.
  - Oclusão NATURAL (só se o .npz tiver keypoints_verdade, i.e. vídeo sintético): erro nas juntas
    atrás do pilar contra a posição verdadeira.

Uso: python avaliar_pose.py [pose_data.npz] [--saida metricas.csv] [--epocas 150]
"""
import argparse
import csv
from pathlib import Path

from coco17 import carregar_pose, mascarar_oclusao
from modelos_pose import erro_px, rodar_metodos


def avaliar(nome_cenario, kpts, vis, verdade, mascara, p, epocas, linhas):
    saidas = rodar_metodos(kpts, vis, p["W"], p["H"], epocas=epocas)
    for metodo, est in saidas.items():
        medio, rmse = erro_px(est, verdade, mascara)
        linhas.append([nome_cenario, metodo, f"{medio:.2f}", f"{rmse:.2f}", int(mascara.sum())])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada", nargs="?", default="pose_data.npz")
    ap.add_argument("--saida", help="padrão: <entrada>_metricas.csv")
    ap.add_argument("--epocas", type=int, default=150)
    a = ap.parse_args()

    a.saida = a.saida or f"{Path(a.entrada).stem}_metricas.csv"
    p = carregar_pose(a.entrada)
    linhas = []

    vis_art, mascara_art = mascarar_oclusao(p["vis"])
    print(f"\n### Oclusão artificial ({int(mascara_art.sum())} pontos escondidos)")
    avaliar("artificial", p["kpts"], vis_art, p["kpts"], mascara_art, p, a.epocas, linhas)

    if p["verdade"] is not None:
        mascara_nat = p["vis"] == 0
        print(f"\n### Oclusão natural, contra a verdade ({int(mascara_nat.sum())} pontos)")
        avaliar("natural", p["kpts"], p["vis"], p["verdade"], mascara_nat, p, a.epocas, linhas)

    with open(a.saida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cenario", "metodo", "erro_medio_px", "rmse_px", "n_pontos"])
        w.writerows(linhas)

    print("\ncenario      metodo         erro medio (px)   RMSE (px)   pontos")
    for c, m, e, r, n in linhas:
        print(f"{c:<12} {m:<14} {e:>15} {r:>11} {n:>8}")
    print(f"\nSalvo: {a.saida}")
