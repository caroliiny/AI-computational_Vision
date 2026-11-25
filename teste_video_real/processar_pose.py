"""Extrai o esqueleto COCO-17 (YOLOv8-pose) de um vídeo e salva o .npz (formato em coco17.py).

Uso: python processar_pose.py [video.mp4] [--saida pose.npz] [--conf 0.45] [--modelo yolov8x-pose.pt]
Sem --saida, grava <nome_do_video>_pose.npz (não sobrescreve o de outros vídeos).

A função extrair_pose() também é usada pelo fluxo de oclusão (executar_oclusao.ps1), tanto para o vídeo
original (referência) quanto para o vídeo com oclusão (entrada dos modelos).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def extrair_pose(video, saida, conf=0.45, modelo="yolov8x-pose.pt", model=None):
    """Roda o YOLOv8-pose quadro a quadro e salva `saida` (.npz). Retorna o dict com os arrays salvos.

    `model` permite reaproveitar um YOLO já carregado (evita recarregar entre vídeos).
    """
    video = Path(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Erro: não foi possível abrir o vídeo '{video}'.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if model is None:
        from ultralytics import YOLO
        print(f"Carregando {modelo} (download automático na 1ª execução)...")
        model = YOLO(modelo)

    keypoints_history, visibility_history, confidence_history = [], [], []
    centro_anterior = None  # centro da caixa da pessoa seguida (evita trocar de pessoa entre quadros)
    sem_pessoa = 0
    varias_pessoas = 0

    print(f"Processando {video.name}: {total} quadros, {width}x{height}, {fps:.2f} fps")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        r = model(frame, verbose=False)[0]
        kpts_all = r.keypoints.data.cpu().numpy()          # (n_pessoas, 17, 3)

        if len(kpts_all) > 0:
            varias_pessoas += len(kpts_all) > 1
            centros = r.boxes.xywh[:, :2].cpu().numpy()
            if centro_anterior is None:                    # 1ª detecção: a pessoa mais confiável
                i = int(np.argmax(r.boxes.conf.cpu().numpy()))
            else:                                          # depois: a mais próxima da anterior
                i = int(np.argmin(np.linalg.norm(centros - centro_anterior, axis=1)))
            centro_anterior = centros[i]
            xy, c = kpts_all[i, :, :2], kpts_all[i, :, 2]
        else:                                              # ninguém detectado (oclusão total)
            sem_pessoa += 1
            xy, c = np.zeros((17, 2)), np.zeros(17)

        keypoints_history.append(xy)
        confidence_history.append(c)
        visibility_history.append((c > conf).astype(int))
        if len(keypoints_history) % 100 == 0:
            print(f"  {len(keypoints_history)}/{total}")

    cap.release()

    if not keypoints_history:
        sys.exit("Erro: nenhum quadro lido do vídeo.")

    dados = dict(keypoints=np.array(keypoints_history),
                 visibility=np.array(visibility_history),
                 confidence=np.array(confidence_history),
                 fps=fps, dimensions=(width, height))
    np.savez(saida, **dados)

    vis = dados["visibility"]
    n = len(vis)
    completas = sum((vis[t - 10:t + 1] == 1).all() for t in range(10, n))
    print(f"\nConcluído: {n} quadros salvos em '{saida}'.")
    print(f"  Taxa de oclusão (juntas): {(1 - vis.mean()) * 100:.1f}%")
    print(f"  Quadros sem nenhuma pessoa: {sem_pessoa}")
    print(f"  Quadros com mais de uma pessoa (o rastreio segue só uma): {varias_pessoas}")
    print(f"  Janelas de treino com as 17 juntas visíveis: {completas}")
    if completas < 30:
        print("  AVISO: poucas janelas totalmente visíveis; o treino de LSTM/GRU/Transformer/T-GNN pode falhar.")
    if n < 300:
        print("  AVISO: vídeo curto (< ~300 quadros); resultados das redes tendem a ser pouco confiáveis.")
    if vis.mean() > 0.999:
        print("  AVISO: nenhuma oclusão natural detectada; só a avaliação com oclusão artificial é útil.")
    return dados


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="video_teste.mp4")
    ap.add_argument("--saida", help="padrão: <video>_pose.npz")
    ap.add_argument("--conf", type=float, default=0.45,
                    help="junta com confiança abaixo disso é tratada como ocluída")
    ap.add_argument("--modelo", default="yolov8x-pose.pt",
                    help="ex.: yolov8n-pose.pt é bem mais rápido em CPU")
    a = ap.parse_args()
    extrair_pose(a.video, a.saida or Path(a.video).stem + "_pose.npz", a.conf, a.modelo)
