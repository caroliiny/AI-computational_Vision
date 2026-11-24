import cv2
import numpy as np

import modelos_pose  # noqa: F401  (ajusta o sys.path para coco17 e EKF; deve vir primeiro)
from coco17 import ARESTAS as COCO_LIMBS, carregar_pose
from ekf import rodar_ekf_pose

# 1. Carregar dados extraídos (padrão COCO-17)
pose = carregar_pose('pose_data.npz')
kpts, vis, fps = pose['kpts'], pose['vis'], pose['fps']
width, height = pose['W'], pose['H']

# EKF real (um filtro por junta); só as juntas ocluídas usam a estimativa
kpts_estimados = np.where((vis == 0)[..., None], rodar_ekf_pose(kpts, vis), kpts)

# Configurar vídeo de saída
cap = cv2.VideoCapture('video_teste.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('resultado_visual_benchmark.mp4', fourcc, fps, (width, height))

frame_idx = 0
print("A renderizar o vídeo anotado...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(kpts):
        break

    kpt_frame = kpts_estimados[frame_idx]
    vis_frame = vis[frame_idx]

    # Desenhar conexões (ossos)
    for p1, p2 in COCO_LIMBS:
        pt1 = (int(kpt_frame[p1, 0]), int(kpt_frame[p1, 1]))
        pt2 = (int(kpt_frame[p2, 0]), int(kpt_frame[p2, 1]))
        if pt1 != (0, 0) and pt2 != (0, 0):
            color_limb = (255, 128, 0) if (vis_frame[p1] == 1 and vis_frame[p2] == 1) else (0, 165, 255)
            cv2.line(frame, pt1, pt2, color_limb, 2)

    # Desenhar nós (articulações)
    for j in range(17):
        x, y = int(kpt_frame[j, 0]), int(kpt_frame[j, 1])
        if x > 0 and y > 0:
            color_node = (0, 255, 0) if vis_frame[j] == 1 else (0, 0, 255)
            cv2.circle(frame, (x, y), 5, color_node, -1)

    # Legendas
    cv2.putText(frame, f"Quadro: {frame_idx + 1}/{len(kpts)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Verde: Visivel (YOLO) | Vermelho: Ocluido (Estimado)", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("Vídeo guardado com sucesso como 'resultado_visual_benchmark.mp4'.")