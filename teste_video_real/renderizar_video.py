import cv2
import numpy as np

# 1. Carregar dados extraídos
data = np.load('pose_data.npz')
kpts = data['keypoints']      # (N_frames, 17, 2)
vis = data['visibility']       # (N_frames, 17)
fps = int(data['fps'])
width, height = data['dimensions']

# Conexões do esqueleto COCO (17 articulações)
COCO_LIMBS = [
    (0, 1), (0, 2), (1, 3), (2, 4),         # Cabeça
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Tronco e membros superiores
    (5, 11), (6, 12), (11, 12),              # Anca
    (11, 13), (13, 15), (12, 14), (14, 16)   # Membros inferiores
]

def simular_ekf_estavel(kpts_seq, vis_seq, max_vel=15.0):
    N, J, _ = kpts_seq.shape
    kpts_pred = np.copy(kpts_seq)
    for j in range(J):
        last_pos = None
        last_vel = np.array([0.0, 0.0])
        for t in range(N):
            if vis_seq[t, j] == 1 and not (kpts_seq[t, j, 0] == 0 and kpts_seq[t, j, 1] == 0):
                if last_pos is not None:
                    vel = kpts_seq[t, j] - last_pos
                    last_vel = np.clip(vel, -max_vel, max_vel)
                last_pos = kpts_seq[t, j].astype(float).copy()
                kpts_pred[t, j] = kpts_seq[t, j]
            else:
                if last_pos is not None:
                    last_pos += last_vel
                    kpts_pred[t, j] = last_pos.copy()
                    last_vel *= 0.90
                else:
                    kpts_pred[t, j] = np.array([0.0, 0.0])
    return kpts_pred

kpts_estimados = simular_ekf_estavel(kpts, vis)

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