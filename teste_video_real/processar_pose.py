"""Extrai o esqueleto COCO-17 (YOLOv8-pose) de video_teste.mp4 e salva pose_data.npz (formato em coco17.py)."""
import cv2
import numpy as np
from ultralytics import YOLO

print("A carregar o modelo YOLOv8-Pose (download automático na 1ª execução)...")
model = YOLO('yolov8x-pose.pt')

video_path = 'video_teste.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Erro: Não foi possível abrir o vídeo '{video_path}'. Verifique o nome do ficheiro.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

CONF_THRESHOLD = 0.45  # junta com confiança abaixo disso é tratada como ocluída
keypoints_history, visibility_history, confidence_history = [], [], []
centro_anterior = None  # centro da caixa da pessoa seguida (evita trocar de pessoa entre quadros)

print("A processar os quadros do vídeo...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    r = model(frame, verbose=False)[0]
    kpts_all = r.keypoints.data.cpu().numpy()          # (n_pessoas, 17, 3)

    if len(kpts_all) > 0:
        centros = r.boxes.xywh[:, :2].cpu().numpy()
        if centro_anterior is None:                    # 1ª detecção: a pessoa mais confiável
            i = int(np.argmax(r.boxes.conf.cpu().numpy()))
        else:                                          # depois: a mais próxima da anterior
            i = int(np.argmin(np.linalg.norm(centros - centro_anterior, axis=1)))
        centro_anterior = centros[i]
        xy, conf = kpts_all[i, :, :2], kpts_all[i, :, 2]
    else:                                              # ninguém detectado (oclusão total)
        xy, conf = np.zeros((17, 2)), np.zeros(17)

    keypoints_history.append(xy)
    confidence_history.append(conf)
    visibility_history.append((conf > CONF_THRESHOLD).astype(int))

cap.release()

np.savez('pose_data.npz',
         keypoints=np.array(keypoints_history),
         visibility=np.array(visibility_history),
         confidence=np.array(confidence_history),
         fps=fps, dimensions=(width, height))

print(f" Concluído! {len(keypoints_history)} quadros salvos em 'pose_data.npz'.")
