import cv2
import numpy as np
from ultralytics import YOLO

# 1. Carregar o modelo YOLOv8-Pose (farpa download automático na 1ª execução)
print("A carregar o modelo YOLOv8-Pose...")
model = YOLO('yolov8x-pose.pt')

video_path = 'video_teste.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Erro: Não foi possível abrir o vídeo '{video_path}'. Verifique o nome do ficheiro.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

keypoints_history = []
visibility_history = []

CONF_THRESHOLD = 0.45  # Limiar de confiança para considerar oclusão

print("A processar os quadros do vídeo...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    if len(results[0].keypoints) > 0 and results[0].keypoints.data.shape[1] > 0:
        # Extrai coordenadas e confiança do primeiro alvo humano
        kpts = results[0].keypoints.data[0].cpu().numpy()  # Shape (17, 3)
        xy = kpts[:, :2]
        conf = kpts[:, 2]

        # Visibilidade: 1 para junta visível, 0 para junta ocluída
        vis = (conf > CONF_THRESHOLD).astype(int)
    else:
        # Se a pessoa estiver 100% oculta atrás do pilar
        xy = np.zeros((17, 2))
        vis = np.zeros(17, dtype=int)

    keypoints_history.append(xy)
    visibility_history.append(vis)

cap.release()

# Guardar os dados extraídos num ficheiro compacto .npz
np.savez('pose_data.npz', 
         keypoints=np.array(keypoints_history), 
         visibility=np.array(visibility_history),
         fps=fps, dimensions=(width, height))

print(f" Concluído! {len(keypoints_history)} quadros salvos em 'pose_data.npz'.")