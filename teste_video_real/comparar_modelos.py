"""EKF real (um filtro por junta COCO-17) sobre o vídeo real. Requer pose_data.npz (processar_pose.py)."""
from modelos_pose import comparar_jitter

comparar_jitter(["EKF"], "trajetoria_pulso.png")
