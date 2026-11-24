"""LSTM e GRU sobre o vídeo real (COCO-17). Requer pose_data.npz (processar_pose.py)."""
from modelos_pose import comparar_jitter

comparar_jitter(["LSTM", "GRU"], "comparativo_redes_temporais_treinadas.png")
