"""Transformer temporal e T-GNN (grafo COCO-17) sobre o vídeo real. Requer pose_data.npz (processar_pose.py)."""
from modelos_pose import comparar_jitter

comparar_jitter(["Transformer", "T-GNN"], "comparativo_transformer_tgnn.png")
