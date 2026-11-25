"""Transformer e T-GNN sobre o vídeo real. Uso: python testar_transformer_tgnn.py [pose.npz] [--png saida.png]"""
from modelos_pose import executar_jitter

executar_jitter(["Transformer", "T-GNN"], "comparativo_transformer_tgnn.png")
