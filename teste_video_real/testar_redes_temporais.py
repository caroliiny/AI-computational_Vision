"""LSTM e GRU sobre o vídeo real. Uso: python testar_redes_temporais.py [pose.npz] [--png saida.png]"""
from modelos_pose import executar_jitter

executar_jitter(["LSTM", "GRU"], "comparativo_redes_temporais_treinadas.png")
