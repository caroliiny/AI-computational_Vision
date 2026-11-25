"""EKF real (um filtro por junta) sobre o vídeo real. Uso: python comparar_modelos.py [pose.npz] [--png saida.png]"""
from modelos_pose import executar_jitter

executar_jitter(["EKF"], "trajetoria_pulso.png")
