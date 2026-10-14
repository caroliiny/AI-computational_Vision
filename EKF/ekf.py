"""
EKF para estimar a posição do alvo durante a oclusão (Pessoa 2 - Aretuza).

Entrada : trajetoria_real.csv  (frame,x_real,y_real,visivel)  -- gerado pela Clara
Saída   : previsoes_ekf.csv    (frame,x_ekf,y_ekf,oclusao)
          trajetoria_ekf.png   (gráfico)

Regra principal: onde visivel == 0 o filtro SÓ faz predição.
x_real/y_real desses quadros nunca entram como observação.

Uso: python ekf.py [trajetoria_real.csv] [previsoes_ekf.csv]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Parâmetros (ajustáveis; ver explicação no .md) ----------------
SIGMA_OBS = 1.5                      # incerteza da observação (mesma unidade das coords)
Q_DIAG = [0.05, 0.05, 0.005, 0.002, 1e-5]   # ruído de processo: x, y, v, theta, omega
N_INIT = 10                          # quadros visíveis usados para iniciar o estado
DT = 1.0                             # 1 quadro (velocidade em unidades/quadro)

# Estado: [x, y, v, theta, omega]   (modelo de movimento não linear - CTRV)
H = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0]], dtype=float)
Q = np.diag(Q_DIAG)
R = np.eye(2) * SIGMA_OBS ** 2


def f(x):
    """Modelo de transição: posição avança na direção theta; theta gira com omega."""
    px, py, v, th, om = x
    return np.array([px + v * np.cos(th) * DT,
                     py + v * np.sin(th) * DT,
                     v,
                     th + om * DT,
                     om])


def jacobiana(x):
    """Jacobiana de f no estado atual (linearização do EKF)."""
    _, _, v, th, _ = x
    return np.array([[1, 0, np.cos(th) * DT, -v * np.sin(th) * DT, 0],
                     [0, 1, np.sin(th) * DT,  v * np.cos(th) * DT, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 1, DT],
                     [0, 0, 0, 0, 1]])


def predicao(x, P):
    F = jacobiana(x)
    return f(x), F @ P @ F.T + Q


def correcao(x, P, z):
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)               # ganho de Kalman
    return x + K @ (z - H @ x), (np.eye(5) - K @ H) @ P


def rodar_ekf(frames, z, visivel):
    """z: (T,2) observações (ignoradas onde visivel == 0). Retorna (T,2) estimativas."""
    T = len(frames)
    idx_vis = np.where(visivel)[0]
    if len(idx_vis) < N_INIT:
        raise ValueError("Poucos quadros visíveis para iniciar o filtro.")
    a, b = idx_vis[0], idx_vis[N_INIT - 1]
    d = (z[b] - z[a]) / (frames[b] - frames[a])
    x = np.array([z[b, 0], z[b, 1], np.hypot(*d), np.arctan2(d[1], d[0]), 0.0])
    P = np.diag([SIGMA_OBS ** 2, SIGMA_OBS ** 2, 1.0, 0.1, 0.01])

    est = np.zeros((T, 2))
    est[:b + 1] = z[:b + 1]                      # quadros de inicialização
    for t in range(b + 1, T):
        x, P = predicao(x, P)                    # sempre
        if visivel[t]:                           # correção só com observação real
            x, P = correcao(x, P, z[t])
        est[t] = x[:2]
    return est


if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else "trajetoria_real.csv"
    saida = sys.argv[2] if len(sys.argv) > 2 else "previsoes_ekf.csv"

    df = pd.read_csv(entrada).sort_values("frame").reset_index(drop=True)
    frames = df["frame"].to_numpy()
    visivel = df["visivel"].to_numpy().astype(bool)
    z = df[["x_real", "y_real"]].to_numpy(float)
    z[~visivel] = np.nan                         # garante: oculto => sem observação

    est = rodar_ekf(frames, z, visivel)

    out = pd.DataFrame({"frame": frames,
                        "x_ekf": est[:, 0].round(2),
                        "y_ekf": est[:, 1].round(2),
                        "oclusao": (~visivel).astype(int)})
    out.to_csv(saida, index=False)

    # gráfico da trajetória (a verdade só aparece aqui, para visualização)
    ver = df[["x_real", "y_real"]].to_numpy(float)
    ocl = ~visivel
    plt.figure(figsize=(7, 5))
    plt.plot(ver[:, 0], ver[:, 1], "k", lw=1, label="Verdadeiro")
    plt.plot(est[:, 0], est[:, 1], "b--", label="EKF")
    plt.plot(est[ocl, 0], est[ocl, 1], "r.", ms=4, label="EKF na oclusão")
    plt.gca().invert_yaxis()                     # y de imagem cresce para baixo
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    plt.title("Trajetória estimada pelo EKF")
    plt.tight_layout(); plt.savefig("trajetoria_ekf.png", dpi=150)

    print(f"Salvo: {saida} e trajetoria_ekf.png")
    print(f"Quadros em oclusão: {int(ocl.sum())}")
