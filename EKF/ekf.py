"""
EKF para estimar a posição do alvo durante a oclusão (Pessoa 2).

Entrada : CSV com  frame,x_real,y_real,visivel  [,confianca]
          (sintético: gerado pela Clara; real: extraído de um detector de pose)
Saída   : previsoes_ekf.csv  (frame,x_ekf,y_ekf,oclusao)
          trajetoria_ekf.png
          + erro médio e RMSE (X, Y) nos quadros ocluídos, junto da linha de base
            "velocidade constante" (mesmo formato do gru.py)

Regra principal: onde visivel == 0 (ou a coordenada é NaN) o filtro SÓ faz predição.

Uso: python ekf.py [entrada.csv] [previsoes_ekf.csv]
Ex.: python ekf.py ../LSTM_P3/trajetoria_teste.csv previsoes_ekf_teste.csv
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Parâmetros (ajustáveis; ver explicacao_ekf_gru.md) ----------------
SIGMA_OBS = 1.5                              # incerteza da observação (unidade das coords)
Q_DIAG = [0.05, 0.05, 0.05, 0.005, 1e-4]     # ruído de processo por quadro: x, y, v, theta, omega
N_INIT = 10                                  # quadros visíveis para iniciar o estado
GATE = 9.21                                  # limiar (qui-quadrado, 2 gl, 99%) p/ rejeitar detecções absurdas
JANELA_BASE = 5                              # quadros usados pela linha de base (igual ao gru.py)

# Estado: [x, y, v, theta, omega]  (modelo não linear - CTRV)
H = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0]], dtype=float)


def f(x, dt):
    px, py, v, th, om = x
    return np.array([px + v * np.cos(th) * dt,
                     py + v * np.sin(th) * dt,
                     v, th + om * dt, om])


def jacobiana(x, dt):
    _, _, v, th, _ = x
    return np.array([[1, 0, np.cos(th) * dt, -v * np.sin(th) * dt, 0],
                     [0, 1, np.sin(th) * dt,  v * np.cos(th) * dt, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 1, dt],
                     [0, 0, 0, 0, 1]])


def rodar_ekf(frames, z, visivel, conf=None, sigma_obs=SIGMA_OBS, q_diag=Q_DIAG):
    """z: (T,2). Retorna estimativas (T,2) e nº de detecções rejeitadas.

    sigma_obs e q_diag permitem ajustar o filtro à escala dos dados (ex.: pixels de vídeo real).
    """
    Q0 = np.diag(q_diag)
    R0 = np.eye(2) * sigma_obs ** 2
    T = len(frames)
    idx_vis = np.where(visivel)[0]
    if len(idx_vis) < N_INIT:
        raise ValueError("Poucos quadros visíveis para iniciar o filtro.")
    a, b = idx_vis[0], idx_vis[N_INIT - 1]
    d = (z[b] - z[a]) / (frames[b] - frames[a])
    x = np.array([z[b, 0], z[b, 1], np.hypot(*d), np.arctan2(d[1], d[0]), 0.0])
    P = np.diag([sigma_obs ** 2, sigma_obs ** 2, 1.0, 0.1, 0.01])

    est = np.zeros((T, 2))
    est[:b + 1] = np.where(np.isnan(z[:b + 1]), np.nan, z[:b + 1])
    rejeitadas = 0
    rej_seq = 0                                              # rejeições seguidas (evita o filtro travar)
    for t in range(b + 1, T):
        dt = float(frames[t] - frames[t - 1])            # tolera quadros faltando
        F = jacobiana(x, dt)
        x = f(x, dt)                                      # predição (sempre)
        P = F @ P @ F.T + Q0 * dt
        if visivel[t]:
            R = R0 / max(conf[t], 0.1) if conf is not None else R0
            S = H @ P @ H.T + R
            inov = z[t] - H @ x
            reaparece = not visivel[t - 1]                    # 1º quadro após oclusão: sempre aceita
            # descarta outliers do detector, mas aceita após 3 rejeições seguidas
            if reaparece or rej_seq >= 3 or inov @ np.linalg.solve(S, inov) <= GATE:
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ inov
                P = (np.eye(5) - K @ H) @ P
                rej_seq = 0
            else:
                rejeitadas += 1
                rej_seq += 1
        est[t] = x[:2]
    return est, rejeitadas


def rodar_ekf_pose(kpts, vis, sigma_obs=3.0, q_diag=(0.5, 0.5, 0.5, 0.02, 1e-3)):
    """EKF sobre um esqueleto COCO-17: um filtro independente por junta.

    kpts: (T,17,2) em pixels; vis: (T,17) com 1 = detectada, 0 = ocluída.
    Retorna (T,17,2): a trajetória filtrada, com a predição do EKF nas juntas ocluídas.
    Juntas com poucos quadros visíveis (< N_INIT) são devolvidas sem alteração.
    """
    T, J, _ = kpts.shape
    frames = np.arange(T)
    est = kpts.astype(float).copy()
    for j in range(J):
        visivel = vis[:, j].astype(bool) & ~(kpts[:, j] == 0).all(axis=1)
        if visivel.sum() < N_INIT:
            continue
        z = kpts[:, j].astype(float).copy()
        z[~visivel] = np.nan
        est_j, _ = rodar_ekf(frames, z, visivel, sigma_obs=sigma_obs, q_diag=list(q_diag))
        ok = ~np.isnan(est_j).any(axis=1)
        est[ok, j] = est_j[ok]
    return est


def linha_base(historico, n):
    """Velocidade constante: repete o deslocamento médio dos últimos quadros."""
    v = np.diff(historico[-(JANELA_BASE + 1):], axis=0).mean(axis=0)
    return historico[-1] + v * np.arange(1, n + 1)[:, None]


def calcular_linha_base(z, ocl):
    """Aplica a linha de base em cada bloco de oclusão (quando há histórico suficiente)."""
    base = z.copy()
    T = len(z)
    i = 0
    while i < T:
        if ocl[i]:
            j = i
            while j < T and ocl[j]:
                j += 1
            if i > JANELA_BASE + 1 and not np.isnan(base[i - JANELA_BASE - 1:i]).any():
                base[i:j] = linha_base(base[:i], j - i)
            i = j
        else:
            i += 1
    return base


def erros(ref, prev):
    dif = ref - prev
    dist = np.sqrt((dif ** 2).sum(axis=1))
    return dist.mean(), np.sqrt((dif[:, 0] ** 2).mean()), np.sqrt((dif[:, 1] ** 2).mean())


if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else "trajetoria_real.csv"
    saida = sys.argv[2] if len(sys.argv) > 2 else "previsoes_ekf.csv"

    df = pd.read_csv(entrada).sort_values("frame").reset_index(drop=True)
    frames = df["frame"].to_numpy()
    z = df[["x_real", "y_real"]].to_numpy(float)
    ver = z.copy()                                        # referência (antes de ocultar)
    visivel = df["visivel"].to_numpy().astype(bool) & ~np.isnan(z).any(axis=1)
    z[~visivel] = np.nan                                  # oculto => sem observação
    conf = df["confianca"].to_numpy(float) if "confianca" in df else None

    est, rej = rodar_ekf(frames, z, visivel, conf)
    ocl = ~visivel

    pd.DataFrame({"frame": frames,
                  "x_ekf": est[:, 0].round(2),
                  "y_ekf": est[:, 1].round(2),
                  "oclusao": ocl.astype(int)}).to_csv(saida, index=False)

    # Erro apenas nos quadros ocluídos que têm referência e estimativa
    m = ocl & ~np.isnan(ver).any(axis=1) & ~np.isnan(est).any(axis=1)
    if m.any():
        em, rx, ry = erros(ver[m], est[m])
        print(f"EKF                  -> Erro médio: {em:.2f} | RMSE X: {rx:.2f} | RMSE Y: {ry:.2f}")
        base = calcular_linha_base(z, ocl)
        mb = m & ~np.isnan(base).any(axis=1)
        if mb.any():
            em, rx, ry = erros(ver[mb], base[mb])
            print(f"Velocidade constante -> Erro médio: {em:.2f} | RMSE X: {rx:.2f} | RMSE Y: {ry:.2f}")
    print(f"Salvo: {saida}  |  quadros em oclusão: {int(ocl.sum())}  |  detecções rejeitadas: {rej}")

    plt.figure(figsize=(7, 5))
    plt.plot(ver[:, 0], ver[:, 1], "k", lw=1, label="Referência")
    plt.plot(est[:, 0], est[:, 1], "b--", label="EKF")
    plt.plot(est[ocl, 0], est[ocl, 1], "r.", ms=4, label="EKF na oclusão")
    plt.gca().invert_yaxis()
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    plt.title("Trajetória estimada pelo EKF")
    plt.tight_layout(); plt.savefig("trajetoria_ekf.png", dpi=150)