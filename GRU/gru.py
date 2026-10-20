"""
GRU para estimar a posição do alvo durante a oclusão (Pessoa 2).

Entrada : CSV com  frame,x_real,y_real,visivel
          (sintético: gerado pela Clara; real: extraído de um detector de pose)
Saída   : previsoes_gru.csv  (frame,x_gru,y_gru,oclusao)
          trajetoria_gru.png

Protocolo (igual ao do LSTM, para a comparação ser justa):
  - treina SÓ com os quadros visíveis anteriores à primeira oclusão;
  - dentro da oclusão não há observação: a GRU prevê de forma recursiva,
    realimentando a própria previsão como entrada.

Formulação: a rede prevê o DESLOCAMENTO entre quadros como a velocidade média da
janela recente + uma CORREÇÃO aprendida (limitada por RES_MAX). Se a rede não souber
o que fazer, a previsão vira "manter a velocidade recente".

Estabilidade (com poucos dados a previsão recursiva é muito sensível à inicialização):
  - ensemble de N_MODELOS GRUs com sementes diferentes (a previsão é a média);
  - parada antecipada (early stopping) com validação nos últimos 20% das janelas;
  - GRU pequena e regularização L2.

Também calcula uma linha de base (velocidade constante) para contextualizar o erro.

Uso: python gru.py [entrada.csv] [previsoes_gru.csv]
Ex.: python gru.py ../LSTM_P3/trajetoria_teste.csv previsoes_gru_teste.csv
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- Parâmetros ----------------
JANELA = 5        # nº de deslocamentos passados usados para prever o próximo
UNIDADES = 16     # neurônios da camada GRU
EPOCAS = 200      # máximo; o early stopping costuma parar antes
N_MODELOS = 5     # tamanho do ensemble
RES_MAX = 0.5     # limite da correção (em desvios-padrão do deslocamento)
SEMENTE = 0


def criar_sequencias(d, janela):
    X = np.array([d[i:i + janela] for i in range(len(d) - janela)], dtype=np.float32)
    y = np.array([d[i + janela] for i in range(len(d) - janela)], dtype=np.float32)
    return X, y


def criar_modelo(janela):
    modelo = Sequential([
        Input(shape=(janela, 2)),
        GRU(UNIDADES, kernel_regularizer=l2(1e-3)),
        Dense(8, activation="relu"),
        Dense(2),
    ])
    modelo.compile(optimizer="adam", loss="mse")
    return modelo


def treinar(historico):
    """historico: (N,2) posições visíveis consecutivas. Retorna ensemble, mu, sd."""
    d = np.diff(historico, axis=0)                 # deslocamento por quadro
    mu = d.mean(axis=0)
    sd = d.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    dn = ((d - mu) / sd).astype(np.float32)

    X, y = criar_sequencias(dn, JANELA)
    if len(X) == 0:
        raise ValueError("Histórico insuficiente para treinar a GRU.")
    y = y - X.mean(axis=1)                         # alvo = variação em relação à velocidade média da janela

    usar_val = len(X) >= 20                        # validação só se houver janelas suficientes
    modelos = []
    for k in range(N_MODELOS):
        tf.keras.utils.set_random_seed(SEMENTE + k)
        modelo = criar_modelo(JANELA)
        parar = EarlyStopping(monitor="val_loss" if usar_val else "loss", patience=20,
                              restore_best_weights=True)
        modelo.fit(X, y, epochs=EPOCAS, batch_size=min(16, len(X)),
                   validation_split=0.2 if usar_val else 0.0,
                   callbacks=[parar], shuffle=False, verbose=0)
        modelos.append(modelo)
    return modelos, mu, sd


def prever(modelos, mu, sd, historico, n):
    """Prevê n posições após 'historico' (recursivo, sem novas observações)."""
    d = (np.diff(historico[-(JANELA + 1):], axis=0) - mu) / sd
    janela = list(d.astype(np.float32))
    pos = historico[-1].astype(float).copy()
    saida = []
    for _ in range(n):
        ult = np.array(janela[-JANELA:], dtype=np.float32)
        res = np.mean([m(ult[None], training=False).numpy()[0] for m in modelos], axis=0)
        res = np.clip(res, -RES_MAX, RES_MAX)       # evita que a correção "dispare"
        dn = ult.mean(axis=0) + res                 # velocidade recente + correção aprendida
        janela.append(dn.astype(np.float32))        # realimenta a previsão
        pos = pos + dn * sd + mu                    # volta à escala original e integra
        saida.append(pos.copy())
    return np.array(saida)


def linha_base(historico, n):
    """Velocidade constante: repete o deslocamento médio dos últimos quadros."""
    v = np.diff(historico[-(JANELA + 1):], axis=0).mean(axis=0)
    return historico[-1] + v * np.arange(1, n + 1)[:, None]


def erros(ref, prev):
    dif = ref - prev
    dist = np.sqrt((dif ** 2).sum(axis=1))
    return dist.mean(), np.sqrt((dif[:, 0] ** 2).mean()), np.sqrt((dif[:, 1] ** 2).mean())


if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else "trajetoria_real.csv"
    saida = sys.argv[2] if len(sys.argv) > 2 else "previsoes_gru.csv"

    df = pd.read_csv(entrada).sort_values("frame").reset_index(drop=True)
    frames = df["frame"].to_numpy()
    ref = df[["x_real", "y_real"]].to_numpy(float)            # referência (pode ter NaN)
    visivel = df["visivel"].to_numpy().astype(bool) & ~np.isnan(ref).any(axis=1)
    ocl = ~visivel
    T = len(frames)

    if not ocl.any():
        raise ValueError("Não há quadros em oclusão neste CSV.")
    primeiro = int(np.argmax(ocl))
    if primeiro <= JANELA + 1:
        raise ValueError("Poucos quadros visíveis antes da oclusão para treinar.")

    z = ref.copy()
    z[ocl] = np.nan                                           # oculto => sem observação

    modelos, mu, sd = treinar(z[:primeiro])

    est = z.copy()
    base = z.copy()
    i = primeiro
    while i < T:                                              # trata cada bloco de oclusão
        if ocl[i]:
            j = i
            while j < T and ocl[j]:
                j += 1
            est[i:j] = prever(modelos, mu, sd, est[:i], j - i)
            base[i:j] = linha_base(base[:i], j - i)
            i = j
        else:
            i += 1

    pd.DataFrame({"frame": frames,
                  "x_gru": est[:, 0].round(2),
                  "y_gru": est[:, 1].round(2),
                  "oclusao": ocl.astype(int)}).to_csv(saida, index=False)

    # Erro apenas nos quadros ocluídos que têm referência
    m = ocl & ~np.isnan(ref).any(axis=1)
    if m.any():
        em, rx, ry = erros(ref[m], est[m])
        print(f"GRU                  -> Erro médio: {em:.2f} | RMSE X: {rx:.2f} | RMSE Y: {ry:.2f}")
        em, rx, ry = erros(ref[m], base[m])
        print(f"Velocidade constante -> Erro médio: {em:.2f} | RMSE X: {rx:.2f} | RMSE Y: {ry:.2f}")
    print(f"Salvo: {saida}  |  quadros em oclusão: {int(ocl.sum())}  |  ensemble: {N_MODELOS} GRUs")

    plt.figure(figsize=(7, 5))
    plt.plot(ref[:, 0], ref[:, 1], "k", lw=1, label="Referência")
    plt.plot(est[:, 0], est[:, 1], "g--", label="GRU")
    plt.plot(est[ocl, 0], est[ocl, 1], "r.", ms=4, label="GRU na oclusão")
    plt.gca().invert_yaxis()
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    plt.title("Trajetória estimada pela GRU")
    plt.tight_layout(); plt.savefig("trajetoria_gru.png", dpi=150)