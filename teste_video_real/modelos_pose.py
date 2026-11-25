"""
Métodos de preenchimento de oclusão sobre o esqueleto COCO-17, compartilhados pelos scripts desta pasta:
EKF (real, por junta), LSTM, GRU, Transformer temporal e T-GNN.  Todos recebem (T,17,2) + vis (T,17).
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / "EKF")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from coco17 import carregar_pose, matriz_adjacencia
from ekf import rodar_ekf_pose

SEQ_LEN = 10
ESCALA = 50.0          # a rede aprende o deslocamento entre quadros (valores pequenos) multiplicado por isso


# ---------------------------------------------------------------- arquiteturas (entrada/saída: 34 = 17 x 2)
class PoseLSTM(nn.Module):
    def __init__(self, dim=34, oculto=64):
        super().__init__()
        self.rnn = nn.LSTM(dim, oculto, batch_first=True, num_layers=2)
        self.fc = nn.Linear(oculto, dim)

    def forward(self, x):
        return self.fc(self.rnn(x)[0])


class PoseGRU(nn.Module):
    def __init__(self, dim=34, oculto=64):
        super().__init__()
        self.rnn = nn.GRU(dim, oculto, batch_first=True, num_layers=2)
        self.fc = nn.Linear(oculto, dim)

    def forward(self, x):
        return self.fc(self.rnn(x)[0])


class PoseTransformer(nn.Module):
    def __init__(self, dim=34, d_model=64, nhead=4, camadas=2):
        super().__init__()
        self.embedding = nn.Linear(dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))      # ordem temporal da janela
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True), num_layers=camadas)
        self.fc = nn.Linear(d_model, dim)

    def forward(self, x):
        return self.fc(self.encoder(self.embedding(x) + self.pos[:, :x.shape[1]]))


class PoseTGNN(nn.Module):
    """Convolução em grafo sobre as 17 juntas (arestas COCO) + GRU temporal."""

    def __init__(self, oculto=32):
        super().__init__()
        self.register_buffer("A", torch.tensor(matriz_adjacencia(), dtype=torch.float32))
        self.g1 = nn.Linear(2, oculto)
        self.g2 = nn.Linear(oculto, oculto)
        self.gru = nn.GRU(17 * oculto, 64, batch_first=True)
        self.fc = nn.Linear(64, 34)

    def forward(self, x):
        B, S, _ = x.shape
        h = x.reshape(B, S, 17, 2)
        h = torch.relu(self.g1(torch.matmul(self.A, h)))
        h = torch.relu(self.g2(torch.matmul(self.A, h)))
        return self.fc(self.gru(h.reshape(B, S, -1))[0])


MODELOS = {"LSTM": PoseLSTM, "GRU": PoseGRU, "Transformer": PoseTransformer, "T-GNN": PoseTGNN}


# ---------------------------------------------------------------- treino e inferência
def normalizar(kpts, W, H):
    return (kpts / np.array([W, H])).reshape(len(kpts), -1)            # (T,34) em [0,1]


def desnormalizar(X, W, H):
    return X.reshape(len(X), 17, 2) * np.array([W, H])


def prever_quadro(modelo, janela):
    """Próximo quadro = último quadro da janela + deslocamento previsto pela rede."""
    return janela[:, -1] + modelo(janela)[:, -1] / ESCALA


def treinar(modelo, X, vis, epocas=150, lr=3e-3, semente=0):
    """Treina só com janelas em que as 17 juntas estão visíveis (sem usar nenhuma posição ocluída)."""
    ins, alvos = [], []
    for t in range(SEQ_LEN, len(X)):
        if (vis[t - SEQ_LEN:t + 1] == 1).all():
            ins.append(X[t - SEQ_LEN:t])
            alvos.append(X[t])
    if not ins:
        raise ValueError("Nenhuma janela com as 17 juntas visíveis para treinar.")
    entrada = torch.tensor(np.array(ins), dtype=torch.float32)
    alvo = torch.tensor(np.array(alvos), dtype=torch.float32)
    torch.manual_seed(semente)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    modelo.train()
    for _ in range(epocas):
        otimizador.zero_grad()
        perda = nn.functional.mse_loss(prever_quadro(modelo, entrada) * ESCALA, alvo * ESCALA)
        perda.backward()
        otimizador.step()
    print(f"  {type(modelo).__name__}: {len(ins)} janelas de treino")
    return modelo


def preencher_oclusao(modelo, X, vis):
    """Substitui só as juntas ocluídas por previsões recursivas (a previsão alimenta a janela seguinte)."""
    X_pred = X.copy()
    modelo.eval()
    with torch.no_grad():
        for t in range(SEQ_LEN, len(X)):
            oculta = vis[t] == 0
            if oculta.any():
                janela = torch.tensor(X_pred[t - SEQ_LEN:t], dtype=torch.float32)[None]
                prev = prever_quadro(modelo, janela)[0].numpy()
                X_pred[t, np.repeat(oculta, 2)] = prev[np.repeat(oculta, 2)]
    return X_pred


def manter_ultimo(kpts, vis):
    """Linha de base: a junta ocluída fica parada na última posição conhecida."""
    out = kpts.copy()
    for t in range(1, len(out)):
        oculta = vis[t] == 0
        out[t, oculta] = out[t - 1, oculta]
    return out


def rodar_metodos(kpts, vis, W, H, nomes=None, epocas=150):
    """Roda os métodos pedidos (padrão: todos). Retorna {nome: (T,17,2)} alterando só as juntas ocluídas."""
    nomes = nomes or ["Último valor", "EKF", *MODELOS]
    oculta = (vis == 0)[..., None]
    X = normalizar(kpts, W, H)
    saidas = {}
    for nome in nomes:
        print(f"Método: {nome}")
        if nome == "Último valor":
            est = manter_ultimo(kpts, vis)
        elif nome == "EKF":
            est = rodar_ekf_pose(kpts, vis)
        else:
            modelo = treinar(MODELOS[nome](), X, vis, epocas=epocas)
            est = desnormalizar(preencher_oclusao(modelo, X, vis), W, H)
        saidas[nome] = np.where(oculta, est, kpts)
    return saidas


# ---------------------------------------------------------------- métricas e gráfico
def erro_px(pred, verdade, mascara):
    """(erro médio, RMSE) em pixels nos pontos de `mascara` (T,17)."""
    if not mascara.any():
        return float("nan"), float("nan")
    d = np.linalg.norm(pred - verdade, axis=-1)[mascara]
    return d.mean(), np.sqrt((d ** 2).mean())


def jitter(kpts):
    return np.mean(np.var(np.diff(kpts, axis=0), axis=0))


def comparar_jitter(nomes, png, entrada="pose_data.npz"):
    """Vídeo real sem gabarito: compara o jitter de cada método com o da detecção do YOLO."""
    p = carregar_pose(entrada)
    saidas = rodar_metodos(p["kpts"], p["vis"], p["W"], p["H"], nomes)
    print("\n=========== JITTERING (vídeo real, oclusão natural) ===========")
    print(f"Detecção base (YOLOv8): {jitter(p['kpts']):.2f}")
    for nome, est in saidas.items():
        print(f"{nome:<22}: {jitter(est):.2f}")
    print(f"Taxa de oclusão: {(1 - p['vis'].mean()) * 100:.2f}%")
    plt.figure(figsize=(12, 5))
    plt.plot(p["kpts"][:, 10, 0], "--", color="gray", alpha=0.6, label="YOLOv8 base")
    for nome, est in saidas.items():
        plt.plot(est[:, 10, 0], label=nome)
    plt.title("Pulso direito (COCO 10) sob oclusão")
    plt.xlabel("Quadro"); plt.ylabel("Posição X (px)"); plt.legend(); plt.grid(True)
    plt.savefig(png)
    print(f"Gráfico salvo em '{png}'.")


def executar_jitter(nomes, png_padrao):
    """Linha de comando comum aos scripts de teste: python script.py [pose.npz] [--png saida.png]."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pose", nargs="?", default="pose_data.npz", help="saída de processar_pose.py")
    ap.add_argument("--png", help=f"padrão: <pose>_{png_padrao}")
    a = ap.parse_args()
    comparar_jitter(nomes, a.png or f"{Path(a.pose).stem}_{png_padrao}", a.pose)
