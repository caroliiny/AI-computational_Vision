import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Carregar dados
data = np.load('pose_data.npz')
kpts = data['keypoints']      # (N_frames, 17, 2)
vis = data['visibility']       # (N_frames, 17)
N_frames, J, C = kpts.shape
W, H = data['dimensions']

# Normalização de coordenadas (0 a 1)
kpts_norm = kpts.copy()
kpts_norm[:, :, 0] /= W
kpts_norm[:, :, 1] /= H
X_flat = kpts_norm.reshape(N_frames, -1) # Shape: (N_frames, 34)

# --- Definição das Arquiteturas ---

class PoseLSTM(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=64):
        super(PoseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

class PoseGRU(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=64):
        super(PoseGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out)

# --- Preparação do Conjunto de Treino (Apenas quadros 100% visíveis) ---

SEQ_LEN = 10
train_inputs, train_targets = [], []

for t in range(SEQ_LEN, N_frames):
    # Usa para treino apenas sequências onde todos os quadros e juntas estavam visíveis
    if np.all(vis[t-SEQ_LEN:t+1] == 1):
        train_inputs.append(X_flat[t-SEQ_LEN:t])
        train_targets.append(X_flat[t])

X_train = torch.tensor(np.array(train_inputs), dtype=torch.float32)
Y_train = torch.tensor(np.array(train_targets), dtype=torch.float32)

print(f"Amostras de treino extraídas de zonas sem oclusão: {len(X_train)}")

# --- Função de Treino ---

def treinar_modelo(model, epochs=60, lr=0.005):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs[:, -1, :], Y_train)
        loss.backward()
        optimizer.step()
    return model

print("A treinar modelo LSTM...")
model_lstm = treinar_modelo(PoseLSTM())

print("A treinar modelo GRU...")
model_gru = treinar_modelo(PoseGRU())

# --- Inferência com Preenchimento de Oclusão ---

def prever_sequencia(model, kpts_seq, vis_seq):
    X_pred = np.copy(X_flat)
    model.eval()
    
    with torch.no_grad():
        for t in range(SEQ_LEN, N_frames):
            if np.any(vis_seq[t] == 0): # Se houver juntas ocluídas no quadro
                seq_in = torch.tensor(X_pred[t-SEQ_LEN:t], dtype=torch.float32).unsqueeze(0)
                pred_out = model(seq_in)
                pred_frame = pred_out[0, -1].numpy()
                
                # Substitui apenas as coordenadas das juntas ocluídas
                mask_occluded = np.repeat(vis_seq[t] == 0, 2)
                X_pred[t, mask_occluded] = pred_frame[mask_occluded]
                
    X_pred = X_pred.reshape(N_frames, J, C)
    X_pred[:, :, 0] *= W
    X_pred[:, :, 1] *= H
    return X_pred

kpts_lstm = prever_sequencia(model_lstm, kpts_norm, vis)
kpts_gru = prever_sequencia(model_gru, kpts_norm, vis)

# --- Métricas e Gráfico ---

def calc_jitter(kpts_arr):
    diff = np.diff(kpts_arr, axis=0)
    return np.mean(np.var(diff, axis=0))

j_yolo = calc_jitter(kpts)
j_lstm = calc_jitter(kpts_lstm)
j_gru = calc_jitter(kpts_gru)

print("\n================ COMPARATIVO APÓS TREINO ================")
print(f"Deteção Base (YOLOv8):   {j_yolo:.2f}")
print(f"Predição Temporal LSTM:  {j_lstm:.2f}")
print(f"Predição Temporal GRU:   {j_gru:.2f}")
print("=========================================================\n")

plt.figure(figsize=(12, 5))
plt.plot(kpts[:, 10, 0], label='YOLOv8 Base', color='gray', linestyle='--', alpha=0.6)
plt.plot(kpts_lstm[:, 10, 0], label='LSTM (Treinada)', color='green')
plt.plot(kpts_gru[:, 10, 0], label='GRU (Treinada)', color='orange')
plt.title('Comparativo de Trajetória: LSTM vs GRU (Treinadas) Sob Oclusão')
plt.xlabel('Quadro (Frame)')
plt.ylabel('Posição X (Pixels)')
plt.legend()
plt.grid(True)
plt.savefig('comparativo_redes_temporais_treinadas.png')
plt.show()