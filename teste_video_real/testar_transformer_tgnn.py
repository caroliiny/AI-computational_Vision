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

# Normalização de coordenadas
kpts_norm = kpts.copy()
kpts_norm[:, :, 0] /= W
kpts_norm[:, :, 1] /= H
X_flat = kpts_norm.reshape(N_frames, -1) # Shape: (N_frames, 34)

# --- Topologia do Grafo do Corpo Humano (17 Juntas - COCO) ---
COCO_EDGES = [
    (0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10),
    (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)
]

def construir_matriz_adjacencia():
    A = np.eye(17) # Auto-conexões
    for i, j in COCO_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Normalização por grau
    D = np.diag(np.sum(A, axis=1)**(-0.5))
    A_norm = D @ A @ D
    return torch.tensor(A_norm, dtype=torch.float32)

A_adj = construir_matriz_adjacencia()

# --- Arquitetura 1: Transformer Temporal ---

class PoseTransformer(nn.Module):
    def __init__(self, input_dim=34, d_model=64, nhead=4, num_layers=2):
        super(PoseTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x_emb = self.embedding(x)
        out = self.transformer(x_emb)
        return self.fc(out)

# --- Arquitetura 2: T-GNN (Temporal Graph Neural Network) ---

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, A):
        # x shape: (batch, seq_len, 17, in_features)
        x_g = torch.matmul(A, x)
        return torch.relu(self.fc(x_g))

class PoseTGNN(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=32):
        super(PoseTGNN, self).__init__()
        self.gconv1 = GraphConv(in_channels, hidden_dim)
        self.gconv2 = GraphConv(hidden_dim, hidden_dim)
        self.temporal_gru = nn.GRU(17 * hidden_dim, 64, batch_first=True)
        self.fc_out = nn.Linear(64, 34)

    def forward(self, x, A):
        # x input shape: (batch, seq_len, 34)
        B, S, _ = x.shape
        x_graph = x.view(B, S, 17, 2)
        
        # Convolução Espacial em Grafo
        h1 = self.gconv1(x_graph, A)
        h2 = self.gconv2(h1, A)
        
        # Flatten para Camada Temporal
        h_flat = h2.view(B, S, -1)
        out_gru, _ = self.temporal_gru(h_flat)
        return self.fc_out(out_gru)

# --- Preparação de Treino ---

SEQ_LEN = 10
train_inputs, train_targets = [], []

for t in range(SEQ_LEN, N_frames):
    if np.all(vis[t-SEQ_LEN:t+1] == 1):
        train_inputs.append(X_flat[t-SEQ_LEN:t])
        train_targets.append(X_flat[t])

X_train = torch.tensor(np.array(train_inputs), dtype=torch.float32)
Y_train = torch.tensor(np.array(train_targets), dtype=torch.float32)

# --- Treino dos Modelos ---

def treinar(model, is_tgnn=False, epochs=60, lr=0.003):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        if is_tgnn:
            outputs = model(X_train, A_adj)
        else:
            outputs = model(X_train)
            
        loss = criterion(outputs[:, -1, :], Y_train)
        loss.backward()
        optimizer.step()
    return model

print("A treinar Transformer Temporal...")
model_trans = treinar(PoseTransformer())

print("A treinar T-GNN (Espaço-Temporal)...")
model_tgnn = treinar(PoseTGNN(), is_tgnn=True)

# --- Inferência e Preenchimento de Oclusão ---

def prever_sequencia(model, is_tgnn=False):
    X_pred = np.copy(X_flat)
    model.eval()
    
    with torch.no_grad():
        for t in range(SEQ_LEN, N_frames):
            if np.any(vis[t] == 0):
                seq_in = torch.tensor(X_pred[t-SEQ_LEN:t], dtype=torch.float32).unsqueeze(0)
                if is_tgnn:
                    pred_out = model(seq_in, A_adj)
                else:
                    pred_out = model(seq_in)
                    
                pred_frame = pred_out[0, -1].numpy()
                mask_occluded = np.repeat(vis[t] == 0, 2)
                X_pred[t, mask_occluded] = pred_frame[mask_occluded]
                
    X_pred = X_pred.reshape(N_frames, J, C)
    X_pred[:, :, 0] *= W
    X_pred[:, :, 1] *= H
    return X_pred

kpts_trans = prever_sequencia(model_trans, is_tgnn=False)
kpts_tgnn = prever_sequencia(model_tgnn, is_tgnn=True)

# --- Métricas e Gráficos ---

def calc_jitter(kpts_arr):
    diff = np.diff(kpts_arr, axis=0)
    return np.mean(np.var(diff, axis=0))

print("\n================ BENCHMARK FINAL DE JITTERING ================")
print(f"1. Deteção Base (YOLOv8):      {calc_jitter(kpts):.2f}")
print(f"2. Transformer Temporal:        {calc_jitter(kpts_trans):.2f}")
print(f"3. T-GNN (Espaço-Temporal):    {calc_jitter(kpts_tgnn):.2f}")
print("==============================================================\n")

plt.figure(figsize=(12, 5))
plt.plot(kpts[:, 10, 0], label='YOLOv8 Base', color='gray', linestyle='--', alpha=0.5)
plt.plot(kpts_trans[:, 10, 0], label='Transformer Temporal', color='purple')
plt.plot(kpts_tgnn[:, 10, 0], label='T-GNN', color='red')
plt.title('Comparativo Final: Transformer Temporal vs T-GNN Sob Oclusão')
plt.xlabel('Quadro (Frame)')
plt.ylabel('Posição X (Pixels)')
plt.legend()
plt.grid(True)
plt.savefig('comparativo_transformer_tgnn.png')
plt.show()