"""
Padrão COCO-17 compartilhado por todos os métodos (EKF, LSTM, GRU, Transformer, T-GNN).

Formato único de pose (arquivo .npz):
    keypoints   (T, 17, 2)  coordenadas x, y em pixels (ordem COCO, ver NOMES)
    visibility  (T, 17)     1 = junta detectada, 0 = junta ocluída
    fps, dimensions (largura, altura)
    keypoints_verdade (T, 17, 2)  opcional: posição real (só existe no vídeo sintético)

Gerado por: teste_video_real/processar_pose.py (vídeo real, YOLOv8-pose)
            modelagem de oclusão/código/rastreamento_oclusao.py (vídeo sintético)
"""
import numpy as np

NOMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle"]

ARESTAS = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
           (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

# Grupos de juntas que uma oclusão real tende a esconder (usado na oclusão artificial)
GRUPOS = {"braco_d": [6, 8, 10], "braco_e": [5, 7, 9],
          "perna_d": [12, 14, 16], "perna_e": [11, 13, 15], "cabeca": [0, 1, 2, 3, 4]}


def carregar_pose(caminho):
    """Lê o .npz e valida o padrão COCO-17. Retorna dict com kpts, vis, verdade, W, H, fps."""
    d = np.load(caminho)
    kpts = d["keypoints"].astype(float)
    vis = d["visibility"].astype(int)
    if kpts.ndim != 3 or kpts.shape[1:] != (17, 2) or vis.shape != kpts.shape[:2]:
        raise ValueError(f"{caminho}: esperado keypoints (T,17,2) e visibility (T,17) no padrão COCO-17; "
                         f"recebido {kpts.shape} e {vis.shape}.")
    W, H = (int(v) for v in d["dimensions"])
    return {"kpts": kpts, "vis": vis, "W": W, "H": H, "fps": int(d["fps"]),
            "verdade": d["keypoints_verdade"].astype(float) if "keypoints_verdade" in d.files else None}


def matriz_adjacencia():
    """Adjacência normalizada do grafo COCO-17 (com auto-conexões), (17,17) numpy."""
    A = np.eye(17)
    for i, j in ARESTAS:
        A[i, j] = A[j, i] = 1.0
    D = np.diag(A.sum(axis=1) ** -0.5)
    return D @ A @ D


def mascarar_oclusao(vis, seed=0, n_blocos=8, duracao=15, historico=12):
    """Oclusão artificial: esconde grupos de juntas que estavam visíveis, por `duracao` quadros.

    Como a posição das juntas escondidas é conhecida (detecção original), dá para medir o erro em pixels
    de cada método mesmo num vídeo real sem gabarito. Cada bloco exige `historico` quadros visíveis antes.
    Retorna (vis_art, mascara): mascara (T,17) é True onde a junta foi escondida artificialmente.
    """
    T = len(vis)
    if T < historico + duracao + 1:
        raise ValueError("Vídeo curto demais para a oclusão artificial.")
    rng = np.random.default_rng(seed)
    grupos = list(GRUPOS.values())
    vis_art = vis.copy()
    blocos = 0
    for _ in range(n_blocos * 50):
        if blocos == n_blocos:
            break
        g = grupos[rng.integers(len(grupos))]
        ini = int(rng.integers(historico, T - duracao))
        if (vis_art[ini - historico:ini + duracao][:, g] == 1).all():
            vis_art[ini:ini + duracao, g] = 0
            blocos += 1
    if blocos == 0:
        raise ValueError("Não encontrei trechos visíveis suficientes para a oclusão artificial.")
    return vis_art, (vis == 1) & (vis_art == 0)
