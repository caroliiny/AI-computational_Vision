"""
Protocolo único de treino e avaliação da previsão de pose sob oclusão (usado por treinar_previsao.py e
avaliar_previsao.py, para que todos os métodos vejam exatamente os mesmos dados).

  ENTRADA dos métodos : somente a pose detectada no VÍDEO COM OCLUSÃO (oclusao_pose.npz), até o quadro t0.
  REFERÊNCIA          : pose do VÍDEO ORIGINAL (referencia_pose.npz); só serve de alvo no treino e de
                        gabarito na avaliação, nunca entra como entrada.
  PREVISÃO            : a partir de t0 (último quadro com a junta visível), prevê t0+1 ... t0+20 sem usar
                        nenhum quadro posterior a t0 (previsão, não interpolação).
  AVALIAÇÃO           : erro em t0+h, h em {5, 10, 20}, só em oclusões que duram >= h quadros e só em quadros
                        depois do ponto de corte treino/teste (o treino nunca vê o trecho de teste).
"""
import json
from pathlib import Path

import numpy as np

import modelos_pose  # noqa: F401  (ajusta o sys.path para coco17 e EKF; deve vir primeiro)
from coco17 import carregar_pose
from modelos_pose import SEQ_LEN, normalizar

HORIZONTES = (5, 10, 20)
H_MAX = max(HORIZONTES)
MIN_VISIVEL_EKF = 10          # o EKF (EKF/ekf.py) precisa de N_INIT quadros visíveis da junta para iniciar


def carregar_dados(pasta, frac_treino=0.6):
    """Lê referência, observação (vídeo com oclusão) e cenários; valida a sincronia entre eles."""
    pasta = Path(pasta)
    ref = carregar_pose(pasta / "referencia_pose.npz")
    obs = carregar_pose(pasta / "oclusao_pose.npz")
    if len(ref["kpts"]) != len(obs["kpts"]) or (ref["W"], ref["H"]) != (obs["W"], obs["H"]) \
            or abs(ref["fps"] - obs["fps"]) > 0.05:
        raise ValueError("Vídeo original e vídeo com oclusão não estão sincronizados "
                         "(quadros, resolução ou FPS diferentes).")
    with open(pasta / "cenarios.json", encoding="utf-8") as f:
        cenarios = json.load(f)["cenarios"]
    T = len(ref["kpts"])
    corte = int(T * frac_treino)
    if corte < SEQ_LEN + H_MAX + 1 or T - corte < SEQ_LEN + H_MAX + 1:
        raise ValueError(f"Vídeo curto demais ({T} quadros) para separar treino e teste.")
    X, valido = preencher_causal(obs["kpts"], obs["vis"])
    return dict(ref=ref["kpts"], ref_vis=ref["vis"], obs=obs["kpts"], obs_vis=obs["vis"],
                X=X, valido=valido, W=ref["W"], H=ref["H"], fps=ref["fps"], T=T,
                corte=corte, cenarios=cenarios)


def preencher_causal(kpts, vis):
    """Série de entrada das redes: juntas ocultas repetem o último valor observado (só olha o passado).

    Juntas nunca observadas até o quadro usam o centro das juntas visíveis daquele quadro. Quadros em que
    nada foi observado ainda ficam inválidos (valido = False). Retorna (X (T,17,2), valido (T,)).
    """
    T = len(kpts)
    X = np.zeros((T, 17, 2))
    valido = np.ones(T, bool)
    ultimo = np.full((17, 2), np.nan)
    for t in range(T):
        v = vis[t] == 1
        ultimo[v] = kpts[t, v]
        quadro = ultimo.copy()
        faltam = np.isnan(quadro[:, 0])
        if faltam.any():
            if v.any():
                quadro[faltam] = kpts[t, v].mean(axis=0)
            else:
                valido[t] = False
        X[t] = np.nan_to_num(quadro)
    return X, valido


def janela_valida(d, t0):
    return t0 >= SEQ_LEN - 1 and d["valido"][t0 - SEQ_LEN + 1:t0 + 1].all()


def amostras_treino(d):
    """Janelas de treino: entrada = vídeo com oclusão (10 quadros até t0); alvo = referência em t0+1..t0+20.

    Só usa t0 tais que t0+20 < corte. Retorna (entrada (N,10,34), alvo (N,20,34), mascara (N,20,34)),
    com mascara = 1 onde a referência tem a junta detectada (não há alvo confiável onde o YOLO falhou).
    """
    Xn = normalizar(d["X"], d["W"], d["H"])
    Rn = normalizar(d["ref"], d["W"], d["H"])
    Rv = np.repeat(d["ref_vis"], 2, axis=1)
    ins, alvos, masc = [], [], []
    for t0 in range(SEQ_LEN - 1, d["corte"] - H_MAX):
        if not janela_valida(d, t0):
            continue
        ins.append(Xn[t0 - SEQ_LEN + 1:t0 + 1])
        alvos.append(Rn[t0 + 1:t0 + 1 + H_MAX])
        masc.append(Rv[t0 + 1:t0 + 1 + H_MAX])
    if not ins:
        raise ValueError("Nenhuma janela de treino válida (pessoa não detectada no trecho de treino).")
    return np.array(ins), np.array(alvos), np.array(masc, dtype=float)


def duracao_oclusao(vis_obs, t0, j):
    """Quadros seguidos, a partir de t0+1, em que a junta j fica sem detecção no vídeo com oclusão."""
    n = 0
    while t0 + 1 + n < len(vis_obs) and vis_obs[t0 + 1 + n, j] == 0:
        n += 1
    return n


def cenario_do_quadro(cenarios, t):
    for c in cenarios:
        if c["ini"] <= t <= c["fim"]:
            return c
    return None


def eventos_teste(d):
    """Origens de previsão do teste: junta visível em t0 e ocluída em t0+1 (início de uma oclusão).

    Retorna lista de dicts {t0, junta, duracao, cenario, categoria}, só com t0 >= corte e janela válida.
    Os mesmos eventos valem para os cinco métodos.
    """
    eventos = []
    for t0 in range(d["corte"], d["T"] - 1):
        if not janela_valida(d, t0):
            continue
        c = cenario_do_quadro(d["cenarios"], t0 + 1)
        for j in range(17):
            visivel_antes = d["obs_vis"][:t0 + 1, j].sum()        # só o passado até t0 (iniciar o EKF)
            if d["obs_vis"][t0, j] == 1 and d["obs_vis"][t0 + 1, j] == 0 and visivel_antes >= MIN_VISIVEL_EKF:
                n = duracao_oclusao(d["obs_vis"], t0, j)
                if n >= min(HORIZONTES):
                    eventos.append(dict(t0=t0, junta=j, duracao=n,
                                        cenario=c["tipo"] if c else "natural",
                                        categoria=c["categoria"] if c else "natural"))
    return eventos


def pares_avaliaveis(d, eventos):
    """(evento, h) avaliáveis: oclusão dura >= h quadros e a referência existe em t0+h."""
    for e in eventos:
        for h in HORIZONTES:
            if e["duracao"] >= h and d["ref_vis"][e["t0"] + h, e["junta"]] == 1:
                yield e, h
