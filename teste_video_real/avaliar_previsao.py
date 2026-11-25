"""
Avalia EKF, LSTM, GRU, Transformer e T-GNN (mais a linha de base "Último valor") na previsão da pose
durante oclusões do vídeo, a 5, 10 e 20 quadros, contra a pose do vídeo ORIGINAL.

Todos os métodos usam os mesmos eventos de teste (protocolo.eventos_teste) e a mesma entrada: a pose
detectada no vídeo com oclusão até t0 (último quadro com a junta visível). Nenhum quadro posterior a t0 é
usado para prever t0+1 ... t0+20. O EKF roda causalmente sobre a série e, dentro da oclusão, só faz a etapa
de predição; suas previsões só são lidas enquanto a junta continua oculta.

Uso: python avaliar_previsao.py pasta_de_trabalho
Grava em <pasta>: metricas_horizonte.csv, metricas_categoria.csv, metricas_cenario.csv, erros_eventos.csv,
                  eventos.json, previsoes.npz (usado por visualizar_previsao.py)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from modelos_pose import MODELOS, SEQ_LEN, normalizar  # deve vir primeiro: ajusta o sys.path (coco17, EKF)
from coco17 import NOMES
from ekf import rodar_ekf_pose
from protocolo import H_MAX, carregar_dados, duracao_oclusao, eventos_teste, pares_avaliaveis
from treinar_previsao import rollout

METODOS = ["Último valor", "EKF", *MODELOS]


def prever_todos(d, origens, pasta):
    """{método: (n_origens, 20, 17, 2)} em pixels; NaN onde o método não tem previsão válida (EKF após a
    junta reaparecer)."""
    n = len(origens)
    saidas = {"Último valor": np.repeat(d["X"][origens][:, None], H_MAX, axis=1)}

    ekf = rodar_ekf_pose(d["obs"], d["obs_vis"])
    prev = np.full((n, H_MAX, 17, 2), np.nan)
    for i, t0 in enumerate(origens):
        for j in range(17):
            k = min(duracao_oclusao(d["obs_vis"], t0, j), H_MAX)
            if k:
                prev[i, :k, j] = ekf[t0 + 1:t0 + 1 + k, j]
    saidas["EKF"] = prev

    Xn = normalizar(d["X"], d["W"], d["H"])
    janelas = torch.tensor(np.array([Xn[t0 - SEQ_LEN + 1:t0 + 1] for t0 in origens]), dtype=torch.float32)
    escala = np.array([d["W"], d["H"]])
    for nome, classe in MODELOS.items():
        modelo = classe()
        modelo.load_state_dict(torch.load(Path(pasta) / "modelos" / f"{nome}.pt"))
        modelo.eval()
        with torch.no_grad():
            p = rollout(modelo, janelas).numpy()
        saidas[nome] = p.reshape(n, H_MAX, 17, 2) * escala
    return saidas


def resumir(df, chaves):
    g = df.groupby(chaves)["erro_px"]
    r = pd.DataFrame({"erro_medio_px": g.mean(), "rmse_px": g.apply(lambda s: np.sqrt((s ** 2).mean())),
                      "mediana_px": g.median(), "n": g.size()}).round(2).reset_index()
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta")
    a = ap.parse_args()
    pasta = Path(a.pasta)

    with open(pasta / "treino.json", encoding="utf-8") as f:
        treino = json.load(f)
    d = carregar_dados(pasta, treino["frac_treino"])
    assert d["corte"] == treino["corte"], "ponto de corte treino/teste mudou desde o treino"

    eventos = eventos_teste(d)
    pares = list(pares_avaliaveis(d, eventos))
    if not pares:
        raise SystemExit("Nenhuma oclusão avaliável no trecho de teste (vídeo curto ou sem oclusões longas "
                         "o bastante). Use um vídeo mais longo ou --frac-treino menor no treino.")
    origens = sorted({e["t0"] for e, _ in pares})
    indice = {t0: i for i, t0 in enumerate(origens)}
    print(f"Teste: quadros {d['corte']}-{d['T'] - 1}; {len(origens)} origens, {len(pares)} previsões por método")

    saidas = prever_todos(d, origens, pasta)

    linhas = []
    for nome in METODOS:
        for e, h in pares:
            pred = saidas[nome][indice[e["t0"]], h - 1, e["junta"]]
            ref = d["ref"][e["t0"] + h, e["junta"]]
            linhas.append(dict(metodo=nome, horizonte=h, categoria=e["categoria"], cenario=e["cenario"],
                               t0=e["t0"], junta=NOMES[e["junta"]], erro_px=float(np.linalg.norm(pred - ref))))
    df = pd.DataFrame(linhas)
    if df["erro_px"].isna().any():
        raise RuntimeError("Há previsões ausentes (NaN) na avaliação; verifique o EKF/protocolo.")

    df.round(2).to_csv(pasta / "erros_eventos.csv", index=False)
    por_h = resumir(df, ["metodo", "horizonte"])
    por_h["metodo"] = pd.Categorical(por_h["metodo"], METODOS)
    por_h.sort_values(["horizonte", "metodo"]).to_csv(pasta / "metricas_horizonte.csv", index=False)
    resumir(df, ["categoria", "metodo", "horizonte"]).to_csv(pasta / "metricas_categoria.csv", index=False)
    resumir(df, ["cenario", "metodo", "horizonte"]).to_csv(pasta / "metricas_cenario.csv", index=False)

    with open(pasta / "eventos.json", "w", encoding="utf-8") as f:
        json.dump(dict(origens=origens, eventos=[e for e in eventos if e["t0"] in indice]), f, indent=1, ensure_ascii=False)
    np.savez(pasta / "previsoes.npz", origens=np.array(origens), metodos=np.array(METODOS),
             previsoes=np.stack([saidas[m] for m in METODOS]))

    print("\nErro médio em pixels (menor = melhor) — quadros à frente x método:")
    print(por_h.pivot(index="metodo", columns="horizonte", values="erro_medio_px").loc[METODOS]
          .to_string(float_format=lambda v: f"{v:7.2f}"))
    print(f"\nPrevisões avaliadas por horizonte: "
          f"{df[df.metodo == METODOS[0]].groupby('horizonte').size().to_dict()}")
    print(f"Resultados em {pasta}")
