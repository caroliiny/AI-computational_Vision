"""
Treina LSTM, GRU, Transformer e T-GNN para prever a pose até 20 quadros à frente.
(O EKF não tem treino: usa o modelo cinemático fixo de EKF/ekf.py.)

  entrada do treino : pose detectada no vídeo COM oclusão (10 quadros até t0, juntas ocultas repetem o
                      último valor observado — ver protocolo.preencher_causal)
  alvo do treino    : pose do vídeo ORIGINAL em t0+1 ... t0+20 (só nos quadros de treino; o trecho de
                      teste nunca é visto)
  perda             : MSE sobre os 20 passos gerados recursivamente (a previsão realimenta a janela),
                      ignorando juntas que a referência não detectou.

As quatro redes usam as arquiteturas de modelos_pose.py, a mesma semente, épocas e dados.

Uso: python treinar_previsao.py pasta_de_trabalho [--epocas 100] [--frac-treino 0.6] [--semente 0]
Grava <pasta>/modelos/<metodo>.pt e <pasta>/treino.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from modelos_pose import ESCALA, MODELOS, SEQ_LEN, prever_quadro
from protocolo import H_MAX, amostras_treino, carregar_dados


def rollout(modelo, janela, passos=H_MAX):
    """Previsão recursiva: (B,10,34) -> (B,passos,34); cada previsão entra na janela seguinte."""
    prevs = []
    for _ in range(passos):
        p = prever_quadro(modelo, janela)
        prevs.append(p)
        janela = torch.cat([janela[:, 1:], p[:, None]], dim=1)
    return torch.stack(prevs, dim=1)


def treinar(modelo, ins, alvos, masc, epocas, lr=3e-3, batch=64, semente=0):
    gerador = torch.Generator().manual_seed(semente)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    n = len(ins)
    modelo.train()
    for ep in range(epocas):
        total = 0.0
        for idx in torch.randperm(n, generator=gerador).split(batch):
            otimizador.zero_grad()
            erro = (rollout(modelo, ins[idx]) - alvos[idx]) * ESCALA
            perda = (erro ** 2 * masc[idx]).sum() / masc[idx].sum().clamp(min=1)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            total += perda.item() * len(idx)
        if (ep + 1) % max(1, epocas // 5) == 0 or ep == 0:
            print(f"    época {ep + 1:>4}/{epocas}  perda {total / n:.4f}")
    return modelo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta")
    ap.add_argument("--epocas", type=int, default=100)
    ap.add_argument("--frac-treino", type=float, default=0.6, help="fração inicial do vídeo usada no treino")
    ap.add_argument("--semente", type=int, default=0)
    a = ap.parse_args()

    d = carregar_dados(a.pasta, a.frac_treino)
    ins, alvos, masc = amostras_treino(d)
    print(f"Treino: quadros 0-{d['corte'] - 1} ({len(ins)} janelas de {SEQ_LEN} quadros -> {H_MAX} à frente); "
          f"teste: quadros {d['corte']}-{d['T'] - 1}")
    ins, alvos, masc = (torch.tensor(v, dtype=torch.float32) for v in (ins, alvos, masc))

    pasta_modelos = Path(a.pasta) / "modelos"
    pasta_modelos.mkdir(exist_ok=True)
    for nome, classe in MODELOS.items():
        print(f"  {nome}")
        torch.manual_seed(a.semente)                       # mesma inicialização/condições para todos
        modelo = treinar(classe(), ins, alvos, masc, a.epocas, semente=a.semente)
        torch.save(modelo.state_dict(), pasta_modelos / f"{nome}.pt")

    with open(Path(a.pasta) / "treino.json", "w", encoding="utf-8") as f:
        json.dump(dict(epocas=a.epocas, frac_treino=a.frac_treino, semente=a.semente, corte=d["corte"],
                       n_janelas=len(ins), modelos=list(MODELOS)), f, indent=2)
    print(f"Modelos salvos em {pasta_modelos}")
