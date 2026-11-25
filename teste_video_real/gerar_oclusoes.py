"""Gera o vídeo com oclusões artificiais a partir do vídeo original limpo.

Desenha obstáculos opacos (pilares verticais e muros baixos) sobre a imagem, sem alterar duração, FPS,
resolução nem a ordem dos quadros. Os obstáculos são posicionados usando a pose de REFERÊNCIA (vídeo
original) apenas para criar o cenário; os modelos nunca recebem essa pose.

Uso: python gerar_oclusoes.py video_original.mp4 pasta_de_trabalho [--semente 0] [--intervalo 25]
  lê    <pasta>/referencia_pose.npz   (saída de processar_pose.py no vídeo original)
  grava <pasta>/video_oclusao.mp4  e  <pasta>/cenarios.json

Tipos de cenário (todos aparecem em ciclos, com posição/largura/altura sorteadas):
  parcial      pilar_estreito, muro_baixo_pernas   (alguns braços/tronco/pernas)
  grande       pilar_largo, muro_alto               (região grande do corpo)
  quase_total  bloco_quase_total                    (só a cabeça pode aparecer)
  total        bloco_total                          (pessoa inteira escondida)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import modelos_pose  # noqa: F401  (ajusta o sys.path para coco17)
from coco17 import NOMES, carregar_pose

TIPOS = [("pilar_estreito", "parcial"), ("muro_baixo_pernas", "parcial"),
         ("pilar_largo", "grande"), ("muro_alto", "grande"),
         ("bloco_quase_total", "quase_total"), ("bloco_total", "total")]


def caixa_pessoa(kpts, vis, ini, fim):
    """Caixa (x0,y0,x1,y1) mediana e a união da pessoa nos quadros [ini, fim], pela pose de referência."""
    caixas = []
    for t in range(ini, fim + 1):
        v = vis[t] == 1
        if v.sum() >= 5:
            p = kpts[t, v]
            caixas.append([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
    if len(caixas) < (fim - ini + 1) // 2:
        return None, None                       # pessoa ausente/pouco detectada nesse trecho
    c = np.array(caixas)
    uniao = [c[:, 0].min(), c[:, 1].min(), c[:, 2].max(), c[:, 3].max()]
    return np.median(c, axis=0), np.array(uniao)


def retangulo(tipo, mediana, uniao, W, H, rng):
    """Retângulo do obstáculo (x0,y0,x1,y1) em pixels, com largura/altura/posição sorteadas."""
    x0, y0, x1, y1 = mediana
    bw, bh = x1 - x0, y1 - y0
    y0 = y0 - 0.08 * bh                         # a caixa das juntas não inclui o topo da cabeça
    bh = y1 - y0
    u = rng.uniform
    if tipo == "pilar_estreito":                # pilar fino: um braço ou uma faixa do tronco
        w, c = bw * u(0.15, 0.30), x0 + bw * u(0.25, 0.75)
        r = (c - w / 2, 0, c + w / 2, H)
    elif tipo == "muro_baixo_pernas":           # muro baixo: pernas
        r = (x0 - 0.3 * bw, y0 + bh * u(0.55, 0.75), x1 + 0.3 * bw, H)
    elif tipo == "pilar_largo":                 # pilar largo: metade do corpo
        w, c = bw * u(0.5, 0.8), x0 + bw * u(0.3, 0.7)
        r = (c - w / 2, 0, c + w / 2, H)
    elif tipo == "muro_alto":                   # muro alto: do quadril/tronco para baixo
        r = (x0 - 0.3 * bw, y0 + bh * u(0.30, 0.50), x1 + 0.3 * bw, H)
    elif tipo == "bloco_quase_total":           # tudo menos o topo da cabeça
        ux0, uy0, ux1, _ = uniao
        r = (ux0 - 0.05 * bw, uy0 + bh * u(0.12, 0.20), ux1 + 0.05 * bw, H)
    else:                                       # bloco_total
        ux0, uy0, ux1, _ = uniao
        r = (ux0 - 0.15 * bw, uy0 - 0.10 * bh, ux1 + 0.15 * bw, H)
    x0, y0, x1, y1 = r
    return [int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))]


def planejar(kpts, vis, fps, W, H, rng, intervalo):
    """Sorteia os cenários ao longo do vídeo: blocos com todos os tipos em ordem aleatória.

    Cada oclusão dura de ~1 a 2 s (>= 30 quadros, para que o horizonte de 20 quadros seja avaliável) e há
    pelo menos `intervalo` quadros sem obstáculo antes de cada uma (histórico visível para os modelos).
    """
    T = len(kpts)
    dur_min, dur_max = max(30, int(fps)), max(45, int(2 * fps))
    cenarios, t, fila = [], intervalo, []
    while True:
        dur = int(rng.integers(dur_min, dur_max + 1))
        if t + dur + intervalo > T:
            break
        if not fila:
            fila = [TIPOS[i] for i in rng.permutation(len(TIPOS))]
        tipo, cat = fila.pop()
        ini, fim = t, t + dur - 1
        mediana, uniao = caixa_pessoa(kpts, vis, ini, fim)
        if mediana is None or (mediana[3] - mediana[1]) < 0.1 * H:
            t += intervalo                      # sem pessoa nesse trecho: tenta mais adiante
            fila.append((tipo, cat))
            continue
        ret = retangulo(tipo, mediana, uniao, W, H, rng)
        cenarios.append(dict(id=len(cenarios), tipo=tipo, categoria=cat, ini=ini, fim=fim, retangulo=ret))
        t = fim + 1 + intervalo
    return cenarios


def cobertura(cen, kpts, vis):
    """Só para relatório: fração dos pontos de referência dentro do retângulo e juntas mais cobertas."""
    x0, y0, x1, y1 = cen["retangulo"]
    ini, fim = cen["ini"], cen["fim"]
    k, v = kpts[ini:fim + 1], vis[ini:fim + 1] == 1
    dentro = (k[..., 0] >= x0) & (k[..., 0] <= x1) & (k[..., 1] >= y0) & (k[..., 1] <= y1) & v
    cen["fracao_cobertura"] = round(float(dentro.sum() / max(v.sum(), 1)), 3)
    por_junta = dentro.sum(0) / np.maximum(v.sum(0), 1)
    cen["juntas_cobertas"] = [NOMES[j] for j in range(17) if por_junta[j] >= 0.5]


def desenhar(frame, tipo, r):
    x0, y0, x1, y1 = r
    if x1 <= x0 or y1 <= y0:
        return
    parede = tipo.startswith("muro")
    cor = (60, 80, 150) if parede else (120, 120, 120)         # BGR: tijolo / concreto
    cv2.rectangle(frame, (x0, y0), (x1, y1), cor, -1)
    if parede:
        for y in range(y0 + 20, y1, 20):                       # fiadas de tijolo
            cv2.line(frame, (x0, y), (x1, y), (40, 55, 110), 2)
    else:
        for x in range(x0 + 12, x1, 24):                       # veios do concreto
            cv2.line(frame, (x, y0), (x, y1), (100, 100, 100), 1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (30, 30, 30), 3)


def gerar(video, pasta, semente=0, intervalo=25):
    pasta = Path(pasta)
    ref = carregar_pose(pasta / "referencia_pose.npz")
    kpts, vis, W, H, fps = ref["kpts"], ref["vis"], ref["W"], ref["H"], ref["fps"]

    cenarios = planejar(kpts, vis, fps, W, H, np.random.default_rng(semente), intervalo)
    if not cenarios:
        sys.exit("Erro: vídeo curto ou sem pessoa detectada; nenhum cenário de oclusão pôde ser planejado.")
    for c in cenarios:
        cobertura(c, kpts, vis)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Erro: não foi possível abrir o vídeo '{video}'.")
    saida = pasta / "video_oclusao.mp4"
    out = cv2.VideoWriter(str(saida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for c in cenarios:
            if c["ini"] <= n <= c["fim"]:
                desenhar(frame, c["tipo"], c["retangulo"])
        out.write(frame)
        n += 1
    cap.release()
    out.release()

    # sincronia: mesmo nº de quadros (e, por construção, mesmo FPS e resolução) que o original e a pose
    cap = cv2.VideoCapture(str(saida))
    n_saida = 0
    while cap.read()[0]:
        n_saida += 1
    fps_saida = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not (n == n_saida == len(kpts)) or abs(fps_saida - fps) > 0.05:
        sys.exit(f"Erro de sincronia: original={n} quadros, pose={len(kpts)}, oclusão={n_saida} quadros; "
                 f"fps {fps:.3f} -> {fps_saida:.3f}.")

    with open(pasta / "cenarios.json", "w", encoding="utf-8") as f:
        json.dump(dict(fps=fps, largura=W, altura=H, n_quadros=n, semente=semente, cenarios=cenarios),
                  f, indent=2, ensure_ascii=False)

    print(f"Vídeo com oclusão: {saida}  ({n} quadros, {fps:.2f} fps, {W}x{H} — idêntico ao original)")
    print(f"{len(cenarios)} oclusões planejadas ({pasta / 'cenarios.json'}):")
    for c in cenarios:
        print(f"  #{c['id']:<2} quadros {c['ini']:>4}-{c['fim']:<4} {c['tipo']:<18} [{c['categoria']}] "
              f"cobertura {c['fracao_cobertura'] * 100:>3.0f}%  juntas: {', '.join(c['juntas_cobertas']) or '-'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="vídeo original limpo")
    ap.add_argument("pasta", help="pasta de trabalho do vídeo (contém referencia_pose.npz)")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--intervalo", type=int, default=25, help="quadros sem obstáculo entre oclusões")
    a = ap.parse_args()
    gerar(a.video, a.pasta, a.semente, a.intervalo)
