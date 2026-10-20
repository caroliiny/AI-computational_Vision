#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rastreamento_oclusao.py
=======================
Rastreamento de um alvo humano durante uma oclusão: EKF  x  LSTM  x  GRU.

O script faz tudo de ponta a ponta:

  1. Gera um cenário 2D sintético (fundo, chão com grade, pilar opaco, alvo humano
     articulado) e um movimento IRREGULAR (muda de velocidade e faz curvas).
  2. Grava DOIS vídeos com exatamente a mesma trajetória:
        - video_sem_oclusao.mp4 : o alvo passa NA FRENTE do pilar (sempre visível)
        - video_sintetico.mp4   : o alvo passa ATRÁS do pilar (oclusão geométrica)
     e guarda a posição verdadeira de TODOS os quadros (inclusive os ocultos).
  3. Estima a posição durante a oclusão com três métodos que só enxergam as
     OBSERVAÇÕES ruidosas dos quadros visíveis (os quadros ocultos viram NaN):
        - EKF  (filtro de Kalman estendido, modelo de curva CTRV)
        - LSTM (rede recorrente, predição recursiva)
        - GRU  (rede recorrente, predição recursiva)
  4. Compara deslocamento real x previsto (erro euclidiano, MAE, RMSE, FDE),
     gera gráficos, um vídeo de comparação e uma avaliação Monte Carlo.

As redes LSTM/GRU são implementadas do zero em NumPy (forward, BPTT e Adam), então
o único requisito é: numpy, opencv-python, matplotlib, pillow (+ imageio-ffmpeg
opcional, para gravar MP4 em H.264).

Uso:
    python rastreamento_oclusao.py                # execução completa
    python rastreamento_oclusao.py --rapido       # teste rápido (menos treino)
    python rastreamento_oclusao.py --seed 12      # outro vídeo / outra trajetória
    python rastreamento_oclusao.py --autoteste    # confere gradientes e Jacobiano

Estrutura de saída (a mesma sugerida no documento do trabalho):
    saida/01_video/     video_sem_oclusao.mp4, video_sintetico.mp4 (limpos) e
                        video_sem_oclusao_dados.mp4, video_sintetico_dados.mp4 (com x,y na tela),
                        trajetoria_real.csv, observacoes.csv, pose_real.csv
    saida/02_ekf/       previsoes_ekf.csv
    saida/03_lstm/      previsoes_lstm.csv
    saida/03_gru/       previsoes_gru.csv
    saida/04_avaliacao/ metricas.csv, erro_por_quadro.csv, metricas_monte_carlo.csv,
                        graficos.png, video_comparacao.mp4, parametros.json
"""
import argparse
import csv
import json
import os
import time
from dataclasses import dataclass

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# 0. PARÂMETROS GERAIS
# ============================================================================
FPS = 25
DT = 1.0 / FPS
DURACAO_S = 14.0
N = int(FPS * DURACAO_S)            # 350 quadros
LARG, ALT = 960, 540                # cena
ALT_PAINEL = 200                    # painel de erro (vídeo de comparação)
HORIZONTE = 235                     # linha do horizonte (px)
BASE_PILAR = 520                    # y da base do pilar (perto da câmera)
SIGMA_OBS = 1.0                     # ruído do "detector" da câmera (px)

JANELA = 10                         # tamanho da sequência de entrada das redes
ESC_IN = 10.0                       # normalização das entradas (px)
ESC_OUT = 3.0                       # normalização das saídas   (px)

SEED_TREINO, SEED_VAL, SEED_MC = 10_000, 20_000, 30_000

COR = {                             # RGB
    "real": (245, 245, 245),
    "EKF": (230, 126, 14),
    "LSTM": (0, 150, 210),
    "GRU": (60, 170, 80),
}
METODOS = ["EKF", "LSTM", "GRU"]
FONTE = font_manager.findfont("DejaVu Sans")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# 1. TRAJETÓRIA, POSE E CENÁRIO
# ============================================================================
def _suave(rng, n, sigma):
    """Ruído gaussiano suavizado (variância 1) -> irregularidade plausível."""
    k = int(4 * sigma)
    t = np.arange(-k, k + 1)
    g = np.exp(-0.5 * (t / sigma) ** 2)
    g /= g.sum()
    r = np.convolve(rng.standard_normal(n + 2 * k), g, mode="valid")
    return r / r.std()


def altura_corpo(y_quadril):
    """Altura do corpo (px) em função da profundidade (perspectiva leve)."""
    return (125.0 + 0.35 * (y_quadril - 380.0)) / 0.818


def gerar_trajetoria(rng):
    """Movimento irregular: velocidade variável + curvas suaves, sempre para a direita.

    Retorna pos (N,2) = centro do corpo (quadril) em pixels, vel (N,) px/s, ang (N,) rad.
    Faz amostragem por rejeição para o alvo ficar dentro da imagem.
    """
    t = np.arange(N) * DT
    while True:
        a1, T1, p1 = rng.uniform(.15, .40), rng.uniform(5, 9), rng.uniform(0, 2 * np.pi)
        a2, T2, p2 = rng.uniform(.05, .18), rng.uniform(2, 4), rng.uniform(0, 2 * np.pi)
        ang = (a1 * np.sin(2 * np.pi * t / T1 + p1) + a2 * np.sin(2 * np.pi * t / T2 + p2)
               + 0.10 * _suave(rng, N, 18))
        v0 = rng.uniform(52, 66)
        b1, P1, q1 = rng.uniform(.15, .35), rng.uniform(3, 7), rng.uniform(0, 2 * np.pi)
        b2, P2, q2 = rng.uniform(.05, .15), rng.uniform(1.5, 3), rng.uniform(0, 2 * np.pi)
        vel = v0 * (1 + b1 * np.sin(2 * np.pi * t / P1 + q1) + b2 * np.sin(2 * np.pi * t / P2 + q2)
                    + 0.06 * _suave(rng, N, 10))
        vel = np.maximum(vel, 0.4 * v0)
        x0, y0 = rng.uniform(45, 90), rng.uniform(330, 360)
        dx = vel * np.cos(ang) * DT
        dy = vel * np.sin(ang) * 0.5 * DT          # 0.5 = achatamento da profundidade
        x = x0 + np.concatenate([[0], np.cumsum(dx)[:-1]])
        y = y0 + np.concatenate([[0], np.cumsum(dy)[:-1]])
        if 800 <= x[-1] <= 925 and 305 <= y.min() and y.max() <= 395:
            return np.stack([x, y], 1), vel, ang


@dataclass
class Cenario:
    seed: int
    pos: np.ndarray        # (N,2) posição VERDADEIRA do centro do corpo
    vel: np.ndarray        # (N,) velocidade escalar (px/s)
    ang: np.ndarray        # (N,) direção do movimento (rad)
    altura: np.ndarray     # (N,) altura do corpo em px
    fase: np.ndarray       # (N,) fase do passo (rad)
    visivel: np.ndarray    # (N,) bool: False = totalmente escondido pelo pilar
    parcial: np.ndarray    # (N,) bool: parcialmente coberto (ainda detectável)
    obs: np.ndarray        # (N,2) observação ruidosa da câmera; NaN quando oculto
    oc: tuple              # (f0, f1): quadros ocultos = f0 .. f1-1
    pilar: tuple           # (x_esq, x_dir) em px


def montar_cenario(seed, exigir_manobra=False):
    """Trajetória + pilar posicionado de modo que o alvo fique oculto em [f0, f1).

    O pilar é uma barreira geométrica: o alvo só é 'oculto' quando o corpo inteiro
    está atrás dele. O movimento verdadeiro NÃO para durante a oclusão.
    exigir_manobra=True garante que há curva e mudança de velocidade atrás do pilar
    (critério independente dos algoritmos), para o vídeo ser um teste interessante.
    """
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        pos, vel, ang = gerar_trajetoria(rng)
        alt = altura_corpo(pos[:, 1])
        hw = 0.17 * alt                                    # meia-largura do corpo
        f0 = int(rng.integers(100, 171))
        f1 = f0 + int(rng.integers(50, 63))                # 2,0 a 2,5 s oculto
        pe = np.min(pos[f0:f1, 0] - hw[f0:f1])
        pd = np.max(pos[f0:f1, 0] + hw[f0:f1])
        ext_e, ext_d = pos[:, 0] - hw, pos[:, 0] + hw
        oculto = (ext_e >= pe) & (ext_d <= pd)
        if not np.array_equal(np.where(oculto)[0], np.arange(f0, f1)):
            continue
        if exigir_manobra:
            dang = np.max(np.abs(ang[f0:f1] - ang[f0 - 1]))
            dvel = np.max(np.abs(vel[f0:f1] / vel[f0 - 1] - 1))
            if dang < 0.25 or dvel < 0.20:
                continue
        parcial = (ext_d > pe) & (ext_e < pd) & ~oculto
        sig = SIGMA_OBS * (1 + 1.5 * parcial)              # parcial => detecção pior
        obs = pos + rng.standard_normal((N, 2)) * sig[:, None]
        obs[oculto] = np.nan
        cad = vel / (0.55 * alt)                           # ciclos de passo por segundo
        fase = 2 * np.pi * np.concatenate([[0], np.cumsum(cad * DT)[:-1]])
        return Cenario(seed, pos, vel, ang, alt, fase, ~oculto, parcial, obs, (f0, f1), (pe, pd))
    raise RuntimeError("não consegui montar o cenário; tente outro --seed")


def trajetoria_treino(seed):
    """Só trajetória + observações ruidosas completas (sem oclusão) para treinar."""
    rng = np.random.default_rng(seed)
    pos, _, _ = gerar_trajetoria(rng)
    return pos, pos + rng.standard_normal((N, 2)) * SIGMA_OBS


JUNTAS = ["cabeca", "pescoco", "ombro", "cotovelo_d", "pulso_d", "cotovelo_e", "pulso_e",
          "quadril", "joelho_d", "tornozelo_d", "joelho_e", "tornozelo_e"]


def esqueleto(cx, cy, Hc, dirx, fase, vel):
    """Pose 2D (vista lateral) com marcha animada. Retorna dict junta -> (x, y) em px."""
    A = np.clip(0.25 + 0.0045 * vel, 0.25, 0.70)
    L = 0.26 * Hc
    U1, U2 = 0.17 * Hc, 0.16 * Hc
    ombro = (cx, cy - 0.30 * Hc)
    j = {"quadril": (cx, cy), "pescoco": (cx, cy - 0.33 * Hc), "cabeca": (cx, cy - 0.41 * Hc - 0.02 * Hc),
         "ombro": ombro}
    for lado, ph in (("d", 0.0), ("e", np.pi)):
        a = A * np.sin(fase + ph)                          # coxa (+ = para frente)
        k = 0.9 * A * max(0.0, np.cos(fase + ph))          # flexão do joelho na fase de balanço
        jx, jy = cx + dirx * L * np.sin(a), cy + L * np.cos(a)
        j["joelho_" + lado] = (jx, jy)
        j["tornozelo_" + lado] = (jx + dirx * L * np.sin(a - k), jy + L * np.cos(a - k))
        b = -0.8 * A * np.sin(fase + ph)                   # braço oposto à perna
        ex, ey = ombro[0] + dirx * U1 * np.sin(b), ombro[1] + U1 * np.cos(b)
        j["cotovelo_" + lado] = (ex, ey)
        j["pulso_" + lado] = (ex + dirx * U2 * np.sin(b + 0.35), ey + U2 * np.cos(b + 0.35))
    return j


# ============================================================================
# 2. RENDERIZAÇÃO
# ============================================================================
def _p(pt):
    return int(round(pt[0])), int(round(pt[1]))


def criar_fundo():
    """Céu em degradê + chão bege com grade em perspectiva."""
    img = np.zeros((ALT, LARG, 3), np.uint8)
    for y in range(HORIZONTE):
        u = y / HORIZONTE
        img[y] = (52 + 20 * u, 58 + 22 * u, 74 + 26 * u)
    for y in range(HORIZONTE, ALT):
        u = (y - HORIZONTE) / (ALT - HORIZONTE)
        img[y] = (148 + 10 * u, 143 + 10 * u, 128 + 10 * u)
    grade = (176, 171, 156)
    vx = LARG // 2
    for i in range(-9, 10):                                # linhas de profundidade
        cv2.line(img, (vx, HORIZONTE), (int(vx + i * 190), ALT), grade, 1, cv2.LINE_AA)
    for k in range(0, 19):                                 # linhas transversais
        y = HORIZONTE + 900.0 / (3 + 1.5 * k) - 900.0 / (3 + 1.5 * 40)
        if HORIZONTE < y < ALT:
            cv2.line(img, (0, int(y)), (LARG, int(y)), grade, 1, cv2.LINE_AA)
    return img


def criar_pilar(pilar):
    """Retorna (rgb, mascara) do pilar opaco (preto, borda esquerda mais clara, base)."""
    pe, pd = int(round(pilar[0])), int(round(pilar[1]))
    rgb = np.zeros((ALT, LARG, 3), np.uint8)
    m = np.zeros((ALT, LARG), bool)
    xs, xd = max(pe, 0), min(pd, LARG)
    rgb[0:BASE_PILAR, xs:xd] = (24, 24, 24)
    rgb[0:BASE_PILAR, xs:min(xs + 10, xd)] = (52, 52, 54)
    rgb[0:BASE_PILAR, max(xd - 8, xs):xd] = (14, 14, 14)
    m[0:BASE_PILAR, xs:xd] = True
    xb0, xb1 = max(pe - 9, 0), min(pd + 9, LARG)
    rgb[BASE_PILAR - 14:BASE_PILAR + 4, xb0:xb1] = (16, 16, 16)
    m[BASE_PILAR - 14:BASE_PILAR + 4, xb0:xb1] = True
    return rgb, m


def desenhar_pessoa(img, j, Hc, estilo="real", cor=None, esp=None):
    """estilo 'real': esquerda azul / direita vermelha, juntas e cabeça brancas.
       estilo 'fantasma': traço único na cor do método, cabeça só com contorno."""
    esp = esp or max(3, int(round(0.03 * Hc)))
    seg_d = [("ombro", "cotovelo_d"), ("cotovelo_d", "pulso_d"), ("quadril", "joelho_d"),
             ("joelho_d", "tornozelo_d")]
    seg_e = [("ombro", "cotovelo_e"), ("cotovelo_e", "pulso_e"), ("quadril", "joelho_e"),
             ("joelho_e", "tornozelo_e")]
    tronco = [("pescoco", "quadril")]
    if estilo == "real":
        cd, ce, ct = (214, 62, 70), (52, 118, 200), (235, 235, 235)
    else:
        cd = ce = ct = cor
    for lista, c in ((seg_e, ce), (tronco, ct), (seg_d, cd)):
        for a, b in lista:
            cv2.line(img, _p(j[a]), _p(j[b]), c, esp, cv2.LINE_AA)
    cx, cy = j["cabeca"]
    if estilo == "real":
        cv2.ellipse(img, _p((cx, cy)), (int(0.055 * Hc), int(0.075 * Hc)), 0, 0, 360, (240, 240, 240), -1,
                    cv2.LINE_AA)
        for nome in JUNTAS:
            if nome != "cabeca":
                cv2.circle(img, _p(j[nome]), max(2, esp // 2 + 1), (255, 255, 255), -1, cv2.LINE_AA)
    else:
        cv2.circle(img, _p((cx, cy)), int(0.07 * Hc), cor, max(2, esp - 1), cv2.LINE_AA)


def desenhar_cena(fundo, fundo_pilar, pilar_rgb, pilar_m, cen, i, ocluir):
    """Um quadro da cena. ocluir=True: alvo ATRÁS do pilar; False: alvo NA FRENTE."""
    img = (fundo if ocluir else fundo_pilar).copy()
    x, y = cen.pos[i]
    Hc = cen.altura[i]
    j = esqueleto(x, y, Hc, 1.0, cen.fase[i], cen.vel[i])
    pe_y = y + 0.52 * Hc
    cv2.ellipse(img, _p((x, pe_y + 3)), (int(0.22 * Hc), max(3, int(0.035 * Hc))), 0, 0, 360,
                (112, 107, 95), -1, cv2.LINE_AA)
    desenhar_pessoa(img, j, Hc, "real")
    if ocluir:
        img[pilar_m] = pilar_rgb[pilar_m]
    return img


def hud(img, t, oculto, erros=None, linhas=None, legenda=True, rotulo_oc="OCLUSÃO — predição"):
    """Textos do vídeo (PIL, para acentuação correta)."""
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f1, f2 = ImageFont.truetype(FONTE, 22), ImageFont.truetype(FONTE, 17)

    def caixa(x, y, txt, fonte, fg, bg, pad=6):
        w = d.textlength(txt, font=fonte)
        d.rectangle([x, y, x + w + 2 * pad, y + fonte.size + 2 * pad], fill=bg)
        d.text((x + pad, y + pad - 1), txt, font=fonte, fill=fg)
        return x + w + 2 * pad

    xe = caixa(10, 10, f"t = {t:5.2f} s", f1, (255, 255, 255), (0, 0, 0))
    if oculto:
        caixa(xe + 8, 10, rotulo_oc, f1, (255, 255, 255), (200, 40, 45))
    else:
        caixa(xe + 8, 10, "alvo visível", f1, (255, 255, 255), (46, 125, 50))
    x = 10
    for nome in (["real"] + METODOS if legenda else []):
        w = d.textlength(nome, font=f2)
        d.rectangle([x, 52, x + w + 34, 52 + 17 + 12], fill=(0, 0, 0))
        d.text((x + 6, 56), "●", font=f2, fill=COR[nome])
        d.text((x + 26, 56), nome, font=f2, fill=(255, 255, 255))
        x += w + 42
    if erros is not None:
        txt = "erro de deslocamento  " + "   ".join(f"{k} {v:4.1f} px" for k, v in erros.items())
        caixa(10, 90, txt, f2, (255, 255, 255), (0, 0, 0))
    y = 128 if erros is not None else (90 if legenda else 52)
    for txt in (linhas or []):                              # dados numéricos (coordenadas etc.)
        caixa(10, y, txt, f2, (255, 255, 255), (0, 0, 0))
        y += 33
    return np.asarray(im)


def anotar_dados(img, cen, i):
    """Versão ANOTADA do vídeo: quadro, tempo, posição real (x,y), estado e rastro da trajetória."""
    oculto = not cen.visivel[i]
    base = img.copy()
    pts = np.round(cen.pos[:i + 1]).astype(np.int32)
    vis = cen.visivel[:i + 1]
    for k in range(1, len(pts)):                            # rastro: branco = visível, amarelo = oculto
        cv2.line(base, tuple(pts[k - 1]), tuple(pts[k]), (255, 215, 0) if not vis[k] else (255, 255, 255), 2,
                 cv2.LINE_AA)
    cx, cy = _p(cen.pos[i])
    cor = (255, 215, 0) if oculto else (255, 70, 70)
    cv2.drawMarker(base, (cx, cy), cor, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    cv2.circle(base, (cx, cy), 9, cor, 2, cv2.LINE_AA)
    x, y = cen.pos[i]
    linhas = [f"quadro {i:3d}   x_real = {x:6.1f}   y_real = {y:6.1f} px"]
    return hud(base, i * DT, oculto, None, linhas, legenda=False, rotulo_oc="OCLUSÃO (alvo oculto)")


class Painel:
    """Gráfico do erro de deslocamento x tempo na oclusão (parte de baixo do vídeo)."""

    def __init__(self, dur_oc, ymax):
        self.fig = plt.figure(figsize=(LARG / 100, ALT_PAINEL / 100), dpi=100, facecolor="#fafcfa")
        self.ax = self.fig.add_axes([0.07, 0.27, 0.91, 0.68], facecolor="#fafcfa")
        self.ax.set_xlim(0, dur_oc * 1.06)
        self.ax.set_ylim(0, ymax)
        self.ax.set_xlabel("tempo na oclusão (s)", fontsize=9)
        self.ax.set_ylabel("erro de deslocamento (px)", fontsize=9)
        self.ax.grid(alpha=.3)
        self.ax.tick_params(labelsize=8)
        self.linhas = {m: self.ax.plot([], [], color=np.array(COR[m]) / 255, lw=2, label=m)[0]
                       for m in METODOS}
        self.ax.legend(loc="upper left", fontsize=8)
        self.cache = None

    def quadro(self, tt=None, erros=None):
        if tt is not None:
            for m in METODOS:
                self.linhas[m].set_data(tt, erros[m])
            self.cache = None
        if self.cache is None:
            self.fig.canvas.draw()
            self.cache = np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()
        return self.cache

    def fechar(self):
        plt.close(self.fig)


class PainelPos:
    """Gráficos x(t) e y(t) da posição REAL, atualizados quadro a quadro (vídeos *_dados)."""

    def __init__(self, cen):
        f0, f1 = cen.oc
        self.t = np.arange(N) * DT
        self.pos = cen.pos
        self.fig = plt.figure(figsize=(LARG / 100, ALT_PAINEL / 100), dpi=100, facecolor="#fafcfa")
        self.ax = []
        self.lin, self.pt = [], []
        for k, (rot, cor) in enumerate((("x real (px)", "#1f77b4"), ("y real (px)", "#8e44ad"))):
            ax = self.fig.add_axes([0.07 + 0.50 * k, 0.27, 0.40, 0.68], facecolor="#fafcfa")
            ax.axvspan(f0 * DT, f1 * DT, color="#d62f3a", alpha=.15, label="intervalo da oclusão")
            ax.set_xlim(0, N * DT)
            lo, hi = cen.pos[:, k].min(), cen.pos[:, k].max()
            ax.set_ylim(lo - 0.08 * (hi - lo) - 5, hi + 0.08 * (hi - lo) + 5)
            ax.set_xlabel("tempo (s)", fontsize=9)
            ax.set_ylabel(rot, fontsize=9)
            ax.grid(alpha=.3)
            ax.tick_params(labelsize=8)
            self.lin.append(ax.plot([], [], color=cor, lw=2)[0])
            self.pt.append(ax.plot([], [], "o", color=cor, ms=6)[0])
            if k == 0:
                ax.legend(loc="upper left", fontsize=8)
            self.ax.append(ax)

    def quadro(self, i):
        for k in range(2):
            self.lin[k].set_data(self.t[:i + 1], self.pos[:i + 1, k])
            self.pt[k].set_data([self.t[i]], [self.pos[i, k]])
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def fechar(self):
        plt.close(self.fig)


class Gravador:
    """MP4 em H.264 via imageio-ffmpeg; se não houver, cai para OpenCV (mp4v)."""

    def __init__(self, caminho, tam):
        self.w = None
        try:
            import imageio
            self.w = imageio.get_writer(caminho, fps=FPS, codec="libx264", quality=8, macro_block_size=1,
                                        pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"])
            self.modo = "imageio"
        except Exception:
            self.w = cv2.VideoWriter(caminho, cv2.VideoWriter_fourcc(*"mp4v"), FPS, tam)
            self.modo = "cv2"

    def escrever(self, rgb):
        if self.modo == "imageio":
            self.w.append_data(rgb)
        else:
            self.w.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def fechar(self):
        self.w.close() if self.modo == "imageio" else self.w.release()


def salvar_csv(caminho, cab, linhas):
    with open(caminho, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cab)
        w.writerows(linhas)


def gerar_videos_e_dados(cen, pasta):
    """Etapa da Clara: vídeo SEM oclusão, vídeo COM oclusão e CSVs da trajetória verdadeira."""
    os.makedirs(pasta, exist_ok=True)
    fundo = criar_fundo()
    prgb, pm = criar_pilar(cen.pilar)
    fundo_pilar = fundo.copy()
    fundo_pilar[pm] = prgb[pm]
    g_sem = Gravador(os.path.join(pasta, "video_sem_oclusao.mp4"), (LARG, ALT))
    g_com = Gravador(os.path.join(pasta, "video_sintetico.mp4"), (LARG, ALT))
    g_sem_d = Gravador(os.path.join(pasta, "video_sem_oclusao_dados.mp4"), (LARG, ALT + ALT_PAINEL))
    g_com_d = Gravador(os.path.join(pasta, "video_sintetico_dados.mp4"), (LARG, ALT + ALT_PAINEL))
    painel = PainelPos(cen)                                  # mesma posição real nos dois vídeos
    for i in range(N):
        q_sem = desenhar_cena(fundo, fundo_pilar, prgb, pm, cen, i, ocluir=False)
        q_com = desenhar_cena(fundo, fundo_pilar, prgb, pm, cen, i, ocluir=True)
        g_sem.escrever(q_sem)                                # versões limpas (entrega do grupo)
        g_com.escrever(q_com)
        pn = painel.quadro(i)
        g_sem_d.escrever(np.vstack([anotar_dados(q_sem, cen, i), pn]))   # versões com dados + gráfico
        g_com_d.escrever(np.vstack([anotar_dados(q_com, cen, i), pn]))
    painel.fechar()
    for g in (g_sem, g_com, g_sem_d, g_com_d):
        g.fechar()

    salvar_csv(os.path.join(pasta, "trajetoria_real.csv"), ["frame", "x_real", "y_real", "visivel"],
               [[i, f"{cen.pos[i, 0]:.2f}", f"{cen.pos[i, 1]:.2f}", int(cen.visivel[i])] for i in range(N)])
    salvar_csv(os.path.join(pasta, "observacoes.csv"), ["frame", "x_obs", "y_obs", "visivel"],
               [[i, "" if np.isnan(cen.obs[i, 0]) else f"{cen.obs[i, 0]:.2f}",
                 "" if np.isnan(cen.obs[i, 1]) else f"{cen.obs[i, 1]:.2f}", int(cen.visivel[i])]
                for i in range(N)])
    cab, linhas = ["frame"], []
    for nome in JUNTAS:
        cab += [f"{nome}_x", f"{nome}_y"]
    for i in range(N):
        j = esqueleto(*cen.pos[i], cen.altura[i], 1.0, cen.fase[i], cen.vel[i])
        linhas.append([i] + [f"{v:.2f}" for nome in JUNTAS for v in j[nome]])
    salvar_csv(os.path.join(pasta, "pose_real.csv"), cab, linhas)
    return fundo, prgb, pm


# ============================================================================
# 3. EKF  (modelo de curva CTRV: estado = [x, y, v, theta, omega])
# ============================================================================
class EKF:
    """Filtro de Kalman Estendido com o modelo de velocidade e taxa de giro constantes.

    estado  s = [x, y, v, theta, omega]     (v em px/s, theta em rad, omega em rad/s)
    predição  s' = f(s),  P' = F P F^T + Q   (f é NÃO linear -> por isso 'estendido')
    correção  só quando há observação z = [x, y] ruidosa (quadro visível):
              K = P H^T (H P H^T + R)^-1 ;  s = s + K (z - H s) ;  P = (I - K H) P
    Durante a oclusão só roda a predição, e P cresce (incerteza aumenta).
    omega decai com constante de tempo tau (curvas não duram para sempre).
    """

    def __init__(self, q_v=1.0, q_w=0.03, tau=1.5, q_pos=0.05, q_th=0.005, sigma=SIGMA_OBS):
        self.Q = np.diag([q_pos ** 2, q_pos ** 2, q_v ** 2, q_th ** 2, q_w ** 2])
        self.rho = np.exp(-DT / tau)
        self.R = np.eye(2) * sigma ** 2
        self.H = np.zeros((2, 5))
        self.H[0, 0] = self.H[1, 1] = 1.0
        self.s = None
        self.P = None

    def f(self, s):
        x, y, v, th, w = s
        if abs(w) > 1e-3:
            th2 = th + w * DT
            x2 = x + v / w * (np.sin(th2) - np.sin(th))
            y2 = y + v / w * (np.cos(th) - np.cos(th2))
        else:                                                 # limite w->0 (ponto médio), contínuo em w
            m = th + w * DT / 2
            x2 = x + v * np.cos(m) * DT
            y2 = y + v * np.sin(m) * DT
        return np.array([x2, y2, v, th + w * DT, w * self.rho])

    def jacobiano(self, s):
        x, y, v, th, w = s
        F = np.eye(5)
        if abs(w) > 1e-3:
            th2 = th + w * DT
            s0, s2, c0, c2 = np.sin(th), np.sin(th2), np.cos(th), np.cos(th2)
            F[0, 2] = (s2 - s0) / w
            F[0, 3] = v / w * (c2 - c0)
            F[0, 4] = v * (DT * c2 / w - (s2 - s0) / w ** 2)
            F[1, 2] = (c0 - c2) / w
            F[1, 3] = v / w * (s2 - s0)
            F[1, 4] = v * (DT * s2 / w - (c0 - c2) / w ** 2)
        else:
            m = th + w * DT / 2
            F[0, 2], F[0, 3], F[0, 4] = np.cos(m) * DT, -v * np.sin(m) * DT, -v * np.sin(m) * DT ** 2 / 2
            F[1, 2], F[1, 3], F[1, 4] = np.sin(m) * DT, v * np.cos(m) * DT, v * np.cos(m) * DT ** 2 / 2
        F[3, 4] = DT
        F[4, 4] = self.rho
        return F

    def iniciar(self, z):
        """Estado inicial por reta de mínimos quadrados nas primeiras observações."""
        t = np.arange(len(z)) * DT
        px, py = np.polyfit(t, z[:, 0], 1), np.polyfit(t, z[:, 1], 1)
        vx, vy = px[0], py[0]
        self.s = np.array([z[-1, 0], z[-1, 1], np.hypot(vx, vy), np.arctan2(vy, vx), 0.0])
        self.P = np.diag([SIGMA_OBS ** 2, SIGMA_OBS ** 2, 15.0 ** 2, 0.3 ** 2, 0.3 ** 2])

    def predizer(self):
        F = self.jacobiano(self.s)
        self.s = self.f(self.s)
        self.P = F @ self.P @ F.T + self.Q

    def corrigir(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.s = self.s + K @ (z - self.H @ self.s)
        I_KH = np.eye(5) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T       # forma de Joseph (estável)


def rodar_ekf(obs, **kw):
    """Roda o EKF sobre as observações (NaN = oculto). Retorna est (N,2), P2 (N,2,2), previsto (N,)."""
    ekf, K0 = EKF(**kw), 6
    est, P2, prev = np.full((N, 2), np.nan), np.zeros((N, 2, 2)), np.zeros(N, bool)
    for i in range(N):
        if ekf.s is None:
            est[i] = obs[i]
            P2[i] = np.eye(2) * SIGMA_OBS ** 2
            if i == K0 - 1:
                ekf.iniciar(obs[:K0])
            continue
        ekf.predizer()
        if np.isnan(obs[i, 0]):
            prev[i] = True                                    # só predição (oclusão)
        else:
            ekf.corrigir(obs[i])
        est[i], P2[i] = ekf.s[:2], ekf.P[:2, :2]
    return est, P2, prev


def sintonizar_ekf(cen_val):
    """Escolhe q_v, q_w, tau do EKF minimizando o MAE na oclusão dos cenários de VALIDAÇÃO."""
    melhor = (np.inf, None)
    for q_v in (0.3, 1.0, 3.0):
        for q_w in (0.01, 0.03, 0.1, 0.3):
            for tau in (0.35, 0.7, 1.5, 3.0, 1e9):
                kw = dict(q_v=q_v, q_w=q_w, tau=tau)
                mae = np.mean([erro_oclusao(c, rodar_ekf(c.obs, **kw)[0][c.oc[0]:c.oc[1]]).mean()
                               for c in cen_val])
                if mae < melhor[0]:
                    melhor = (mae, kw)
    return melhor[1], melhor[0]


# ============================================================================
# 4. LSTM e GRU em NumPy (forward, BPTT, Adam)
# ============================================================================
def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


class RNN:
    """Rede many-to-one:  sequência (B,T,2) -> RNN (LSTM|GRU) -> Dense -> (B,2).

    LSTM: i,f,g,o ;  c_t = f*c_{t-1} + i*g ;  h_t = o*tanh(c_t)
    GRU : r,u,n   ;  h_t = (1-u)*n + u*h_{t-1}
    """

    def __init__(self, tipo, hid=32, n_in=2, n_out=2, seed=0):
        assert tipo in ("lstm", "gru")
        self.tipo, self.hid = tipo, hid
        rng = np.random.default_rng(seed)
        G = 4 if tipo == "lstm" else 3
        s = 1.0 / np.sqrt(hid)
        self.p = {"Wx": rng.uniform(-s, s, (n_in, G * hid)),
                  "Wh": rng.uniform(-s, s, (hid, G * hid)),
                  "b": np.zeros(G * hid),
                  "Wo": rng.uniform(-s, s, (hid, n_out)),
                  "bo": np.zeros(n_out)}
        if tipo == "lstm":
            self.p["b"][hid:2 * hid] = 1.0                    # bias do forget gate = 1

    # ---------------- forward ----------------
    def forward(self, X, guardar=True):
        p, Hd = self.p, self.hid
        B, T, _ = X.shape
        h = np.zeros((B, Hd))
        c = np.zeros((B, Hd))
        cache = []
        for t in range(T):
            x = X[:, t]
            if self.tipo == "lstm":
                z = x @ p["Wx"] + h @ p["Wh"] + p["b"]
                i, f = _sig(z[:, :Hd]), _sig(z[:, Hd:2 * Hd])
                g, o = np.tanh(z[:, 2 * Hd:3 * Hd]), _sig(z[:, 3 * Hd:])
                c2 = f * c + i * g
                tc = np.tanh(c2)
                h2 = o * tc
                if guardar:
                    cache.append((x, h, c, i, f, g, o, tc))
                h, c = h2, c2
            else:
                xw = x @ p["Wx"] + p["b"]
                r = _sig(xw[:, :Hd] + h @ p["Wh"][:, :Hd])
                u = _sig(xw[:, Hd:2 * Hd] + h @ p["Wh"][:, Hd:2 * Hd])
                n = np.tanh(xw[:, 2 * Hd:] + (r * h) @ p["Wh"][:, 2 * Hd:])
                h2 = (1 - u) * n + u * h
                if guardar:
                    cache.append((x, h, r, u, n))
                h = h2
        y = h @ p["Wo"] + p["bo"]
        return y, (cache, h)

    def prever(self, X):
        return self.forward(X, guardar=False)[0]

    # ---------------- backward (BPTT) ----------------
    def backward(self, dy, estado):
        cache, hT = estado
        p, Hd = self.p, self.hid
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["Wo"], g["bo"] = hT.T @ dy, dy.sum(0)
        dh = dy @ p["Wo"].T
        dc = np.zeros_like(dh)
        for t in reversed(range(len(cache))):
            if self.tipo == "lstm":
                x, hp, cp, i, f, gg, o, tc = cache[t]
                do = dh * tc
                dc = dc + dh * o * (1 - tc ** 2)
                dz = np.concatenate([dc * gg * i * (1 - i), dc * cp * f * (1 - f),
                                     dc * i * (1 - gg ** 2), do * o * (1 - o)], 1)
                g["Wx"] += x.T @ dz
                g["Wh"] += hp.T @ dz
                g["b"] += dz.sum(0)
                dh = dz @ p["Wh"].T
                dc = dc * f
            else:
                x, hp, r, u, n = cache[t]
                dn_pre = dh * (1 - u) * (1 - n ** 2)
                du_pre = dh * (hp - n) * u * (1 - u)
                d_rh = dn_pre @ p["Wh"][:, 2 * Hd:].T
                dr_pre = d_rh * hp * r * (1 - r)
                g["Wh"][:, 2 * Hd:] += (r * hp).T @ dn_pre
                g["Wh"][:, :Hd] += hp.T @ dr_pre
                g["Wh"][:, Hd:2 * Hd] += hp.T @ du_pre
                dzx = np.concatenate([dr_pre, du_pre, dn_pre], 1)
                g["Wx"] += x.T @ dzx
                g["b"] += dzx.sum(0)
                dh = (dh * u + d_rh * r + dr_pre @ p["Wh"][:, :Hd].T + du_pre @ p["Wh"][:, Hd:2 * Hd].T)
        return g


class Adam:
    def __init__(self, params, b1=0.9, b2=0.999, eps=1e-8):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t, self.b1, self.b2, self.eps = 0, b1, b2, eps

    def passo(self, params, grads, lr):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= lr * mh / (np.sqrt(vh) + self.eps)


# ---------------- dados, predição recursiva, treino ----------------
def montar_janelas(lista):
    """lista de (pos_verdadeira, obs_ruidosa) -> X (n,JANELA,2), Y (n,2).

    Entrada : últimas JANELA observações, em posição RELATIVA à última (invariante à
              posição absoluta na imagem), divididas por ESC_IN.
    Alvo    : posição verdadeira do quadro seguinte menos a última observação, / ESC_OUT.
    """
    Xs, Ys = [], []
    for pos, obs in lista:
        w = np.lib.stride_tricks.sliding_window_view(obs, (JANELA, 2))[:N - JANELA, 0]
        Xs.append((w - w[:, -1:, :]) / ESC_IN)
        Ys.append((pos[JANELA:] - w[:, -1, :]) / ESC_OUT)
    return np.concatenate(Xs), np.concatenate(Ys)


def prever_oclusao_rede(modelo, obs, f0, f1):
    """Predição RECURSIVA: a saída de cada passo volta como entrada do seguinte.

    A rede só vê obs[f0-JANELA:f0] (quadros visíveis). As posições verdadeiras dos
    quadros ocultos nunca entram aqui.
    """
    w = obs[f0 - JANELA:f0].copy()
    assert not np.isnan(w).any(), "a janela de entrada deve conter só quadros visíveis"
    saida = []
    for _ in range(f1 - f0):
        d = modelo.prever(((w - w[-1]) / ESC_IN)[None])[0] * ESC_OUT
        pt = w[-1] + d
        saida.append(pt)
        w = np.vstack([w[1:], pt])
    return np.array(saida)


def prever_lote(modelo, cens):
    """Versão em lote (usada na validação): todos os cenários avançam passo a passo juntos."""
    W = np.stack([c.obs[c.oc[0] - JANELA:c.oc[0]] for c in cens])
    L = max(c.oc[1] - c.oc[0] for c in cens)
    saida = np.zeros((len(cens), L, 2))
    for k in range(L):
        d = modelo.prever((W - W[:, -1:, :]) / ESC_IN) * ESC_OUT
        pt = W[:, -1, :] + d
        saida[:, k] = pt
        W = np.concatenate([W[:, 1:], pt[:, None]], 1)
    return saida


def erro_oclusao(cen, pred):
    f0, f1 = cen.oc
    return np.linalg.norm(cen.pos[f0:f1] - pred, axis=1)


def mae_rollout(modelo, cens):
    sai = prever_lote(modelo, cens)
    return np.mean([erro_oclusao(c, sai[k, :c.oc[1] - c.oc[0]]).mean() for k, c in enumerate(cens)])


def treinar(tipo, X, Y, cen_val, epocas, hid=32, bs=128, lr=4e-3, seed=0):
    """Treina com MSE + Adam (+ clip de gradiente). Seleciona a melhor época pelo MAE da
    PREDIÇÃO RECURSIVA na oclusão dos cenários de validação (não os do teste)."""
    modelo = RNN(tipo, hid, seed=seed)
    adam = Adam(modelo.p)
    rng = np.random.default_rng(seed)
    melhor, pesos, hist = np.inf, None, []
    for ep in range(epocas):
        lr_ep = 0.5 * lr * (1 + np.cos(np.pi * ep / epocas)) + 2e-4
        perm = rng.permutation(len(X))
        perda = 0.0
        for k in range(0, len(X), bs):
            idx = perm[k:k + bs]
            yp, est = modelo.forward(X[idx])
            err = yp - Y[idx]
            perda += (err ** 2).sum()
            g = modelo.backward(2 * err / err.size, est)
            norma = np.sqrt(sum((v ** 2).sum() for v in g.values()))
            if norma > 1.0:
                for v in g.values():
                    v *= 1.0 / norma
            adam.passo(modelo.p, g, lr_ep)
        val = mae_rollout(modelo, cen_val)
        hist.append((ep + 1, perda / Y.size, val))
        if val < melhor:
            melhor, pesos = val, {k: v.copy() for k, v in modelo.p.items()}
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epocas - 1:
            log(f"   {tipo.upper()} época {ep + 1:3d}/{epocas}  MSE treino {perda / Y.size:.4f}  "
                f"MAE validação (recursivo) {val:5.1f} px")
    modelo.p = pesos
    return modelo, melhor, hist


# ============================================================================
# 5. AVALIAÇÃO
# ============================================================================
def metricas(cen, pred):
    f0, f1 = cen.oc
    e = erro_oclusao(cen, pred)
    ref = cen.pos[f0 - 1]                                     # última posição visível
    d_real = np.linalg.norm(cen.pos[f1 - 1] - ref)
    d_prev = np.linalg.norm(pred[-1] - ref)
    return {"MAE_px": e.mean(), "RMSE_px": np.sqrt((e ** 2).mean()), "FDE_px": e[-1], "erro_max_px": e.max(),
            "desloc_real_px": d_real, "desloc_prev_px": d_prev, "erro_desloc_px": abs(d_prev - d_real)}


def rodar_metodos(cen, modelos, kw_ekf):
    """Roda os 3 métodos sobre as OBSERVAÇÕES do cenário. Retorna dict nome -> (n_oc,2) + extras do EKF."""
    f0, f1 = cen.oc
    est, P2, _ = rodar_ekf(cen.obs, **kw_ekf)
    pred = {"EKF": est[f0:f1]}
    for nome in ("LSTM", "GRU"):
        pred[nome] = prever_oclusao_rede(modelos[nome], cen.obs, f0, f1)
    return pred, est, P2


def monte_carlo(modelos, kw_ekf, n_mc):
    """Repete o experimento em n_mc cenários inéditos (uma única trajetória é anedota)."""
    res = {m: [] for m in METODOS}
    curvas = {m: [] for m in METODOS}
    for k in range(n_mc):
        c = montar_cenario(SEED_MC + k)
        pred, _, _ = rodar_metodos(c, modelos, kw_ekf)
        for m in METODOS:
            res[m].append(metricas(c, pred[m]))
            curvas[m].append(erro_oclusao(c, pred[m])[:50])           # menor oclusão = 50 quadros
    return res, {m: np.array(v) for m, v in curvas.items()}


def graficos(cen, pred, est_ekf, mc_curvas, mc_res, caminho):
    f0, f1 = cen.oc
    cores = {m: np.array(COR[m]) / 255 for m in METODOS}
    fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
    a = ax[0, 0]
    a.axvspan(*cen.pilar, color="0.15", alpha=.15, label="pilar")
    a.plot(cen.pos[:, 0], cen.pos[:, 1], color="0.6", lw=1, label="trajetória real (completa)")
    a.plot(cen.pos[f0:f1, 0], cen.pos[f0:f1, 1], color="k", lw=3, label="real (oculto)")
    for m in METODOS:
        a.plot(pred[m][:, 0], pred[m][:, 1], color=cores[m], lw=2.2, label=m)
        a.plot(*pred[m][-1], "o", color=cores[m])
    a.plot(*cen.pos[f0 - 1], "ks", label="último quadro visível")
    a.set_ylim(cen.pos[:, 1].max() + 45, cen.pos[:, 1].min() - 45)      # y da imagem cresce para baixo
    a.set_title("Trajetória na imagem (px)")
    a.set_xlabel("x (px)")
    a.set_ylabel("y (px)")
    a.legend(fontsize=8, loc="lower right")
    a = ax[0, 1]
    t = np.arange(f0, f1)
    for m in METODOS:
        a.plot(t, np.linalg.norm(cen.pos[f0:f1] - pred[m], axis=1), color=cores[m], lw=2, label=m)
    a.set_title("Erro de deslocamento por quadro (vídeo gerado)")
    a.set_xlabel("quadro")
    a.set_ylabel("erro euclidiano (px)")
    a.grid(alpha=.3)
    a.legend()
    a = ax[1, 0]
    tt = np.arange(mc_curvas["EKF"].shape[1]) * DT
    for m in METODOS:
        mu, sd = mc_curvas[m].mean(0), mc_curvas[m].std(0)
        a.plot(tt, mu, color=cores[m], lw=2, label=m)
        a.fill_between(tt, mu - sd, mu + sd, color=cores[m], alpha=.15)
    a.set_title(f"Monte Carlo ({len(mc_curvas['EKF'])} trajetórias inéditas): erro x tempo oculto")
    a.set_xlabel("tempo desde o início da oclusão (s)")
    a.set_ylabel("erro médio ± desvio (px)")
    a.grid(alpha=.3)
    a.legend()
    a = ax[1, 1]
    x = np.arange(len(METODOS))
    v1 = [metricas(cen, pred[m])["MAE_px"] for m in METODOS]
    v2 = [np.mean([r["MAE_px"] for r in mc_res[m]]) for m in METODOS]
    s2 = [np.std([r["MAE_px"] for r in mc_res[m]]) for m in METODOS]
    a.bar(x - .2, v1, .4, color=[cores[m] for m in METODOS], label="vídeo gerado")
    a.bar(x + .2, v2, .4, yerr=s2, color=[cores[m] for m in METODOS], alpha=.45, capsize=4, label="Monte Carlo")
    for xi, v in zip(x - .2, v1):
        a.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    a.set_xticks(x)
    a.set_xticklabels(METODOS)
    a.set_ylabel("MAE na oclusão (px)")
    a.set_title("Erro médio na oclusão (barra escura = vídeo, clara = Monte Carlo)")
    a.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(caminho, dpi=110)
    plt.close(fig)


def video_comparacao(cen, pred, est_ekf, P2, fundo, prgb, pm, caminho):
    """Vídeo no estilo do exemplo: fantasmas EKF/LSTM/GRU + posição real oculta (branco),
    rastros, elipse de incerteza do EKF e painel do erro x tempo na oclusão."""
    f0, f1 = cen.oc
    L = f1 - f0
    erros = {m: np.linalg.norm(cen.pos[f0:f1] - pred[m], axis=1) for m in METODOS}
    ymax = np.ceil(max(e.max() for e in erros.values()) * 1.1 / 10) * 10
    painel = Painel(L * DT, ymax)
    g = Gravador(caminho, (LARG, ALT + ALT_PAINEL))
    # fase do passo dos fantasmas: continua a cadência a partir da última fase visível
    fases = {}
    for m in METODOS:
        v = np.r_[np.linalg.norm(pred[m][0] - cen.obs[f0 - 1]),
                  np.linalg.norm(np.diff(pred[m], axis=0), axis=1)] / DT
        v = np.convolve(np.pad(v, 3, mode="edge"), np.ones(7) / 7, mode="valid")
        alt = altura_corpo(pred[m][:, 1])
        fases[m] = (cen.fase[f0 - 1] + 2 * np.pi * np.cumsum(v / (0.55 * alt)) * DT, v)
    for i in range(N):
        base = desenhar_cena(fundo, fundo, prgb, pm, cen, i, ocluir=True)
        em_oc = f0 <= i < f1
        k = min(i, f1 - 1) - f0                              # último índice de predição disponível
        if i >= f0:
            trilhas = [("real", cen.pos[f0:f0 + k + 1])] + [(m, pred[m][:k + 1]) for m in METODOS]
            for nome, pts in trilhas:
                if len(pts) > 1:
                    cv2.polylines(base, [np.round(pts).astype(np.int32)], False, COR[nome], 3, cv2.LINE_AA)
        if em_oc:
            x, y = cen.pos[i]
            jr = esqueleto(x, y, cen.altura[i], 1.0, cen.fase[i], cen.vel[i])
            desenhar_pessoa(base, jr, cen.altura[i], "fantasma", COR["real"], esp=2)
            for m in METODOS:
                px, py = pred[m][k]
                fase, vel = fases[m][0][k], fases[m][1][k]
                desenhar_pessoa(base, esqueleto(px, py, altura_corpo(py), 1.0, fase, vel),
                                altura_corpo(py), "fantasma", COR[m], esp=4)
            C = P2[i]
            val, vec = np.linalg.eigh(C)
            eixos = np.minimum(2 * np.sqrt(np.maximum(val, 1e-9)), 150)
            cv2.ellipse(base, _p(est_ekf[i]), (int(eixos[1]) + 1, int(eixos[0]) + 1),
                        float(np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))), 0, 360, COR["EKF"], 2,
                        cv2.LINE_AA)
        if em_oc:
            eq = {m: erros[m][k] for m in METODOS}
            lin = [f"quadro {i:3d}   real   x = {cen.pos[i, 0]:6.1f}  y = {cen.pos[i, 1]:6.1f}"]
            lin += [f"{m:<5s}  previsto  x = {pred[m][k, 0]:6.1f}  y = {pred[m][k, 1]:6.1f}" for m in METODOS]
            quadro = hud(base, i * DT, True, eq, lin)
            pn = painel.quadro(np.arange(k + 1) * DT, {m: erros[m][:k + 1] for m in METODOS})
        else:
            quadro = hud(base, i * DT, False, None,
                         [f"quadro {i:3d}   real   x = {cen.pos[i, 0]:6.1f}  y = {cen.pos[i, 1]:6.1f}"])
            pn = painel.quadro()
        g.escrever(np.vstack([quadro, pn]))
    g.fechar()
    painel.fechar()


# ============================================================================
# 6. AUTOTESTE  (confere BPTT por diferenças finitas e o Jacobiano do EKF)
# ============================================================================
def autoteste():
    ok = True
    rng = np.random.default_rng(0)
    for tipo in ("lstm", "gru"):
        m = RNN(tipo, hid=5, seed=1)
        X, Y = rng.standard_normal((4, 6, 2)), rng.standard_normal((4, 2))

        def perda():
            return 0.5 * np.sum((m.prever(X) - Y) ** 2)

        yp, est = m.forward(X)
        g = m.backward(yp - Y, est)
        pior = 0.0
        for k, v in m.p.items():
            for idx in np.ndindex(v.shape):
                old = v[idx]
                v[idx] = old + 1e-6
                a = perda()
                v[idx] = old - 1e-6
                b = perda()
                v[idx] = old
                num = (a - b) / 2e-6
                pior = max(pior, abs(num - g[k][idx]) / (np.abs(g[k]).max() + 1e-8))
        print(f"  gradiente {tipo.upper()}: pior erro (relativo ao maior gradiente) = {pior:.2e}")
        ok &= pior < 1e-5
    e = EKF()
    for s in (np.array([100, 300, 60, 0.3, 0.2]), np.array([100, 300, 60, -0.4, -0.05]),
              np.array([100, 300, 60, 0.1, 0.0])):
        J = np.zeros((5, 5))
        for k in range(5):
            d = np.zeros(5)
            d[k] = 1e-6
            J[:, k] = (e.f(s + d) - e.f(s - d)) / 2e-6
        err = np.abs(J - e.jacobiano(s)).max()
        print(f"  Jacobiano EKF (omega={s[4]:+.2f}): erro máx = {err:.2e}")
        ok &= err < 1e-4
    print("AUTOTESTE:", "OK" if ok else "FALHOU")
    return ok


# ============================================================================
# 7. PROGRAMA PRINCIPAL
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7, help="semente do vídeo/trajetória de teste")
    ap.add_argument("--saida", default="saida", help="pasta de saída")
    ap.add_argument("--rapido", action="store_true", help="treino curto (para testar o código)")
    ap.add_argument("--epocas", type=int, default=None)
    ap.add_argument("--n-treino", type=int, default=None, help="nº de trajetórias de treino")
    ap.add_argument("--n-mc", type=int, default=None, help="nº de trajetórias do Monte Carlo")
    ap.add_argument("--autoteste", action="store_true")
    a = ap.parse_args()
    if a.autoteste:
        raise SystemExit(0 if autoteste() else 1)
    epocas = a.epocas or (4 if a.rapido else 30)
    n_treino = a.n_treino or (40 if a.rapido else 200)
    n_mc = a.n_mc or (10 if a.rapido else 40)
    hid = 32
    d1, d2, d3a, d3b, d4 = [os.path.join(a.saida, p) for p in
                            ("01_video", "02_ekf", "03_lstm", "03_gru", "04_avaliacao")]
    for d in (d2, d3a, d3b, d4):
        os.makedirs(d, exist_ok=True)

    # ---- 1) cenário, vídeos e verdade --------------------------------------
    log("1/5  Gerando cenário, trajetória irregular e vídeos (sem e com oclusão)...")
    cen = montar_cenario(a.seed, exigir_manobra=True)
    f0, f1 = cen.oc
    fundo, prgb, pm = gerar_videos_e_dados(cen, d1)
    log(f"     {N} quadros @ {FPS} fps | oclusão: quadros {f0}..{f1 - 1} ({(f1 - f0) * DT:.2f} s) | "
        f"pilar x=[{cen.pilar[0]:.0f}, {cen.pilar[1]:.0f}] px")

    # ---- 2) dados de treino / validação (trajetórias independentes do vídeo) ---
    log(f"2/5  Preparando dados: {n_treino} trajetórias de treino, 20 de validação...")
    treino = [trajetoria_treino(SEED_TREINO + k) for k in range(n_treino)]
    X, Y = montar_janelas(treino)
    cen_val = [montar_cenario(SEED_VAL + k) for k in range(20)]
    log(f"     {len(X)} janelas de {JANELA} quadros (entrada) -> próxima posição (alvo)")

    # ---- 3) EKF --------------------------------------------------------------
    log("3/5  EKF: sintonizando q_v, q_omega, tau nos cenários de validação...")
    kw_ekf, mae_val = sintonizar_ekf(cen_val)
    log(f"     melhor: {kw_ekf}  (MAE validação {mae_val:.1f} px)")

    # ---- 4) redes ------------------------------------------------------------
    log(f"4/5  Treinando LSTM e GRU ({epocas} épocas, {hid} unidades ocultas)...")
    modelos, hist = {}, {}
    for nome in ("LSTM", "GRU"):
        t0 = time.time()
        modelos[nome], melhor, hist[nome] = treinar(nome.lower(), X, Y, cen_val, epocas, hid=hid, seed=1)
        log(f"     {nome}: melhor MAE de validação {melhor:.1f} px  ({time.time() - t0:.0f} s)")

    # ---- 5) predição no vídeo + avaliação ---------------------------------------
    log("5/5  Predizendo a oclusão do vídeo e avaliando...")
    pred, est_ekf, P2 = rodar_metodos(cen, modelos, kw_ekf)
    quadros = np.arange(f0, f1)
    salvar_csv(os.path.join(d2, "previsoes_ekf.csv"), ["frame", "x_ekf", "y_ekf", "em_oclusao"],
               [[i, f"{est_ekf[i, 0]:.2f}", f"{est_ekf[i, 1]:.2f}", int(f0 <= i < f1)] for i in range(N)])
    for nome, pasta in (("LSTM", d3a), ("GRU", d3b)):
        salvar_csv(os.path.join(pasta, f"previsoes_{nome.lower()}.csv"),
                   ["frame", f"x_{nome.lower()}", f"y_{nome.lower()}"],
                   [[q, f"{p[0]:.2f}", f"{p[1]:.2f}"] for q, p in zip(quadros, pred[nome])])

    linhas, mets = [], {}
    for m in METODOS:
        mets[m] = metricas(cen, pred[m])
        linhas.append([m] + [f"{v:.2f}" for v in mets[m].values()])
    salvar_csv(os.path.join(d4, "metricas.csv"), ["metodo"] + list(mets["EKF"].keys()), linhas)
    salvar_csv(os.path.join(d4, "erro_por_quadro.csv"),
               ["frame", "x_real", "y_real"] + [f"{m}_{c}" for m in METODOS for c in ("x", "y", "erro_px")],
               [[q, f"{cen.pos[q, 0]:.2f}", f"{cen.pos[q, 1]:.2f}"] +
                [v for m in METODOS for v in (f"{pred[m][k, 0]:.2f}", f"{pred[m][k, 1]:.2f}",
                                              f"{np.linalg.norm(cen.pos[q] - pred[m][k]):.2f}")]
                for k, q in enumerate(quadros)])

    log(f"     Monte Carlo em {n_mc} trajetórias inéditas...")
    mc_res, mc_curvas = monte_carlo(modelos, kw_ekf, n_mc)
    chaves = list(mets["EKF"].keys())
    salvar_csv(os.path.join(d4, "metricas_monte_carlo.csv"),
               ["metodo"] + [f"{k}_media" for k in chaves] + [f"{k}_desvio" for k in chaves] + ["MAE_mediana"],
               [[m] + [f"{np.mean([r[k] for r in mc_res[m]]):.2f}" for k in chaves]
                + [f"{np.std([r[k] for r in mc_res[m]]):.2f}" for k in chaves]
                + [f"{np.median([r['MAE_px'] for r in mc_res[m]]):.2f}"] for m in METODOS])

    graficos(cen, pred, est_ekf, mc_curvas, mc_res, os.path.join(d4, "graficos.png"))
    log("     Gerando vídeo de comparação...")
    video_comparacao(cen, pred, est_ekf, P2, fundo, prgb, pm, os.path.join(d4, "video_comparacao.mp4"))

    with open(os.path.join(d4, "parametros.json"), "w", encoding="utf-8") as f:
        json.dump({"fps": FPS, "quadros": N, "seed_video": a.seed, "oclusao_quadros": [f0, f1 - 1],
                   "sigma_obs_px": SIGMA_OBS, "ekf": {"modelo": "CTRV [x,y,v,theta,omega]", **kw_ekf},
                   "redes": {"janela": JANELA, "unidades_ocultas": hid, "epocas": epocas,
                             "trajetorias_treino": n_treino, "otimizador": "Adam lr=4e-3 (cosseno), clip=1.0",
                             "perda": "MSE", "normalizacao": {"entrada_px": ESC_IN, "saida_px": ESC_OUT},
                             "predicao": "recursiva"}}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print(f"RESULTADO NO VÍDEO GERADO  (oclusão: {f1 - f0} quadros = {(f1 - f0) * DT:.2f} s)")
    print(f"{'método':7s} {'MAE':>8s} {'RMSE':>8s} {'FDE':>8s} {'desloc.real':>12s} {'desloc.prev':>12s}")
    for m in METODOS:
        r = mets[m]
        print(f"{m:7s} {r['MAE_px']:8.1f} {r['RMSE_px']:8.1f} {r['FDE_px']:8.1f} "
              f"{r['desloc_real_px']:12.1f} {r['desloc_prev_px']:12.1f}")
    print(f"\nMONTE CARLO ({n_mc} trajetórias inéditas)  MAE médio ± desvio (px)")
    for m in METODOS:
        v = [r["MAE_px"] for r in mc_res[m]]
        print(f"{m:7s} {np.mean(v):6.1f} ± {np.std(v):5.1f}   (mediana {np.median(v):.1f})")
    print("=" * 78)
    log(f"Pronto. Arquivos em: {os.path.abspath(a.saida)}")


if __name__ == "__main__":
    main()
