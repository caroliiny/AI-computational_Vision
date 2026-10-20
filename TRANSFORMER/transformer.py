"""
Transformer temporal para prever a posição de um alvo durante oclusões.

Entrada:
  - CSV com frame, visivel e coordenadas x_real/y_real ou x_obs/y_obs.
  - Opcionalmente, um CSV de referência com frame, x_real e y_real para avaliar
    previsões quando a entrada contém apenas observações.

O modelo é treinado apenas com deslocamentos observados antes da primeira
oclusão. Durante a oclusão, prevê recursivamente os deslocamentos seguintes,
sem usar as coordenadas verdadeiras dos quadros ocultos.

Uso:
    python transformer.py entrada.csv [previsoes_transformer.csv]
    python transformer.py observacoes.csv previsoes.csv --referencia trajetoria_real.csv
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


JANELA = 5
DIM_EMBEDDING = 32
NUM_CABECAS = 2
DIM_FEEDFORWARD = 64
EPOCAS = 200
N_MODELOS = 3
RES_MAX = 0.5
SEMENTE = 0


def carregar_csv(caminho, exigir_visibilidade=True):
    """Carrega frames e coordenadas, aceitando CSVs de trajetória ou observações."""
    with open(caminho, "r", newline="", encoding="utf-8-sig") as arquivo:
        leitor = csv.DictReader(arquivo)
        campos = set(leitor.fieldnames or ())
        if "frame" not in campos:
            raise ValueError(f"O CSV {caminho} precisa conter a coluna 'frame'.")

        pares = (("x_obs", "y_obs"), ("x_real", "y_real"))
        par = next(((x, y) for x, y in pares if {x, y} <= campos), None)
        if par is None:
            raise ValueError(
                f"O CSV {caminho} precisa conter x_obs/y_obs ou x_real/y_real."
            )
        if exigir_visibilidade and "visivel" not in campos:
            raise ValueError(f"O CSV {caminho} precisa conter a coluna 'visivel'.")

        linhas = list(leitor)

    if not linhas:
        raise ValueError(f"O CSV {caminho} está vazio.")

    frames = np.array([int(linha["frame"]) for linha in linhas], dtype=int)
    coordenadas = np.full((len(linhas), 2), np.nan, dtype=float)
    visivel = np.ones(len(linhas), dtype=bool)
    for i, linha in enumerate(linhas):
        try:
            coordenadas[i] = [float(linha[par[0]]), float(linha[par[1]])]
        except (TypeError, ValueError):
            pass
        if "visivel" in campos:
            visivel[i] = linha["visivel"].strip().lower() in {"1", "true", "sim"}

    ordem = np.argsort(frames)
    frames, coordenadas, visivel = frames[ordem], coordenadas[ordem], visivel[ordem]
    if len(np.unique(frames)) != len(frames):
        raise ValueError(f"O CSV {caminho} possui números de quadro duplicados.")
    return frames, coordenadas, visivel


def criar_sequencias(deslocamentos, janela):
    X = np.array(
        [deslocamentos[i:i + janela] for i in range(len(deslocamentos) - janela)],
        dtype=np.float32,
    )
    y = np.array(
        [deslocamentos[i + janela] for i in range(len(deslocamentos) - janela)],
        dtype=np.float32,
    )
    return X, y


def criar_modelo():
    entradas = layers.Input(shape=(JANELA, 4))
    x = layers.Dense(DIM_EMBEDDING)(entradas)

    posicoes = tf.range(start=0, limit=JANELA, delta=1)
    codificacao_posicional = layers.Embedding(
        input_dim=JANELA, output_dim=DIM_EMBEDDING
    )(posicoes)
    x = x + codificacao_posicional

    atencao = layers.MultiHeadAttention(
        num_heads=NUM_CABECAS,
        key_dim=DIM_EMBEDDING // NUM_CABECAS,
        dropout=0.1,
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(
        x + layers.Dropout(0.1)(atencao)
    )

    feedforward = layers.Dense(DIM_FEEDFORWARD, activation="relu")(x)
    feedforward = layers.Dense(DIM_EMBEDDING)(feedforward)
    x = layers.LayerNormalization(epsilon=1e-6)(
        x + layers.Dropout(0.1)(feedforward)
    )
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation="relu")(x)
    saida = layers.Dense(2)(x)

    modelo = tf.keras.Model(entradas, saida)
    modelo.compile(optimizer="adam", loss="mse")
    return modelo


def treinar(historico):
    """Treina um ensemble e retorna modelos e parâmetros de normalização."""
    if len(historico) <= JANELA + 1:
        raise ValueError(
            f"São necessários pelo menos {JANELA + 2} quadros visíveis antes "
            "da oclusão para treinar o Transformer."
        )

    deslocamentos = np.diff(historico, axis=0)
    media = deslocamentos.mean(axis=0)
    desvio = deslocamentos.std(axis=0)
    desvio = np.where(desvio == 0, 1.0, desvio)
    normalizados = ((deslocamentos - media) / desvio).astype(np.float32)

    X, y = criar_sequencias(normalizados, JANELA)
    velocidade_base = X.mean(axis=1, keepdims=True)
    X = np.concatenate(
        [X, np.repeat(velocidade_base, JANELA, axis=1)], axis=2
    )
    y = y - velocidade_base[:, 0]
    if not len(X):
        raise ValueError("Não há sequências suficientes para treinar o Transformer.")

    usar_validacao = len(X) >= 20
    modelos = []
    for k in range(N_MODELOS):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEMENTE + k)
        modelo = criar_modelo()
        parada = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if usar_validacao else "loss",
            patience=20,
            restore_best_weights=True,
        )
        modelo.fit(
            X,
            y,
            epochs=EPOCAS,
            batch_size=min(16, len(X)),
            validation_split=0.2 if usar_validacao else 0.0,
            callbacks=[parada],
            shuffle=False,
            verbose=0,
        )
        modelos.append(modelo)
    return modelos, media, desvio


def prever(modelos, media, desvio, historico, quantidade):
    """Prevê recursivamente as próximas posições usando apenas o histórico."""
    deslocamentos = (np.diff(historico[-(JANELA + 1):], axis=0) - media) / desvio
    janela = list(deslocamentos.astype(np.float32))
    velocidade_base = np.mean(janela[-JANELA:], axis=0)
    base_repetida = np.broadcast_to(
        velocidade_base, (1, JANELA, 2)
    ).astype(np.float32)
    posicao = historico[-1].astype(float).copy()
    previsoes = []

    for _ in range(quantidade):
        entrada_velocidade = np.array(janela[-JANELA:], dtype=np.float32)[None, ...]
        entrada = np.concatenate([entrada_velocidade, base_repetida], axis=2)
        correcao = np.mean(
            [modelo(entrada, training=False).numpy()[0] for modelo in modelos],
            axis=0,
        )
        correcao = np.clip(correcao, -RES_MAX, RES_MAX)
        deslocamento = velocidade_base + correcao
        janela.append(deslocamento.astype(np.float32))
        posicao = posicao + deslocamento * desvio + media
        previsoes.append(posicao.copy())

    return np.asarray(previsoes)


def linha_base(historico, quantidade):
    deslocamento = np.diff(historico[-(JANELA + 1):], axis=0).mean(axis=0)
    passos = np.arange(1, quantidade + 1)[:, None]
    return historico[-1] + deslocamento * passos


def erros(referencia, previsto):
    diferencas = referencia - previsto
    distancia = np.sqrt((diferencas ** 2).sum(axis=1))
    return (
        float(distancia.mean()),
        float(np.sqrt(np.mean(diferencas[:, 0] ** 2))),
        float(np.sqrt(np.mean(diferencas[:, 1] ** 2))),
    )


def carregar_referencia(caminho):
    if caminho is None:
        return None
    frames, coordenadas, _ = carregar_csv(caminho, exigir_visibilidade=False)
    return {int(frame): coordenadas[i] for i, frame in enumerate(frames)}


def salvar_resultados(caminho, frames, estimadas, oclusao):
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["frame", "x_transformer", "y_transformer", "oclusao"])
        for frame, posicao, oculto in zip(frames, estimadas, oclusao):
            escritor.writerow([
                int(frame),
                "" if np.isnan(posicao[0]) else f"{posicao[0]:.2f}",
                "" if np.isnan(posicao[1]) else f"{posicao[1]:.2f}",
                int(oculto),
            ])


def salvar_grafico(caminho, referencia, estimadas, oclusao):
    plt.figure(figsize=(7, 5))
    if referencia is not None:
        plt.plot(referencia[:, 0], referencia[:, 1], "k", lw=1, label="Referência")
    plt.plot(estimadas[:, 0], estimadas[:, 1], "b--", label="Transformer")
    plt.plot(
        estimadas[oclusao, 0], estimadas[oclusao, 1],
        "r.", ms=4, label="Transformer na oclusão",
    )
    plt.gca().invert_yaxis()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Trajetória estimada pelo Transformer temporal")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def executar(entrada, saida, caminho_referencia):
    frames, coordenadas_entrada, visivel = carregar_csv(entrada)
    referencia_externa = carregar_referencia(caminho_referencia)

    referencia = coordenadas_entrada.copy()
    if referencia_externa is not None:
        referencia[:] = np.nan
        for i, frame in enumerate(frames):
            if int(frame) in referencia_externa:
                referencia[i] = referencia_externa[int(frame)]

    visivel &= ~np.isnan(coordenadas_entrada).any(axis=1)
    oclusao = ~visivel
    if not oclusao.any():
        raise ValueError("Não há quadros em oclusão neste CSV.")

    primeiro = int(np.argmax(oclusao))
    historico_inicial = coordenadas_entrada[:primeiro].copy()
    modelos, media, desvio = treinar(historico_inicial)

    observacoes = coordenadas_entrada.copy()
    observacoes[~visivel] = np.nan
    estimadas = observacoes.copy()
    base = observacoes.copy()

    i = primeiro
    while i < len(frames):
        if oclusao[i]:
            fim = i
            while fim < len(frames) and oclusao[fim]:
                fim += 1
            estimadas[i:fim] = prever(
                modelos, media, desvio, estimadas[:i], fim - i
            )
            base[i:fim] = linha_base(base[:i], fim - i)
            i = fim
        else:
            i += 1

    salvar_resultados(saida, frames, estimadas, oclusao)
    caminho_grafico = Path(saida).with_name("trajetoria_transformer.png")
    salvar_grafico(caminho_grafico, referencia, estimadas, oclusao)

    avaliar = oclusao & ~np.isnan(referencia).any(axis=1)
    if avaliar.any():
        media_erro, rmse_x, rmse_y = erros(referencia[avaliar], estimadas[avaliar])
        print(
            f"Transformer temporal -> Erro médio: {media_erro:.2f} | "
            f"RMSE X: {rmse_x:.2f} | RMSE Y: {rmse_y:.2f}"
        )
        media_base, rmse_base_x, rmse_base_y = erros(referencia[avaliar], base[avaliar])
        print(
            f"Velocidade constante -> Erro médio: {media_base:.2f} | "
            f"RMSE X: {rmse_base_x:.2f} | RMSE Y: {rmse_base_y:.2f}"
        )
    else:
        print("Referência ausente nos quadros ocluídos; métricas não calculadas.")

    print(f"Previsões salvas em: {saida}")
    print(f"Gráfico salvo em: {caminho_grafico}")
    print(
        f"Quadros em oclusão: {int(oclusao.sum())} | "
        f"ensemble: {N_MODELOS} Transformers"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrada", help="CSV da trajetória ou das observações")
    parser.add_argument(
        "saida", nargs="?", default="previsoes_transformer.csv",
        help="CSV de saída (padrão: previsoes_transformer.csv)",
    )
    parser.add_argument(
        "--referencia",
        help="CSV opcional com frame,x_real,y_real para calcular métricas",
    )
    args = parser.parse_args()
    executar(args.entrada, args.saida, args.referencia)


if __name__ == "__main__":
    main()
