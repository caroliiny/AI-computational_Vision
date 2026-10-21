"""Avaliação de EKF, LSTM, GRU e Transformer durante a oclusão."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


METODOS = {
    "EKF": ("02_ekf/previsoes_ekf.csv", "x_ekf", "y_ekf"),
    "LSTM": ("03_lstm/previsoes_lstm.csv", "x_lstm", "y_lstm"),
    "GRU": ("03_gru/previsoes_gru.csv", "x_gru", "y_gru"),
    "Transformer": (
        "03_transformer/previsoes_transformer.csv",
        "x_transformer",
        "y_transformer",
    ),
}

CORES = ["#d97706", "#0284c7", "#16a34a", "#9333ea"]


def ler_csv(caminho, colunas):
    """Carrega as linhas indexadas pelo número do quadro."""
    with caminho.open(newline="", encoding="utf-8-sig") as arquivo:
        leitor = csv.DictReader(arquivo)

        faltantes = set(colunas) - set(leitor.fieldnames or [])
        if faltantes:
            raise ValueError(
                f"{caminho.name}: colunas ausentes: {sorted(faltantes)}"
            )

        dados = {}
        for linha in leitor:
            frame = int(linha["frame"])
            if frame in dados:
                raise ValueError(
                    f"{caminho.name}: quadro duplicado: {frame}"
                )
            dados[frame] = linha

    if not dados:
        raise ValueError(f"Arquivo vazio: {caminho}")

    return dados


def obter_pontos(dados, frames, coluna_x, coluna_y, nome):
    """Exige coordenadas válidas para todos os quadros avaliados."""
    pontos = []

    for frame in frames:
        if frame not in dados:
            raise ValueError(f"{nome}: falta o quadro {frame}.")

        linha = dados[frame]
        try:
            pontos.append([
                float(linha[coluna_x]),
                float(linha[coluna_y]),
            ])
        except (ValueError, TypeError):
            raise ValueError(
                f"{nome}: coordenada inválida no quadro {frame}."
            ) from None

    pontos = np.asarray(pontos, dtype=float)

    if not np.isfinite(pontos).all():
        raise ValueError(f"{nome}: coordenadas contêm NaN ou infinito.")

    return pontos


def salvar_csv(caminho, cabecalho, linhas):
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(cabecalho)
        escritor.writerows(linhas)


def avaliar(pasta):
    referencia = ler_csv(
        pasta / "01_video/trajetoria_real.csv",
        ["frame", "x_real", "y_real", "visivel"],
    )

    # A referência define os mesmos quadros ocultos para todos os métodos.
    frames = []
    for frame, linha in referencia.items():
        visivel = linha["visivel"].strip().lower()

        if visivel not in {"0", "1", "false", "true"}:
            raise ValueError(
                f"Visibilidade inválida no quadro {frame}."
            )

        if visivel in {"0", "false"}:
            frames.append(frame)

    frames.sort()

    if not frames:
        raise ValueError("Não há quadros ocultos na referência.")

    real = obter_pontos(
        referencia, frames, "x_real", "y_real", "Referência"
    )

    previsoes = {}
    erros = {}
    tabela = []

    for nome, (arquivo, coluna_x, coluna_y) in METODOS.items():
        dados = ler_csv(
            pasta / arquivo, ["frame", coluna_x, coluna_y]
        )

        previsto = obter_pontos(
            dados, frames, coluna_x, coluna_y, nome
        )

        # Distância euclidiana 2D em cada quadro.
        erro = np.linalg.norm(previsto - real, axis=1)

        previsoes[nome] = previsto
        erros[nome] = erro

        tabela.append([
            nome,
            len(frames),
            float(erro.mean()),
            float(np.sqrt(np.mean(erro ** 2))),
            float(erro[-1]),
            float(erro.max()),
        ])

    saida = pasta / "04_avaliacao"
    saida.mkdir(parents=True, exist_ok=True)

    salvar_csv(
        saida / "comparacao_final.csv",
        [
            "metodo",
            "quadros_ocultos",
            "erro_medio_px",
            "rmse_posicao_px",
            "erro_final_px",
            "erro_maximo_px",
        ],
        tabela,
    )

    salvar_csv(
        saida / "erros_comparacao_final.csv",
        ["frame", "x_real", "y_real"]
        + [f"erro_{nome.lower()}_px" for nome in METODOS],
        [
            [frame, *real[i]]
            + [erros[nome][i] for nome in METODOS]
            for i, frame in enumerate(frames)
        ],
    )

    # Se houver oclusões separadas, não liga os intervalos no gráfico.
    cortes = np.flatnonzero(np.diff(frames) > 1) + 1

    def separar(valores):
        return np.insert(
            np.asarray(valores, dtype=float),
            cortes,
            np.nan,
            axis=0,
        )

    fig, eixos = plt.subplots(2, 2, figsize=(13, 9))

    ax = eixos[0, 0]
    pontos = separar(real)
    ax.plot(
        pontos[:, 0], pontos[:, 1],
        color="black", linewidth=2.5, label="Referência",
    )

    for (nome, previsto), cor in zip(previsoes.items(), CORES):
        pontos = separar(previsto)
        ax.plot(
            pontos[:, 0], pontos[:, 1], color=cor, label=nome
        )

    ax.invert_yaxis()
    ax.set(
        title="Trajetórias durante a oclusão",
        xlabel="X (pixels)",
        ylabel="Y (pixels)",
    )
    ax.legend()
    ax.grid(alpha=0.3)

    ax = eixos[0, 1]
    for (nome, erro), cor in zip(erros.items(), CORES):
        ax.plot(
            separar(frames), separar(erro),
            color=cor, label=nome,
        )

    ax.set(
        title="Erro de posição por quadro",
        xlabel="Quadro",
        ylabel="Distância à referência (pixels)",
    )
    ax.legend()
    ax.grid(alpha=0.3)

    for ax, coluna, titulo in [
        (eixos[1, 0], 2, "Erro médio de posição"),
        (eixos[1, 1], 3, "RMSE de posição"),
    ]:
        valores = [linha[coluna] for linha in tabela]
        barras = ax.bar(list(METODOS), valores, color=CORES)
        ax.bar_label(barras, fmt="%.2f", padding=3)
        ax.set(title=titulo, ylabel="Pixels")
        ax.set_ylim(0, max(valores) * 1.2 + 1)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(saida / "comparacao_final.png", dpi=180)
    plt.close(fig)

    print(f"\nAvaliação em {len(frames)} quadros ocultos.")
    print(f"{'Método':<14} {'Erro médio':>12} {'RMSE':>12}")

    for linha in tabela:
        print(
            f"{linha[0]:<14} {linha[2]:>12.2f} {linha[3]:>12.2f}"
        )

    print(
        "\nLimitação: os métodos têm protocolos de treinamento diferentes."
    )
    print("Esta avaliação usa somente o vídeo sintético desta execução.")
    print(f"Resultados salvos em: {saida.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "execucao", type=Path, help="Pasta com os resultados da execução."
    )
    args = parser.parse_args()

    try:
        avaliar(args.execucao)
    except (OSError, ValueError) as erro:
        parser.exit(1, f"Erro: {erro}\n")


if __name__ == "__main__":
    main()