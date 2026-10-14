import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import csv
import threading
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def carregar_trajetoria(caminho):
    dados = []

    with open(caminho, "r", newline="", encoding="utf-8-sig") as arquivo:
        leitor = csv.DictReader(arquivo)

        for linha in leitor:
            dados.append({
                "frame": int(linha["frame"]),
                "x": float(linha["x_real"]),
                "y": float(linha["y_real"]),
                "visivel": int(linha.get("visivel", 1))
            })

    return dados

def criar_sequencias(posicoes, tamanho_janela=5):
    entradas = []
    saidas = []

    for i in range(len(posicoes) - tamanho_janela):
        entradas.append(
            posicoes[i:i + tamanho_janela]
        )

        saidas.append(
            posicoes[i + tamanho_janela]
        )

    return np.array(entradas), np.array(saidas)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def criar_modelo(tamanho_janela):
    modelo = Sequential([
        LSTM(
            32,
            input_shape=(tamanho_janela, 2)
        ),
        Dense(16, activation="relu"),
        Dense(2)
    ])

    modelo.compile(
        optimizer="adam",
        loss="mse"
    )

    return modelo


def treinar_modelo(posicoes_visiveis, tamanho_janela=5, epocas=100):
    posicoes_visiveis = np.asarray(
        posicoes_visiveis, dtype=np.float32
    )

    minimo = posicoes_visiveis.min(axis=0)
    amplitude = posicoes_visiveis.max(axis=0) - minimo
    amplitude = np.where(amplitude == 0, 1.0, amplitude)

    posicoes_norm = (posicoes_visiveis - minimo) / amplitude

    X, y = criar_sequencias(
        posicoes_norm,
        tamanho_janela
    )

    if len(X) == 0:
        raise ValueError(
            "Não há dados suficientes para treinar a LSTM."
        )

    modelo = criar_modelo(tamanho_janela)

    modelo.fit(
        X,
        y,
        epochs=epocas,
        batch_size=min(16, len(X)),
        verbose=0,
        shuffle=False
    )

    return modelo, minimo, amplitude


def prever_oclusao(modelo, historico, quantidade, tamanho_janela=5,
                   minimo=None, amplitude=None):
    historico = np.asarray(historico, dtype=np.float32)

    if minimo is None:
        minimo = historico.min(axis=0)

    if amplitude is None:
        amplitude = historico.max(axis=0) - minimo

    amplitude = np.where(amplitude == 0, 1.0, amplitude)

    sequencia = (historico - minimo) / amplitude
    sequencia = list(sequencia[-tamanho_janela:])
    previsoes = []

    for _ in range(quantidade):
        entrada = np.array(
            [sequencia[-tamanho_janela:]],
            dtype=np.float32
        )

        proxima = modelo.predict(entrada, verbose=0)[0]
        previsoes.append(proxima)
        sequencia.append(proxima)

    previsoes = np.asarray(previsoes, dtype=np.float32)

    return previsoes * amplitude + minimo

def calcular_erros(real, previsto):
    diferencas = real - previsto

    erros_distancia = np.sqrt(
        np.sum(diferencas ** 2, axis=1)
    )

    erro_medio = np.mean(erros_distancia)

    rmse_x = np.sqrt(
        np.mean(diferencas[:, 0] ** 2)
    )

    rmse_y = np.sqrt(
        np.mean(diferencas[:, 1] ** 2)
    )

    return erro_medio, rmse_x, rmse_y

class LSTMApp:
    def __init__(self, raiz):
        self.raiz = raiz
        self.raiz.title("Previsão de Movimento - LSTM")
        self.raiz.geometry("900x650")

        self.dados = None
        self.caminho_csv = None

        painel = ttk.Frame(raiz, padding=10)
        painel.pack(fill="x")

        ttk.Button(
            painel,
            text="Carregar trajetória CSV",
            command=self.carregar_csv
        ).grid(row=0, column=0, padx=5, pady=5)

        self.lbl_arquivo = ttk.Label(
            painel,
            text="Nenhum arquivo selecionado"
        )
        self.lbl_arquivo.grid(row=0, column=1, padx=5)

        ttk.Label(
            painel, text="Primeiro quadro da oclusão:"
        ).grid(row=1, column=0, sticky="w")

        self.inicio = ttk.Entry(painel, width=10)
        self.inicio.insert(0, "80")
        self.inicio.grid(row=1, column=1, sticky="w")

        ttk.Label(
            painel, text="Último quadro da oclusão:"
        ).grid(row=2, column=0, sticky="w")

        self.fim = ttk.Entry(painel, width=10)
        self.fim.insert(0, "109")
        self.fim.grid(row=2, column=1, sticky="w")

        ttk.Button(
            painel,
            text="Treinar LSTM e prever",
            command=self.executar
        ).grid(row=3, column=0, padx=5, pady=10)

        self.lbl_resultado = ttk.Label(
            raiz,
            text="Carregue o CSV para começar.",
            padding=10
        )
        self.lbl_resultado.pack()

        self.figura = Figure(figsize=(8, 4), dpi=100)
        self.grafico = self.figura.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(
            self.figura, master=raiz
        )
        self.canvas.get_tk_widget().pack(
            fill="both", expand=True, padx=10, pady=10
        )

    def carregar_csv(self):
        caminho = filedialog.askopenfilename(
            title="Selecione a trajetória",
            filetypes=[("Arquivos CSV", "*.csv")]
        )

        if not caminho:
            return

        try:
            dados = carregar_trajetoria(caminho)

            if len(dados) < 15:
                raise ValueError(
                    "O CSV precisa ter pelo menos 15 quadros."
                )

            self.dados = dados
            self.caminho_csv = caminho
            self.lbl_arquivo.config(
                text=f"{len(dados)} quadros carregados"
            )

            self.lbl_resultado.config(
                text="CSV carregado. Configure a oclusão e treine."
            )

        except Exception as erro:
            messagebox.showerror(
                "Erro ao carregar CSV", str(erro)
            )

    def executar(self):
        if self.dados is None:
            messagebox.showwarning(
                "Atenção", "Carregue o arquivo CSV primeiro."
            )
            return

        try:
            inicio = int(self.inicio.get())
            fim = int(self.fim.get())

            frames = np.array([
                d["frame"] for d in self.dados
            ])
            posicoes = np.array([
                [d["x"], d["y"]] for d in self.dados
            ], dtype=np.float32)

            indices_oclusao = np.where(
                (frames >= inicio) & (frames <= fim)
            )[0]

            if len(indices_oclusao) == 0:
                raise ValueError(
                    "Nenhum quadro encontrado nesse intervalo."
                )

            primeiro = indices_oclusao[0]

            # O histórico de treinamento termina antes da oclusão.
            historico = posicoes[:primeiro]

            janela = 5

            if len(historico) <= janela:
                raise ValueError(
                    "Não há histórico suficiente antes da oclusão."
                )

            self.lbl_resultado.config(
                text="Treinando a LSTM... aguarde."
            )
            self.raiz.update_idletasks()

            modelo, minimo, amplitude = treinar_modelo(
                historico,
                tamanho_janela=janela,
                epocas=100
            )

            previsoes = prever_oclusao(
                modelo,
                historico,
                quantidade=len(indices_oclusao),
                tamanho_janela=janela,
                minimo=minimo,
                amplitude=amplitude
            )

            reais = posicoes[indices_oclusao]

            erro_medio, rmse_x, rmse_y = calcular_erros(
                reais, previsoes
            )

            self.grafico.clear()
            self.grafico.plot(
                posicoes[:, 0], posicoes[:, 1],
                label="Trajetória verdadeira",
                color="blue"
            )
            self.grafico.plot(
                previsoes[:, 0], previsoes[:, 1],
                "o--", label="Previsão LSTM",
                color="red"
            )

            self.grafico.axvline(
                x=posicoes[primeiro, 0],
                linestyle=":",
                color="gray"
            )
            self.grafico.set_xlabel("X")
            self.grafico.set_ylabel("Y")
            self.grafico.set_title(
                "Trajetória verdadeira e previsão na oclusão"
            )
            self.grafico.legend()
            self.grafico.grid(True)
            self.canvas.draw()

            self.lbl_resultado.config(
                text=(
                    f"Erro médio: {erro_medio:.2f} | "
                    f"RMSE X: {rmse_x:.2f} | "
                    f"RMSE Y: {rmse_y:.2f}"
                )
            )

            caminho_saida = filedialog.asksaveasfilename(
                title="Salvar previsões",
                defaultextension=".csv",
                initialfile="previsoes_lstm.csv",
                filetypes=[("Arquivos CSV", "*.csv")]
            )

            if caminho_saida:
                with open(
                    caminho_saida, "w",
                    newline="", encoding="utf-8"
                ) as arquivo:
                    escritor = csv.writer(arquivo)
                    escritor.writerow([
                        "frame", "x_lstm", "y_lstm"
                    ])

                    for indice, predicao in zip(
                        indices_oclusao, previsoes
                    ):
                        escritor.writerow([
                            int(frames[indice]),
                            float(predicao[0]),
                            float(predicao[1])
                        ])

                messagebox.showinfo(
                    "Concluído",
                    "Previsões exportadas com sucesso."
                )

        except Exception as erro:
            messagebox.showerror(
                "Erro no treinamento", str(erro)
            )


if __name__ == "__main__":
    raiz = tk.Tk()
    app = LSTMApp(raiz)
    raiz.mainloop()
