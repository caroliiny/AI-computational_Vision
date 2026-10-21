# Benchmark de Estimativa de Pose Sob Oclusão (Sim-to-Real)

Este repositório contém o pipeline completo para extração de articulações humanas e avaliação comparativa de 5 técnicas de predição sob oclusão em vídeo: **EKF (Filtro Cinemático)**, **LSTM**, **GRU**, **Transformer Temporal** e **T-GNN (Graph Neural Network)**.

---

## 📋 Pré-requisitos

* **Python 3.10 ou 3.11** instalado no sistema.
* **VS Code** (ou terminal de preferência).
* Placa gráfica dedicada (opcional, mas recomendado) ou CPU.

---

## 🛠️ Passo a Passo para Configuração do Ambiente

Siga os passos abaixo sequencialmente no terminal do **PowerShell** (no Windows) dentro da pasta do projeto.

### 1. Permitir Execução de Scripts no PowerShell
No Windows, execute o comando abaixo para autorizar a ativação de ambientes virtuais na sessão atual:

```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
```

---

### 2. Criar e Ativar o Ambiente Virtual (`pose_env`)

No diretório raiz do projeto, execute:

```powershell
# Criar o ambiente isolado
python -m venv pose_env

# Ativar o ambiente virtual no Windows
.\pose_env\Scripts\activate
```

> ⚠️ **Nota:** Após a ativação, o prefixo `(pose_env)` deve aparecer em verde no início da linha de comandos do terminal.

---

### 3. Habilitar Suporte a Caminhos Longos no Windows (Evita `WinError 206`)
Para evitar erros ao instalar pacotes profundos do PyTorch, abra o PowerShell **como Administrador** e rode:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

---

### 4. Instalar as Dependências do Projeto

Com o `(pose_env)` ativado, atualize o `pip` e instale as bibliotecas necessárias:

```powershell
python -m pip install --upgrade pip
pip install ultralytics opencv-python numpy matplotlib torch
```

---

## 📂 Estrutura de Pastas do Projeto

Certifique-se de que os ficheiros estão organizados conforme a estrutura abaixo antes de iniciar os testes:

```text
AI-computational_Vision/
└── modelagem de oclusão/
    └── código/
        └── teste_video_real/
            ├── video_teste.mp4                 <-- Vídeo original renomeado
            ├── processar_pose.py               <-- Script de extração (YOLOv8)
            ├── comparar_modelos.py             <-- Teste da baseline EKF
            ├── testar_redes_temporais.py       <-- Teste LSTM e GRU
            ├── testar_transformer_tgnn.py      <-- Teste Transformer e T-GNN
            └── renderizar_video.py             <-- Geração do vídeo final anotado
```

---

## 🚀 Fluxo de Execução dos Scripts

Acesse a subpasta onde estão os scripts e o vídeo antes de rodar os comandos:

```powershell
cd "modelagem de oclusão\código\teste_video_real"
```

### Passo 1: Extrair Keypoints do Vídeo
Lê o vídeo `video_teste.mp4`, aplica o detector YOLOv8x-Pose e salva os dados estruturados no arquivo `pose_data.npz`.

```powershell
python processar_pose.py
```
* **Saída esperada:** Ficheiro `pose_data.npz` gerado com os 754 quadros processados.

---

### Passo 2: Avaliar a Baseline Cinemática (EKF)
Calcula a predição cinemática estabilizada e calcula as métricas de ruído (*jittering*).

```powershell
python comparar_modelos.py
```
* **Saída esperada:** Gráfico `trajetoria_pulso.png` e métricas impressas no terminal.

---

### Passo 3: Avaliar Redes Recorrentes (LSTM vs GRU)
Treina as redes auto-regressivas em quadros visíveis e avalia o comportamento durante as oclusões.

```powershell
python testar_redes_temporais.py
```
* **Saída esperada:** Gráfico `comparativo_redes_temporais_treinadas.png`.

---

### Passo 4: Avaliar Modelos Avançados (Transformer vs T-GNN)
Treina o mecanismo de autoatenção e a rede em grafo espacial baseada na topologia anatómica do esqueleto COCO.

```powershell
python testar_transformer_tgnn.py
```
* **Saída esperada:** Gráfico `comparativo_transformer_tgnn.png` e resumo final de métricas.

---

### Passo 5: Renderizar Vídeo Final Anotado
Gera o vídeo com a sobreposição das articulações em tempo real.

```powershell
python renderizar_video.py
```
* **Saída esperada:** Vídeo `resultado_visual_benchmark.mp4` salvo na pasta.
  * 🟢 **Verde:** Articulação visível (detetada pelo YOLO).
  * 🔴 **Vermelho:** Articulação ocluída (estimada pelo modelo).

---

## ❓ Resolução de Problemas Comuns

1. **`Errno 2: No such file or directory` ao rodar um script:**
   * **Causa:** O terminal está na pasta raiz e não dentro da subpasta `teste_video_real`.
   * **Solução:** Execute `cd "modelagem de oclusão\código\teste_video_real"` no terminal.

2. **`source : O termo 'source' não é reconhecido`:**
   * **Causa:** O comando `source` é para Linux/macOS.
   * **Solução:** No Windows utilize `.\pose_env\Scripts\activate`.

3. **Curvas caindo para valores negativos nos gráficos:**
   * **Causa:** Execução das redes sem o ciclo de treino prévio (`train_loop`).
   * **Solução:** Certifique-se de executar as versões atualizadas dos scripts `testar_redes_temporais.py` e `testar_transformer_tgnn.py`.