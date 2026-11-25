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

Com o `(pose_env)` ativado, instale as bibliotecas necessárias:

```powershell
python -m pip install --upgrade pip
pip install -r ..\requirements.txt
```

---

## 📂 Estrutura da Pasta

Todos os scripts ficam em `teste_video_real\` e devem ser executados **de dentro dessa pasta**
(`cd teste_video_real`). Eles usam `coco17.py` e `EKF\ekf.py` da raiz do repositório.

```text
teste_video_real/
├── processar_pose.py          <-- Extração de keypoints (YOLOv8-pose)
├── modelos_pose.py            <-- EKF, LSTM, GRU, Transformer, T-GNN (COCO-17)
├── avaliar_pose.py            <-- Erro em pixels com oclusão artificial
├── comparar_modelos.py        <-- Jitter do EKF
├── testar_redes_temporais.py  <-- Jitter de LSTM e GRU
├── testar_transformer_tgnn.py <-- Jitter de Transformer e T-GNN
└── renderizar_video.py        <-- Vídeo final anotado
```

---

## 🎥 Como preparar um vídeo real

* **Uma pessoa em foco** (o rastreio segue uma só; com várias pessoas pode trocar de alvo).
* **Câmera fixa** (o modelo trabalha em pixels; movimento de câmera vira "movimento da pessoa").
* **Corpo inteiro no quadro** na maior parte do tempo: o treino das redes usa só janelas de 10 quadros com as
  17 juntas visíveis.
* **Pelo menos ~300 quadros** e **com oclusões naturais** (pilar, objeto, saída parcial do quadro).
* MP4 H.264, 720p–1080p, 25–30 fps constantes (converta se necessário:
  `ffmpeg -i in.mp4 -r 30 -c:v libx264 out.mp4`).

---

## 🚀 Fluxo de Execução (um vídeo qualquer)

Cada script recebe o vídeo/arquivo como argumento e nomeia a saída a partir do nome do vídeo,
então **vídeos diferentes não se sobrescrevem**. Exemplo com `corrida.mp4` (sem argumento, usa `video_teste.mp4`):

```powershell
cd teste_video_real

# 1. Extrai o esqueleto -> corrida_pose.npz  (imprime taxa de oclusão e avisos de qualidade)
python processar_pose.py corrida.mp4

# 2. Métodos sobre o vídeo real (jitter) -> corrida_pose_*.png
python comparar_modelos.py corrida_pose.npz
python testar_redes_temporais.py corrida_pose.npz
python testar_transformer_tgnn.py corrida_pose.npz

# 3. Erro em pixels com oclusão artificial -> corrida_pose_metricas.csv
python avaliar_pose.py corrida_pose.npz

# 4. Vídeo anotado -> corrida_anotado.mp4
python renderizar_video.py corrida.mp4
```

Opções úteis: `processar_pose.py --conf 0.45` (limiar de confiança que define "ocluído"),
`--modelo yolov8n-pose.pt` (muito mais rápido em CPU, menos preciso), `avaliar_pose.py --epocas 150`,
e `--saida`/`--png`/`--pose` para escolher os nomes de saída.

> **Padrão COCO-17:** todos os métodos leem `keypoints (T,17,2)` + `visibility (T,17)`
> (definido em `coco17.py`). O vídeo sintético (`01_video/pose_data.npz`) usa o mesmo formato, e o EKF é o de
> `EKF/ekf.py` (um filtro por junta).

### Como interpretar

* Vídeo real **não tem gabarito**. O erro em pixels de `avaliar_pose.py` compara cada método com a
  *própria detecção do YOLO* escondida artificialmente; mede a capacidade de reconstruir o que o YOLO viu,
  não a posição verdadeira.
* "Ocluído" = junta com confiança do YOLO abaixo de `--conf`; esse limiar influencia os resultados.
* Cores no vídeo anotado: 🟢 articulação visível (YOLO), 🔴 ocluída (estimada pelo EKF).
* Os arquivos `*.npz`, `*.mp4`, `*.png` e `*.pt` são ignorados pelo git; guarde os vídeos reais fora do repositório.

---

## ❓ Resolução de Problemas Comuns

1. **`Nenhuma janela com as 17 juntas visíveis para treinar`:** a pessoa nunca aparece inteira por 11 quadros
   seguidos. Use um vídeo com corpo inteiro ou reduza `--conf` em `processar_pose.py`.
2. **`Vídeo curto demais para a oclusão artificial`:** o vídeo tem menos de 28 quadros.
3. **`source : O termo 'source' não é reconhecido`:** use `.\pose_env\Scripts\activate` no Windows.
4. **Curvas caindo para valores negativos:** execução das redes sem o ciclo de treino; use as versões atuais
   dos scripts.
5. **Lento em CPU:** use `--modelo yolov8n-pose.pt` na extração e `--epocas` menor na avaliação.
