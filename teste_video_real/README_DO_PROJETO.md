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
├── renderizar_video.py        <-- Vídeo final anotado
│
│   # fluxo de oclusão (ver seção "Fluxo de previsão sob oclusão" abaixo)
├── gerar_oclusoes.py          <-- Vídeo com pilares/muros desenhados
├── protocolo.py               <-- Regras comuns de treino/teste (sem vazamento)
├── treinar_previsao.py        <-- Treino das redes (rollout de 20 quadros)
├── avaliar_previsao.py        <-- Erro a 5/10/20 quadros dos 5 métodos
├── visualizar_previsao.py     <-- Gráficos e vídeo de previsões
├── executar_oclusao.ps1       <-- Roda tudo
└── videos_originais/          <-- coloque aqui os vídeos limpos
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

---

## 🧱 Fluxo de previsão sob oclusão (vídeo limpo → oclusão artificial → previsão 5/10/20 quadros)

Este fluxo usa **vídeos reais limpos** como gabarito. Os scripts antigos acima continuam funcionando; os novos
são independentes e reaproveitam `modelos_pose.py` (EKF, LSTM, GRU, Transformer, T-GNN) e `coco17.py`.

### Onde colocar os vídeos

Copie os vídeos originais (`.mp4`, uma pessoa, câmera fixa, sem oclusões) em **`teste_video_real\videos_originais\`**.
Ela é ignorada pelo git (`*.mp4`). Quanto mais longo o vídeo, mais oclusões aparecem no trecho de teste:
use pelo menos ~40 s (≈ 1200 quadros); abaixo de ~600 quadros o teste fica com poucos eventos.

### Como executar

Tudo de uma vez (cada vídeo gera sua própria pasta):

```powershell
cd teste_video_real
.\executar_oclusao.ps1                      # todos os vídeos de videos_originais\
.\executar_oclusao.ps1 -Videos corrida.mp4  # só um
.\executar_oclusao.ps1 -Rapido              # teste rápido do fluxo (YOLO leve, 10 épocas)
```

Ou etapa por etapa (`$p` = pasta de trabalho do vídeo):

```powershell
$p = "resultados_oclusao\corrida"; mkdir $p
# 1. pose de referência (vídeo original, COCO-17)       -> $p\referencia_pose.npz
python processar_pose.py videos_originais\corrida.mp4 --saida $p\referencia_pose.npz
# 2. vídeo com oclusões artificiais (mesmos quadros/FPS) -> $p\video_oclusao.mp4, $p\cenarios.json
python gerar_oclusoes.py videos_originais\corrida.mp4 $p
# 3. pose detectada no vídeo com oclusão (entrada dos modelos) -> $p\oclusao_pose.npz
python processar_pose.py $p\video_oclusao.mp4 --saida $p\oclusao_pose.npz
# 4. treino de LSTM, GRU, Transformer, T-GNN             -> $p\modelos\*.pt, $p\treino.json
python treinar_previsao.py $p --epocas 100
# 5. avaliação dos 5 métodos a 5, 10 e 20 quadros        -> $p\metricas_*.csv
python avaliar_previsao.py $p
# 6. gráficos e vídeo                                    -> $p\grafico_*.png, $p\video_previsoes.mp4
python visualizar_previsao.py $p
```

### O que cada etapa garante

| Etapa | Regra |
|---|---|
| 2 | O vídeo com oclusão é escrito quadro a quadro a partir do original; o script confere que nº de quadros, FPS e resolução são idênticos e aborta se não forem. |
| 2 | Seis tipos de obstáculo, com posição/largura/altura sorteadas (`--semente`): `pilar_estreito`, `muro_baixo_pernas` (**parcial**); `pilar_largo`, `muro_alto` (**grande**); `bloco_quase_total` (**quase_total**); `bloco_total` (**total**). Cada oclusão dura 1–2 s e é precedida de 25+ quadros livres. |
| 3–5 | Os modelos só recebem a pose detectada **no vídeo com oclusão**, até o quadro `t0` (última detecção da junta). A pose do original só é alvo (treino) ou gabarito (avaliação). |
| 4–5 | Treino usa o início do vídeo (60%, `--frac-treino`); a avaliação usa só o final, sem sobreposição de alvos. |
| 5 | Previsão a partir de `t0`: prevê `t0+1…t0+20` sem ver nenhum quadro posterior. O EKF só faz a etapa de predição enquanto a junta está oculta. Avalia-se o erro em `t0+5`, `t0+10`, `t0+20`, apenas nas oclusões que duram pelo menos esse tempo. |
| 5 | Os cinco métodos (e a linha de base "Último valor") usam os mesmos eventos, a mesma entrada e a mesma métrica. Redes: mesma arquitetura de `modelos_pose.py`, semente, épocas e dados. |

### Onde estão os resultados (`resultados_oclusao\<video>\`)

```text
referencia_pose.npz   poses do vídeo original (referência)
video_oclusao.mp4     vídeo com obstáculos (sincronizado com o original)
cenarios.json         tipo, categoria, quadros, retângulo e juntas cobertas de cada oclusão
oclusao_pose.npz      pose detectada no vídeo com oclusão (entrada dos modelos)
modelos\*.pt, treino.json
metricas_horizonte.csv   erro médio / RMSE / mediana (px) por método e horizonte      <- tabela principal
metricas_categoria.csv   o mesmo, por categoria (parcial, grande, quase_total, total)
metricas_cenario.csv     o mesmo, por tipo de obstáculo
erros_eventos.csv        um erro por (método, evento, horizonte, junta)
grafico_horizonte.png, grafico_categoria.png, grafico_trajetorias.png
video_previsoes.mp4      anel verde = referência; pontos coloridos = previsão de cada método
```

### Limitações a ter em mente

* A "oclusão" que os modelos veem é a **ausência de detecção do YOLO** (confiança < `--conf`) no vídeo com
  obstáculo. Se o YOLO ainda detectar uma junta atrás do obstáculo, ela conta como observada.
* O obstáculo aparece de uma vez (a pessoa não "entra" atrás dele), e a referência vem do YOLO no vídeo limpo,
  não de anotação manual: o erro mede a concordância com o YOLO no original.
* Com um único vídeo, treino e teste são trechos diferentes do mesmo vídeo (mesma pessoa e cenário).
  Resultados mais confiáveis pedem vídeos mais longos ou diferentes.
