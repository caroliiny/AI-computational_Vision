# AI-computational_Vision

Projeto dos seminários sobre predição de movimento durante oclusões.

## Começar do zero

Use PowerShell na pasta raiz do repositório. Para começar com um ambiente
Python limpo, crie e ative um ambiente virtual e instale as dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se a ativação do ambiente virtual for bloqueada pelo PowerShell, permita
scripts apenas nesta janela e tente novamente:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Para executar o fluxo completo e gerar vídeos, resultados EKF/LSTM/GRU,
avaliação e previsões do Transformer:

```powershell
.\executar.ps1
```

Cada execução cria uma pasta nova em `resultados\execucao_AAAAMMDD_HHMMSS`.
Assim, execuções anteriores não são sobrescritas. Para uma verificação rápida
do fluxo:

```powershell
.\executar.ps1 -Rapido
```

O modo rápido reduz a quantidade de treino e de cenários Monte Carlo; use o
modo normal para gerar os resultados da apresentação. O script informa o
caminho da pasta de saída no final. Não é necessário executar os scripts de
EKF, GRU ou Transformer separadamente para esse fluxo.

## Abrir vídeos, gráficos e resultados

Depois de executar `.\executar.ps1` ou `.\executar.ps1 -Rapido`, localize a
execução mais recente:

```powershell
$run = Get-ChildItem .\resultados -Directory |
    Sort-Object Name -Descending |
    Select-Object -First 1
$run.FullName
```

Abra os vídeos no reprodutor padrão do Windows:

```powershell
Start-Process (Join-Path $run.FullName "01_video\video_sintetico.mp4")
Start-Process (Join-Path $run.FullName "01_video\video_sem_oclusao.mp4")
Start-Process (Join-Path $run.FullName "04_avaliacao\video_comparacao.mp4")
```

`video_sintetico.mp4` mostra a oclusão, `video_sem_oclusao.mp4` mostra a
mesma trajetória sem ocultação, e `video_comparacao.mp4` mostra as previsões
do EKF, LSTM e GRU durante a oclusão.

Abra as visualizações e tabelas:

```powershell
Start-Process (Join-Path $run.FullName "04_avaliacao\graficos.png")
Start-Process (Join-Path $run.FullName "03_transformer\trajetoria_transformer.png")
Import-Csv (Join-Path $run.FullName "04_avaliacao\metricas.csv") |
    Format-Table -AutoSize
Import-Csv (Join-Path $run.FullName "04_avaliacao\metricas_monte_carlo.csv") |
    Format-Table -AutoSize
```

O primeiro gráfico compara EKF, LSTM e GRU; o segundo é a trajetória do
Transformer. As tabelas são os resultados do vídeo gerado e do Monte Carlo,
respectivamente. Para ver todos os arquivos da execução no Explorer:

```powershell
explorer.exe $run.FullName
```

As variáveis `$run` e os comandos acima devem ser executados na mesma janela
do PowerShell. A seleção pela pasta mais recente também funciona após várias
execuções.

Se o PowerShell bloquear a execução do script, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\executar.ps1 -Rapido
```

## O que é gerado

```text
resultados\
└── execucao_<data_hora>\
    ├── 01_video\          vídeos, trajetória real e observações
    ├── 02_ekf\            previsões do EKF
    ├── 03_lstm\           previsões da LSTM
    ├── 03_gru\            previsões da GRU
    ├── 03_transformer\    previsões e gráfico do Transformer
    └── 04_avaliacao\      métricas, tabelas, gráfico e vídeo comparativo
```

O pipeline principal gera e compara EKF, LSTM e GRU. O script de execução
roda o Transformer sobre as observações da mesma execução e usa a trajetória
real apenas para calcular as métricas. A tabela/gráfico comparativo integrado
do pipeline ainda não inclui o Transformer.

## Código

- `modelagem de oclusão\código\rastreamento_oclusao.py`: pipeline principal
  de cenário sintético, vídeos, EKF, LSTM, GRU e avaliação.
- `TRANSFORMER\transformer.py`: treino e previsão do Transformer temporal.
- `EKF\ekf.py` e `GRU\gru.py`: versões independentes para testes isolados.
- `LSTM_P3\lstm_app.py`: demonstração independente da LSTM com interface
  gráfica; não é necessária para executar o fluxo completo.

Os resultados que já existiam foram preservados em
`arquivo\resultados_anteriores`. As novas execuções vão para `resultados\`.

## Teste isolado opcional da LSTM

O arquivo `LSTM_P3\trajetoria_teste.csv` é um pequeno conjunto sintético para
o app independente:

```powershell
python LSTM_P3\lstm_app.py
```

Carregue o CSV, use a oclusão dos quadros 80 a 109 e clique em **Treinar LSTM
e prever**. Para os scripts independentes EKF e GRU, instale também:

```powershell
python -m pip install -r requirements-extras.txt
```

Eles podem ser executados com:

```powershell
python EKF\ekf.py LSTM_P3\trajetoria_teste.csv resultados\ekf_teste.csv
python GRU\gru.py LSTM_P3\trajetoria_teste.csv resultados\gru_teste.csv
```
