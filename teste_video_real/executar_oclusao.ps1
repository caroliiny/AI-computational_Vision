# Fluxo completo de previsão de pose sob oclusão, para todos os vídeos em videos_originais\ (ou os informados).
#
#   .\executar_oclusao.ps1                          # todos os .mp4 de videos_originais\
#   .\executar_oclusao.ps1 -Videos corrida.mp4      # um vídeo específico
#   .\executar_oclusao.ps1 -Rapido                  # YOLO leve + poucas épocas (só para testar o fluxo)
#   .\executar_oclusao.ps1 -Pular 1                 # reaproveita a extração de pose já feita (começa na etapa 2)
#
# Saída: resultados_oclusao\<nome_do_video>\   (ver README_DO_PROJETO.md)
param(
    [string[]]$Videos,
    [string]$Modelo = "yolov8x-pose.pt",
    [int]$Epocas = 100,
    [int]$Semente = 0,
    [int]$Pular = 0,          # nº de etapas iniciais a pular (1 = pula a extração do vídeo original)
    [switch]$Rapido
)

$ErrorActionPreference = "Continue"   # o Python escreve logs no stderr; as falhas são checadas por $LASTEXITCODE
Set-Location $PSScriptRoot
if ($Rapido) { $Modelo = "yolov8n-pose.pt"; $Epocas = 10 }

$py = if (Test-Path ".\pose_env\Scripts\python.exe") { ".\pose_env\Scripts\python.exe" } else { "python" }

if (-not $Videos) { $Videos = Get-ChildItem .\videos_originais -Filter *.mp4 | ForEach-Object { $_.FullName } }
if (-not $Videos) { throw "Nenhum vídeo encontrado. Coloque os vídeos originais (.mp4) em teste_video_real\videos_originais\" }

function Passo($n, $titulo, [scriptblock]$comando) {
    if ($n -le $Pular) { Write-Host "[$n] $titulo (pulado)" -ForegroundColor DarkGray; return }
    Write-Host "`n[$n] $titulo" -ForegroundColor Cyan
    & $comando
    if ($LASTEXITCODE -ne 0) { throw "Falha na etapa $n ($titulo)." }
}

foreach ($v in $Videos) {
    $video = (Resolve-Path $v).Path
    $nome = [IO.Path]::GetFileNameWithoutExtension($video)
    $pasta = "resultados_oclusao\$nome"
    New-Item -ItemType Directory -Force $pasta | Out-Null
    Write-Host "`n================ $nome ================" -ForegroundColor Green

    Passo 1 "Poses de referência do vídeo original (COCO-17)" {
        & $py processar_pose.py $video --saida "$pasta\referencia_pose.npz" --modelo $Modelo }
    Passo 2 "Gerar vídeo com oclusões artificiais" {
        & $py gerar_oclusoes.py $video $pasta --semente $Semente }
    Passo 3 "Poses detectadas no vídeo com oclusão (entrada dos modelos)" {
        & $py processar_pose.py "$pasta\video_oclusao.mp4" --saida "$pasta\oclusao_pose.npz" --modelo $Modelo }
    Passo 4 "Treinar LSTM, GRU, Transformer e T-GNN" {
        & $py treinar_previsao.py $pasta --epocas $Epocas --semente $Semente }
    Passo 5 "Avaliar os cinco métodos (5, 10 e 20 quadros)" {
        & $py avaliar_previsao.py $pasta }
    Passo 6 "Gráficos e vídeo de previsões" {
        & $py visualizar_previsao.py $pasta }

    Write-Host "`nResultados de '$nome' em: $((Resolve-Path $pasta).Path)" -ForegroundColor Green
}
