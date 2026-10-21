param(
    [switch]$Rapido
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputPath = Join-Path $repoRoot "resultados\execucao_$timestamp"
$modelRoot = Get-ChildItem -LiteralPath $repoRoot -Directory |
    Where-Object { $_.Name -like "modelagem*" } |
    Select-Object -First 1
if (-not $modelRoot) {
    throw "The modelagem source folder was not found."
}
$pipelineFile = Get-ChildItem -LiteralPath $modelRoot.FullName `
    -Filter "rastreamento_oclusao.py" -File -Recurse |
    Select-Object -First 1
if (-not $pipelineFile) {
    throw "rastreamento_oclusao.py was not found under the modelagem folder."
}
$pipelinePath = $pipelineFile.FullName
$transformerPath = Join-Path $repoRoot "TRANSFORMER\transformer.py"

if (Test-Path -LiteralPath $outputPath) {
    throw "A pasta de saída já existe: $outputPath. Execute novamente em outro segundo."
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python não foi encontrado no PATH. Instale Python e reabra o PowerShell."
}

& python -c "import cv2, matplotlib, numpy, tensorflow; from PIL import Image"
if ($LASTEXITCODE -ne 0) {
    throw "Missing dependencies. Install them with: python -m pip install -r requirements.txt"
}

$pipelineArguments = @($pipelinePath, "--saida", $outputPath)
if ($Rapido) {
    $pipelineArguments += "--rapido"
}

Write-Host "Gerando os experimentos em: $outputPath"
& python @pipelineArguments
if ($LASTEXITCODE -ne 0) {
    throw "The main pipeline failed with exit code $LASTEXITCODE. Check the output above."
}

$observacoesPath = Join-Path $outputPath "01_video\observacoes.csv"
$referenciaPath = Join-Path $outputPath "01_video\trajetoria_real.csv"
$transformerOutput = Join-Path $outputPath "03_transformer\previsoes_transformer.csv"
& python $transformerPath $observacoesPath $transformerOutput --referencia $referenciaPath
if ($LASTEXITCODE -ne 0) {
    throw "The Transformer failed with exit code $LASTEXITCODE. Main pipeline results were preserved."
}

$avaliacaoPath = Join-Path $repoRoot "avaliar_comparacao.py"

Write-Host "Gerando a comparação final dos quatro métodos..."
& python $avaliacaoPath $outputPath
if ($LASTEXITCODE -ne 0) {
    throw "A comparação final falhou. Os resultados anteriores foram preservados."
}

Write-Host ""
Write-Host "Run completed. Results:"
Write-Host $outputPath
