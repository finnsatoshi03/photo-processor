# Build the one-file Windows exe.
#   .\build.ps1          -> CPU build (onnxruntime)
#   .\build.ps1 -Gpu     -> GPU build (onnxruntime-gpu, needs NVIDIA CUDA)
param([switch]$Gpu)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    py -3.11 -m venv .venv
}
$pip = ".\.venv\Scripts\pip.exe"
$req = if ($Gpu) { "requirements-gpu.txt" } else { "requirements.txt" }
& $pip install -r $req
& $pip install pyinstaller

& .\.venv\Scripts\pyinstaller.exe photo-processor.spec --noconfirm --clean

Write-Host ""
Write-Host "Built: $PSScriptRoot\dist\photo-processor.exe"
Write-Host "First run downloads the model (~176 MB for u2net) to %LOCALAPPDATA%\PhotoProcessor\models"
Write-Host "Optional: photo-processor.exe --install-startup  (auto-start with Windows)"
