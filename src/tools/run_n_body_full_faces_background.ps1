$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

conda activate cudadev
python (Join-Path $RepoRoot "src\tools\render_n_body_full_faces_mp4.py") --frame-count 240 --fps 60 --force
