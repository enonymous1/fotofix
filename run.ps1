param(
    [ValidateSet("fix", "sort", "organize", "optimize", "stitch", "cleanup")]
    [string]$command = "fix",
    [string]$target_dir = "C:\path\to\your\photos"
)

# run.ps1 - fotofix helper. Install once first:
#   python -m venv .venv
#   .\.venv\Scripts\Activate.ps1
#   pip install -e ".[vision]"

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Error: Virtual environment Python not found at '$venvPython'." -ForegroundColor Red
    Write-Host "Create it with: python -m venv .venv  then  pip install -e `".[vision]`"" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $target_dir)) {
    Write-Host "Error: Target directory '$target_dir' not found." -ForegroundColor Red
    exit 1
}

$choice = Read-Host "Run in (D)ry Run or (L)ive mode? (Default: Dry Run)"
$cliArgs = @($command, "--directory", $target_dir)

if ($choice -match "^l") {
    Write-Host ">>> LIVE MODE: Files will be modified! <<<" -ForegroundColor Yellow
    $cliArgs += "--live"
} else {
    Write-Host ">>> DRY RUN: No changes will be saved. <<<" -ForegroundColor Cyan
}

& $venvPython -m fotofix @cliArgs
