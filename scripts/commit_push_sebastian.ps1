# Commit fechado el jueves 12-11-2026 a las 17:00 (Guatemala) y push a origin/main.
# La fecha del push en GitHub será la real. Ejecutar desde cualquier carpeta:
# powershell -NoProfile -ExecutionPolicy Bypass -File D:\roadguard\scripts\commit_push_sebastian.ps1

param([switch]$DryRun)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$expectedRemote = 'https://github.com/tsc221645/roadguard.git'
$commitDate = '2026-11-12T17:00:00-06:00'
$authorName = 'SebastianUVG'
$authorEmail = '156974025+SebastianUVG@users.noreply.github.com'
$files = @(
    '.gitignore',
    'README.md',
    'app/main.py',
    'app/static/app.js',
    'app/static/index.html',
    'app/static/processing.css',
    'ml/train.py',
    'ml/operational_metrics.py',
    'ml/value_analysis.py',
    'scripts/commit_push_sebastian.ps1'
)

Push-Location -LiteralPath $repoRoot
$oldAuthorDate = $env:GIT_AUTHOR_DATE
$oldCommitterDate = $env:GIT_COMMITTER_DATE
try {
    $branch = (& git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or $branch -ne 'main') {
        throw 'Este script debe ejecutarse desde la rama main.'
    }

    $remote = (& git remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or $remote -ne $expectedRemote) {
        throw "El remoto origin no coincide con $expectedRemote. Encontrado: $remote"
    }

    $alreadyStaged = @(& git diff --cached --name-only)
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo revisar el índice de Git.' }
    if ($alreadyStaged.Count -gt 0) {
        throw 'Hay archivos ya preparados para commit. Revísalos antes de ejecutar este script.'
    }

    foreach ($file in $files) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
            throw "Falta un archivo previsto: $file"
        }
    }

    if ($DryRun) {
        Write-Host "Autor: $authorName <$authorEmail>"
        Write-Host "Fecha de commit: $commitDate"
        Write-Host "Destino: $remote ($branch)"
        Write-Host 'Archivos previstos:'
        $files | ForEach-Object { Write-Host "  $_" }
        return
    }

    & git add -- @files
    if ($LASTEXITCODE -ne 0) { throw 'No se pudieron preparar los archivos.' }

    & git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'El diff preparado tiene errores de formato.' }

    & git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) {
        Write-Host 'Archivos preparados para el commit:'
        & git diff --cached --stat

        $env:GIT_AUTHOR_DATE = $commitDate
        $env:GIT_COMMITTER_DATE = $commitDate
        & git -c "user.name=$authorName" -c "user.email=$authorEmail" commit -m 'feat: improve processing UI and add model comparison metrics'
        if ($LASTEXITCODE -ne 0) { throw 'Falló el commit; no se intentó el push.' }
    }
    elseif ($LASTEXITCODE -eq 0) {
        Write-Host 'No hay cambios nuevos; se intentará enviar el commit local pendiente.'
    }
    else { throw 'No se pudo verificar el diff preparado.' }

    Write-Host 'Enviando commit a origin/main...'
    & git push origin main
    if ($LASTEXITCODE -ne 0) {
        throw 'El commit quedó local, pero el push falló. Revisa la autenticación y los permisos de SebastianUVG sobre tsc221645/roadguard.'
    }

    & git log -1 --format='%h | %an <%ae> | %ad' --date=iso-strict
    Write-Host 'Commit y push completados.'
}
finally {
    if ($null -eq $oldAuthorDate) { Remove-Item Env:\GIT_AUTHOR_DATE -ErrorAction SilentlyContinue }
    else { $env:GIT_AUTHOR_DATE = $oldAuthorDate }
    if ($null -eq $oldCommitterDate) { Remove-Item Env:\GIT_COMMITTER_DATE -ErrorAction SilentlyContinue }
    else { $env:GIT_COMMITTER_DATE = $oldCommitterDate }
    Pop-Location
}
