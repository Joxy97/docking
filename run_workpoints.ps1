param(
    [string]$PythonBin = "python",
    [int]$RunsPerPoint = 30,
    [string]$OutRoot = "test_SA_nr-2000_ns-1000"
)

$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

if (-not (Test-Path -Path $OutRoot)) {
    New-Item -Path $OutRoot -ItemType Directory | Out-Null
}

function Format-Lambda {
    param([double]$Value)
    return ("{0:F1}" -f $Value)
}

function Invoke-Workpoint {
    param(
        [double]$LambdaSingle,
        [double]$LambdaPair
    )

    $lsFmt = Format-Lambda $LambdaSingle
    $lpFmt = Format-Lambda $LambdaPair
    $groupDir = Join-Path $OutRoot "lam_single-$lsFmt`_lam_pair-$lpFmt"

    if (-not (Test-Path -Path $groupDir)) {
        New-Item -Path $groupDir -ItemType Directory | Out-Null
    }

    # Clean previous files for this workpoint inside group folder
    Get-ChildItem -Path $groupDir -Filter "SA_lam-$lpFmt-$lsFmt`_score-*.sdf" -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    Remove-Item -Path (Join-Path $groupDir "scores_sorted.txt") -Force -ErrorAction SilentlyContinue

    Write-Host "============================================================"
    Write-Host "Workpoint: lambda_single=$lsFmt, lambda_pair=$lpFmt"
    Write-Host "============================================================"

    for ($runIdx = 1; $runIdx -le $RunsPerPoint; $runIdx++) {
        Write-Host ("Run {0}/{1}" -f $runIdx, $RunsPerPoint)

        $runOutput = & $PythonBin "main.py" `
            "-sampler" "SA" `
            "-lambda--single" "$LambdaSingle" `
            "-lambda--pair" "$LambdaPair" 2>&1 | Out-String

        Write-Host $runOutput

        if ($LASTEXITCODE -ne 0) {
            throw "main.py failed for lambda_single=$lsFmt, lambda_pair=$lpFmt (run $runIdx)."
        }

        $sdfPath = $null
        $match = [regex]::Match($runOutput, "(?m)^Wrote SDF to (.+)$")
        if ($match.Success) {
            $sdfPath = $match.Groups[1].Value.Trim()
        }

        if (-not $sdfPath -or -not (Test-Path -Path $sdfPath)) {
            $fallback = Get-ChildItem -Path $OutRoot -Filter "SA_lam-$lpFmt-$lsFmt`_score-*.sdf" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
            if ($fallback) {
                $sdfPath = $fallback.FullName
            }
        }

        if (-not $sdfPath -or -not (Test-Path -Path $sdfPath)) {
            throw "Could not find output SDF for run $runIdx (lambda_single=$lsFmt, lambda_pair=$lpFmt)."
        }

        $baseName = [System.IO.Path]::GetFileNameWithoutExtension($sdfPath)
        $runName = "{0}_run-{1}.sdf" -f $baseName, $runIdx.ToString("00")
        $destPath = Join-Path $groupDir $runName
        Move-Item -Path $sdfPath -Destination $destPath -Force
    }

    # Rank by score (lowest first)
    $files = Get-ChildItem -Path $groupDir -Filter "SA_lam-$lpFmt-$lsFmt`_score-*_run-*.sdf" -ErrorAction SilentlyContinue
    $ranked = @()
    foreach ($f in $files) {
        $m = [regex]::Match($f.Name, "_score-([0-9]+(?:\.[0-9]+)?)_run-\d+\.sdf$")
        if ($m.Success) {
            $score = [double]::Parse($m.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
            $ranked += [pscustomobject]@{
                Score = $score
                Path  = $f.FullName
            }
        }
    }

    if ($ranked.Count -eq 0) {
        Write-Warning "No scored SDF files found in $groupDir after runs."
        return
    }

    $ranked = $ranked | Sort-Object Score

    $scoreLines = $ranked | ForEach-Object {
        "{0}`t{1}" -f ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0}", $_.Score)), $_.Path
    }
    Set-Content -Path (Join-Path $groupDir "scores_sorted.txt") -Value $scoreLines

    $best = $ranked[0]
    $bestBase = [System.IO.Path]::GetFileName($best.Path)
    $bestClean = [regex]::Replace($bestBase, "_run-\d+\.sdf$", ".sdf")
    $bestCleanPath = Join-Path $groupDir $bestClean
    Move-Item -Path $best.Path -Destination $bestCleanPath -Force

    for ($i = 1; $i -lt $ranked.Count; $i++) {
        Remove-Item -Path $ranked[$i].Path -Force -ErrorAction SilentlyContinue
    }

    Write-Host "Kept best file: $bestCleanPath"
    Write-Host "Saved ranking:  $(Join-Path $groupDir "scores_sorted.txt")"
}

# Build unique workpoints:
# 1) lambda_single fixed to 1, lambda_pair = 1..5
# 2) lambda_pair fixed to 1, lambda_single = 1..5
$seen = @{}
$workpoints = @()

foreach ($lp in 1..5) {
    $key = "1,$lp"
    if (-not $seen.ContainsKey($key)) {
        $seen[$key] = $true
        $workpoints += $key
    }
}

foreach ($ls in 1..5) {
    $key = "$ls,1"
    if (-not $seen.ContainsKey($key)) {
        $seen[$key] = $true
        $workpoints += $key
    }
}

Write-Host "Workpoints to execute ($($workpoints.Count) unique):"
foreach ($wp in $workpoints) {
    $parts = $wp.Split(",")
    Write-Host "  lambda_single=$($parts[0]), lambda_pair=$($parts[1])"
}

foreach ($wp in $workpoints) {
    $parts = $wp.Split(",")
    Invoke-Workpoint -LambdaSingle ([double]$parts[0]) -LambdaPair ([double]$parts[1])
}

Write-Host "All workpoints completed."
