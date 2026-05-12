param(
  [string]$Manifest = "src\datasets\manifests\tight_inclusion_nyu_full_manifest_run_id.json",
  [string]$DatasetRoot = "",
  [string]$OutputDir = "src\benchmark\ti_full_query_shards_run_id",
  [string]$Split = "heldout_test",
  [string]$Cases = "",
  [string]$Kinds = "",
  [int]$MaxQueriesPerShard = 0,
  [string]$AsciiDrive = "",
  [switch]$Smoke,
  [switch]$Force
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

$RunRoot = $RepoRoot
if ($AsciiDrive -ne "") {
  $drive = $AsciiDrive.TrimEnd(":") + ":"
  $driveRoot = $drive + "\"
  if (!(Test-Path $driveRoot)) {
    try {
      subst $drive "$RepoRoot" | Out-Null
    } catch {
      # Fall back to the original path; the caller will see the native error if
      # this platform cannot create a drive mapping.
    }
  }
  if (Test-Path (Join-Path $driveRoot "src")) {
    $RunRoot = $driveRoot
  }
}

$Exe = Join-Path $RunRoot "src\build_tools\tight_inclusion_full_query_benchmark.exe"
if (!(Test-Path $Exe)) {
  throw "Missing executable: $Exe. Build or copy tight_inclusion_full_query_benchmark.exe first."
}

$ManifestPath = Resolve-Path (Join-Path $RunRoot $Manifest)
$OutputPath = Join-Path $RunRoot $OutputDir
New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null

if ($Smoke) {
  $Split = "unit_smoke"
  $Cases = "unit-tests"
  $MaxQueriesPerShard = 0
}

$manifestJson = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
$files = @($manifestJson.files | Where-Object { $_.split -eq $Split })
if ($Cases -ne "") {
  $caseSet = @{}
  $Cases.Split(",") | ForEach-Object { $caseSet[$_.Trim()] = $true }
  $files = @($files | Where-Object { $caseSet.ContainsKey($_.case) })
}
if ($Kinds -ne "") {
  $kindSet = @{}
  $Kinds.Split(",") | ForEach-Object { $kindSet[$_.Trim()] = $true }
  $files = @($files | Where-Object { $kindSet.ContainsKey($_.kind) })
}

$groups = $files | Group-Object -Property case,kind
if ($groups.Count -eq 0) {
  throw "No manifest groups matched Split=$Split Cases=$Cases Kinds=$Kinds"
}

$summaryRows = @()
foreach ($group in $groups) {
  $caseName = $group.Group[0].case
  $kindName = $group.Group[0].kind
  $safeCase = $caseName -replace '[^A-Za-z0-9_.-]', '_'
  $safeKind = $kindName -replace '[^A-Za-z0-9_.-]', '_'
  $baseName = "ti_full_${Split}_${safeCase}_${safeKind}"
  $jsonl = Join-Path $OutputPath "${baseName}.jsonl"
  $md = Join-Path $OutputPath "${baseName}.md"
  $log = Join-Path $OutputPath "${baseName}.log"

  $expectedQueries = ($group.Group | Measure-Object -Property query_count -Sum).Sum
  $status = "pending"
  if ((Test-Path $jsonl) -and (Test-Path $md) -and !$Force) {
    $status = "skipped_existing"
  } else {
    $args = @(
      "--manifest", "$ManifestPath",
      "--output-jsonl", "$jsonl",
      "--output-md", "$md",
      "--split", "$Split",
      "--case", "$caseName",
      "--kind", "$kindName"
    )
    if ($DatasetRoot -ne "") {
      $args += @("--dataset-root", (Resolve-Path (Join-Path $RunRoot $DatasetRoot)))
    }
    if ($MaxQueriesPerShard -gt 0) {
      $args += @("--max-queries", "$MaxQueriesPerShard")
    }
    "& `"$Exe`" $($args -join ' ')" | Set-Content -Encoding UTF8 -LiteralPath $log
    & $Exe @args 2>&1 | Tee-Object -Append -FilePath $log
    if ($LASTEXITCODE -ne 0) {
      throw "Shard failed: $caseName/$kindName. See $log"
    }
    $status = "done"
  }

  $summaryRows += [PSCustomObject]@{
    split = $Split
    case = $caseName
    kind = $kindName
    files = $group.Count
    expected_queries = $expectedQueries
    status = $status
    jsonl = $jsonl
    md = $md
    log = $log
  }
}

$summaryCsv = Join-Path $OutputPath "shard_summary.csv"
$summaryJson = Join-Path $OutputPath "shard_summary.json"
$summaryRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $summaryCsv
$summaryRows | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -Path $summaryJson
Write-Host "Wrote $summaryCsv"
Write-Host "Wrote $summaryJson"
