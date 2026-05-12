param(
    [string]$Root = ".\datasets\continuous-collision-detection",
    [switch]$IncludeRounded,
    [switch]$IncludeFullSceneTOI,
    [switch]$UseCurlOnly,
    [switch]$SkipArchiveTest,
    [switch]$TrustExistingFiles
)

$ErrorActionPreference = "Stop"

function New-Dir([string]$Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Write-Log([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$stamp] $Message"
}

function Get-HttpSource([string]$Path) {
    return "http://archive.nyu.edu$Path"
}

function Get-RemoteContentLength {
    param(
        [string]$Url,
        [string]$Name
    )

    try {
        $response = Invoke-WebRequest `
            -Uri $Url `
            -Method Head `
            -MaximumRedirection 10 `
            -UseBasicParsing `
            -TimeoutSec 60
        $length = $response.Headers["Content-Length"]
        if ($length -is [array]) {
            $length = $length[-1]
        }
        if ($null -ne $length -and "$length" -ne "") {
            return [Int64]$length
        }
    } catch {
        Write-Log "HEAD failed for ${Name}: $($_.Exception.Message)"
    }

    return $null
}

function Test-ArchiveIntegrity {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not $Path.EndsWith(".tar.gz")) {
        return $true
    }

    if ($SkipArchiveTest) {
        Write-Log "archive test skipped: $Name"
        return $true
    }

    if (-not (Get-Command tar -ErrorAction SilentlyContinue)) {
        Write-Log "tar not found; archive test skipped: $Name"
        return $true
    }

    Write-Log "archive test start: $Name"
    & tar -tzf $Path *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Log "archive test failed: $Name"
        return $false
    }

    Write-Log "archive test passed: $Name"
    return $true
}

function Move-BadExistingFile {
    param(
        [string]$Destination,
        [string]$Name,
        [string]$Reason
    )

    $badDir = Join-Path (Split-Path -Parent $Destination) "incomplete"
    New-Dir $badDir
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $badPath = Join-Path $badDir "$Name.bad-$stamp"
    if (Test-Path -LiteralPath $badPath) {
        $badPath = "$badPath-$([Guid]::NewGuid().ToString('N'))"
    }

    Write-Log "move invalid existing file: $Name reason=$Reason -> $badPath"
    Move-Item -LiteralPath $Destination -Destination $badPath -Force
}

function Test-ExistingDownload {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$Name
    )

    if (-not (Test-Path -LiteralPath $Destination)) {
        return $false
    }

    if ($TrustExistingFiles) {
        Write-Log "trust existing without validation: $Name"
        return $true
    }

    Write-Log "verify existing: $Name"
    $localBytes = (Get-Item -LiteralPath $Destination).Length
    $remoteBytes = Get-RemoteContentLength -Url $Url -Name $Name

    if ($null -ne $remoteBytes) {
        if ($localBytes -ne $remoteBytes) {
            Move-BadExistingFile `
                -Destination $Destination `
                -Name $Name `
                -Reason "size mismatch local=$localBytes remote=$remoteBytes"
            return $false
        }
        Write-Log "size verified: $Name $localBytes bytes"
    } else {
        Write-Log "remote size unavailable; continue with local integrity test: $Name"
    }

    if (-not (Test-ArchiveIntegrity -Path $Destination -Name $Name)) {
        Move-BadExistingFile `
            -Destination $Destination `
            -Name $Name `
            -Reason "archive integrity failed"
        return $false
    }

    Write-Log "skip verified existing: $Name"
    return $true
}

function Download-WithBits {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$Name
    )

    $tmp = "$Destination.bits"
    if (Test-Path -LiteralPath $Destination) {
        Write-Log "skip existing: $Name"
        return
    }

    $existing = Get-BitsTransfer -AllUsers -ErrorAction SilentlyContinue |
        Where-Object { $_.DisplayName -eq "ccd-dataset-$Name" } |
        Select-Object -First 1

    if ($null -eq $existing) {
        Write-Log "BITS start: $Name"
        Start-BitsTransfer `
            -Source $Url `
            -Destination $tmp `
            -DisplayName "ccd-dataset-$Name" `
            -Description "Continuous Collision Detection dataset download" `
            -RetryInterval 60 `
            -RetryTimeout 86400 `
            -Asynchronous | Out-Null
        $job = Get-BitsTransfer -AllUsers | Where-Object { $_.DisplayName -eq "ccd-dataset-$Name" } | Select-Object -First 1
    } else {
        Write-Log "BITS resume existing job: $Name"
        $job = $existing
    }

    while ($true) {
        $job = Get-BitsTransfer -AllUsers | Where-Object { $_.DisplayName -eq "ccd-dataset-$Name" } | Select-Object -First 1
        if ($null -eq $job) {
            throw "BITS job disappeared: $Name"
        }

        if ($job.JobState -eq "Transferred") {
            Complete-BitsTransfer -BitsJob $job
            if (Test-Path -LiteralPath $tmp) {
                Move-Item -LiteralPath $tmp -Destination $Destination -Force
            }
            Write-Log "BITS done: $Name"
            return
        }

        if ($job.JobState -in @("TransientError", "Error")) {
            Write-Log "BITS retry: $Name state=$($job.JobState) error=$($job.ErrorDescription)"
            Resume-BitsTransfer -BitsJob $job -Asynchronous | Out-Null
        } else {
            $pct = 0
            if ($job.BytesTotal -gt 0) {
                $pct = [math]::Round(100.0 * $job.BytesTransferred / $job.BytesTotal, 2)
            }
            Write-Log "BITS progress: $Name $pct% $($job.BytesTransferred)/$($job.BytesTotal)"
        }

        Start-Sleep -Seconds 60
    }
}

function Download-WithCurlLoop {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$Name
    )

    if (Test-Path -LiteralPath $Destination) {
        Write-Log "skip existing: $Name"
        return
    }

    $part = "$Destination.part"
    while ($true) {
        Write-Log "curl attempt: $Name"
        & curl.exe -L --fail --retry 20 --retry-delay 10 --retry-all-errors -C - --connect-timeout 60 -o $part $Url
        $code = $LASTEXITCODE

        if ($code -eq 0) {
            Move-Item -LiteralPath $part -Destination $Destination -Force
            Write-Log "curl done: $Name"
            return
        }

        if ($code -eq 33) {
            Write-Log "server rejected byte-range resume for $Name; restarting this file"
            if (Test-Path -LiteralPath $part) {
                Remove-Item -LiteralPath $part -Force
            }
        } else {
            Write-Log "curl failed: $Name exit=$code; retry in 60s"
        }

        Start-Sleep -Seconds 60
    }
}

function Download-File {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$Name
    )

    if (Test-ExistingDownload -Url $Url -Destination $Destination -Name $Name) {
        return
    }

    if ((Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue) -and -not $UseCurlOnly) {
        Download-WithBits -Url $Url -Destination $Destination -Name $Name
    } else {
        Download-WithCurlLoop -Url $Url -Destination $Destination -Name $Name
    }

    if (-not (Test-ExistingDownload -Url $Url -Destination $Destination -Name $Name)) {
        throw "download validation failed: $Name"
    }
}

$rootPath = Resolve-Path -Path (New-Item -ItemType Directory -Force -Path $Root)
$archivePath = Join-Path $rootPath "nyu-full-dataset-archives"
New-Dir $archivePath

Write-Log "root: $rootPath"

if (Get-Command git -ErrorAction SilentlyContinue) {
    $samplePath = Join-Path $rootPath "Sample-Queries"
    if (Test-Path -LiteralPath (Join-Path $samplePath ".git")) {
        Write-Log "update Sample-Queries"
        git -C $samplePath pull --ff-only
    } else {
        Write-Log "clone Sample-Queries"
        git clone https://github.com/Continuous-Collision-Detection/Sample-Queries $samplePath
    }
} else {
    Write-Log "git not found; skip Sample-Queries clone"
}

$items = @(
    @{ Name = "ccd-queries-handcrafted.tar.gz"; Path = "/bitstream/2451/61519/2/ccd-queries-handcrafted.tar.gz"; Group = "full-query" },
    @{ Name = "ccd-queries-simulation-chain.tar.gz"; Path = "/bitstream/2451/61520/4/ccd-queries-simulation-chain.tar.gz"; Group = "full-query" },
    @{ Name = "ccd-queries-simulation-cow-heads.tar.gz"; Path = "/bitstream/2451/61520/3/ccd-queries-simulation-cow-heads.tar.gz"; Group = "full-query" },
    @{ Name = "ccd-queries-simulation-golf-ball.tar.gz"; Path = "/bitstream/2451/61520/5/ccd-queries-simulation-golf-ball.tar.gz"; Group = "full-query" },
    @{ Name = "ccd-queries-simulation-mat-twist.tar.gz"; Path = "/bitstream/2451/61520/6/ccd-queries-simulation-mat-twist.tar.gz"; Group = "full-query" }
)

if ($IncludeRounded) {
    $items += @(
        @{ Name = "rounded-ccd-queries-handcrafted.tar.gz"; Path = "/bitstream/2451/63808/2/rounded-ccd-queries-handcrafted.tar.gz"; Group = "rounded" },
        @{ Name = "rounded-ccd-queries-simulation-chain-edge-edge.tar.gz"; Path = "/bitstream/2451/63808/6/rounded-ccd-queries-simulation-chain-edge-edge.tar.gz"; Group = "rounded" },
        @{ Name = "rounded-ccd-queries-simulation-chain-vertex-face.tar.gz"; Path = "/bitstream/2451/63808/7/rounded-ccd-queries-simulation-chain-vertex-face.tar.gz"; Group = "rounded" },
        @{ Name = "rounded-ccd-queries-simulation-cow-heads.tar.gz"; Path = "/bitstream/2451/63808/3/rounded-ccd-queries-simulation-cow-heads.tar.gz"; Group = "rounded" },
        @{ Name = "rounded-ccd-queries-simulation-golf-ball.tar.gz"; Path = "/bitstream/2451/63808/4/rounded-ccd-queries-simulation-golf-ball.tar.gz"; Group = "rounded" },
        @{ Name = "rounded-ccd-queries-simulation-mat-twist.tar.gz"; Path = "/bitstream/2451/63808/5/rounded-ccd-queries-simulation-mat-twist.tar.gz"; Group = "rounded" }
    )
}

if ($IncludeFullSceneTOI) {
    $items += @(
        @{ Name = "full-scene-README.md"; Path = "/bitstream/2451/74508/2/README.md"; Group = "full-scene-toi" },
        @{ Name = "armadillo-rollers.tar.gz"; Path = "/bitstream/2451/74508/3/armadillo-rollers.tar.gz"; Group = "full-scene-toi" },
        @{ Name = "cloth-ball.tar.gz"; Path = "/bitstream/2451/74508/4/cloth-ball.tar.gz"; Group = "full-scene-toi" },
        @{ Name = "cloth-funnel.tar.gz"; Path = "/bitstream/2451/74508/5/cloth-funnel.tar.gz"; Group = "full-scene-toi" },
        @{ Name = "n-body-simulation.tar.gz"; Path = "/bitstream/2451/74508/6/n-body-simulation.tar.gz"; Group = "full-scene-toi" },
        @{ Name = "puffer-ball-boxes+queries+mma_bool+roots.tar.gz"; Path = "/bitstream/2451/74508/8/puffer-ball-boxes%2bqueries%2bmma_bool%2broots.tar.gz"; Group = "full-scene-toi" },
        @{ Name = "puffer-ball-frames.tar.gz"; Path = "/bitstream/2451/74508/9/puffer-ball-frames.tar.gz"; Group = "full-scene-toi" },
        @{ Name = "rod-twist-boxes+queries+mma_bool+roots.tar.gz"; Path = "/bitstream/2451/74508/7/rod-twist-boxes%2bqueries%2bmma_bool%2broots.tar.gz"; Group = "full-scene-toi" },
        @{ Name = "rod-twist-frames-0-999.tar.gz"; Path = "/bitstream/2451/74508/10/rod-twist-frames-0-999.tar.gz"; Group = "full-scene-toi" },
        @{ Name = "rod-twist-frames-1000-1999.tar.gz"; Path = "/bitstream/2451/74508/11/rod-twist-frames-1000-1999.tar.gz"; Group = "full-scene-toi" },
        @{ Name = "rod-twist-frames-2000-2999.tar.gz"; Path = "/bitstream/2451/74508/12/rod-twist-frames-2000-2999.tar.gz"; Group = "full-scene-toi" },
        @{ Name = "rod-twist-frames-3000-4000.tar.gz"; Path = "/bitstream/2451/74508/13/rod-twist-frames-3000-4000.tar.gz"; Group = "full-scene-toi" }
    )
}

$manifestPath = Join-Path $rootPath "download-manifest.csv"
$items | ForEach-Object {
    [PSCustomObject]@{
        name = $_.Name
        group = $_.Group
        url = Get-HttpSource $_.Path
    }
} | Export-Csv -NoTypeInformation -Encoding ASCII $manifestPath
Write-Log "manifest: $manifestPath"

foreach ($item in $items) {
    $url = Get-HttpSource $item.Path
    $dst = Join-Path $archivePath $item.Name
    Download-File -Url $url -Destination $dst -Name $item.Name
}

Write-Log "all requested downloads completed"
