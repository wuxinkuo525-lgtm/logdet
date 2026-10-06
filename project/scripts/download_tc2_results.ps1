$ErrorActionPreference = 'Stop'
$clusterTarget = 'xinkuo001@10.96.189.12'
# Each export goes into a new timestamped folder so earlier exports are never overwritten.
$exportDir = 'E:\ntu\computer_vision\cvproject\logdet\runs\tc2_export_' + (Get-Date -Format 'yyyyMMdd_HHmm')

try {
    New-Item -ItemType Directory -Force -Path $exportDir | Out-Null

    Write-Host '[1/3] Packing results on TC2 (no model weights). Enter your NTU password when prompted.'
    # Logs are expanded with ls so that tar does not fail when one kind of log does not exist yet.
    $remoteCommand = 'cd ~/logdet && tar -czf runs/tc2_results_export.tar.gz --exclude=*.pth runs/dino runs/predictions/dino $(ls project/output_logodet_*.out project/error_logodet_*.err 2>/dev/null)'
    & ssh -p 22 $clusterTarget $remoteCommand
    if ($LASTEXITCODE -ne 0) { throw 'Remote packing failed. Check the error above. No download was started.' }

    Write-Host '[2/3] Downloading archive. You may be asked for your password again.'
    & scp -P 22 "${clusterTarget}:logdet/runs/tc2_results_export.tar.gz" "$exportDir\tc2_results_export.tar.gz"
    if ($LASTEXITCODE -ne 0) { throw 'Download failed. The local archive may be incomplete.' }

    Write-Host '[3/3] Extracting results on this Windows computer...'
    & tar -xzf "$exportDir\tc2_results_export.tar.gz" -C $exportDir
    if ($LASTEXITCODE -ne 0) { throw 'Local extraction failed.' }

    Write-Host "SUCCESS: Download and extraction completed. Results: $exportDir" -ForegroundColor Green
} catch {
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
