$python = 'd:/GAN/venv/Scripts/python.exe'
$runner = 'local_repro/experiment_runner.py'

$configs = @(
    @{ name='A'; feval=0.60; subsample=0.40; epochs=1; freeze=$true },
    @{ name='B'; feval=0.62; subsample=0.35; epochs=1; freeze=$true },
    @{ name='C'; feval=0.65; subsample=0.35; epochs=1; freeze=$true },
    @{ name='D'; feval=0.60; subsample=0.40; epochs=2; freeze=$true },
    @{ name='E'; feval=0.62; subsample=0.40; epochs=2; freeze=$true },
    @{ name='F'; feval=0.65; subsample=0.40; epochs=2; freeze=$true },
    @{ name='G'; feval=0.60; subsample=0.45; epochs=1; freeze=$true },
    @{ name='H'; feval=0.62; subsample=0.45; epochs=1; freeze=$true }
)

$results = @()

foreach ($cfg in $configs) {
    Write-Host "=== Running $($cfg.name) ===" -ForegroundColor Cyan

    $freezeFlag = if ($cfg.freeze) { '--freeze-backbone' } else { '' }

    $cmd = @(
        $python,
        $runner,
        '--model-name', 'densenet121',
        '--scenario', 'field',
        '--field-eval-distortion-prob', "$($cfg.feval)",
        '--real-train-subsample', "$($cfg.subsample)",
        '--epochs', "$($cfg.epochs)",
        '--batch-size', '16',
        '--num-workers', '0',
        '--seeds', '42', '1337', '2025',
        '--ratios', '0', '0.25', '0.5', '1.0', '2.0',
        $freezeFlag
    ) | Where-Object { $_ -ne '' }

    & $cmd[0] $cmd[1..($cmd.Length-1)] | Out-Host

    $summary = Import-Csv 'D:\GAN\local_repro\outputs\summary_results.csv'
    $sorted = $summary | Sort-Object {[double]$_.acc_mean} -Descending
    $best = $sorted[0]
    $acc0 = [double](($summary | Where-Object { $_.ratio_pct -eq '0' }).acc_mean)
    $acc50 = [double](($summary | Where-Object { $_.ratio_pct -eq '50' }).acc_mean)

    $results += [PSCustomObject]@{
        Config = $cfg.name
        EvalDistortion = $cfg.feval
        RealSubsample = $cfg.subsample
        Epochs = $cfg.epochs
        Baseline0 = [math]::Round($acc0, 4)
        Acc50 = [math]::Round($acc50, 4)
        BestRatio = [int]$best.ratio_pct
        BestAcc = [math]::Round([double]$best.acc_mean, 4)
        Is50Best = ([int]$best.ratio_pct -eq 50)
    }

    $results | Export-Csv 'D:\GAN\local_repro\outputs\sweep_results.csv' -NoTypeInformation
    Write-Host "=== Completed $($cfg.name) ===`n" -ForegroundColor Green
}

Write-Host "Sweep complete. Saved: D:\GAN\local_repro\outputs\sweep_results.csv" -ForegroundColor Yellow
$results | Format-Table -AutoSize
