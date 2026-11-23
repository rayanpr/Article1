# Balayages complementaires de la phase 3 (SPO+ exact via cvxpylayers).
#
# La grille de reference (lambda \in {0.05, 0.1, 0.3, 0.5, 1, 1.5, 2, 3, 5})
# est couverte par le run de reference `scripts\train_spo_exact.py` (env .venv311).
# Ce script produit les ablations :
#   - balayage du poids de recours beta (a lambda = 0.3)
#   - runs mono-lambda conserves comme artefacts historiques
#
# Usage : powershell -ExecutionPolicy Bypass -File scripts\run_spo_sweep_corrected.ps1
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
$py = ".venv311\Scripts\python.exe"

$runs = @(
    @{ tag = "_b05";      args = @("--lambdas", "0.3", "--beta", "0.5") },
    @{ tag = "_b20";      args = @("--lambdas", "0.3", "--beta", "2.0") },
    @{ tag = "_lam05b10"; args = @("--lambdas", "0.5") },
    @{ tag = "_lam10b10"; args = @("--lambdas", "1.0") },
    @{ tag = "_lam15b10"; args = @("--lambdas", "1.5", "--epochs", "8") },
    @{ tag = "_lam20b10"; args = @("--lambdas", "2.0", "--epochs", "8") },
    @{ tag = "_lam3e10";  args = @("--lambdas", "3.0", "--epochs", "10") },
    @{ tag = "_lam5";     args = @("--lambdas", "5.0") }
)

foreach ($r in $runs) {
    Write-Host "=== sweep $($r.tag) : $($r.args -join ' ') ==="
    & $py scripts\train_spo_exact.py @($r.args) --tag $r.tag 2>&1 |
        Select-String -NotMatch "UserWarning|torch_csr|detach|Consider using|C:.actions-runner"
    Write-Host "=== fin $($r.tag) ==="
}
Write-Host "SWEEP TERMINE"
