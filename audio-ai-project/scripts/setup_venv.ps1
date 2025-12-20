param(
    [string]$EnvName = "venv",
    [string]$Python = "python"
)

Write-Output "Creating virtual environment '$EnvName'..."
& $Python -m venv $EnvName

Write-Output "Upgrading pip in the venv..."
& "${EnvName}\Scripts\python.exe" -m pip install --upgrade pip

Write-Output "Installing requirements from requirements.txt..."
& "${EnvName}\Scripts\python.exe" -m pip install -r requirements.txt

Write-Output "Done. To activate the venv in PowerShell: .\${EnvName}\Scripts\Activate.ps1"
