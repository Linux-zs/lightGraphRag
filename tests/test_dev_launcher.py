import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(not shutil.which("powershell.exe"), reason="Windows launcher contract")
def test_launcher_parser_and_child_exit_detection():
    script = Path(__file__).resolve().parents[1] / "scripts" / "start-dev.ps1"
    literal = "'" + str(script).replace("'", "''") + "'"
    command = """
$ErrorActionPreference = 'Stop'
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(SCRIPT, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'Launcher parse failed' }
$fn = $ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Assert-ChildRunning'}, $true)
Invoke-Expression $fn.Extent.Text
Assert-ChildRunning -Process $null -Name Backend -Logs logs
Assert-ChildRunning -Process ([pscustomobject]@{HasExited=$false}) -Name Backend -Logs logs
try {
    Assert-ChildRunning -Process ([pscustomobject]@{HasExited=$true;ExitCode=42}) -Name Backend -Logs logs
    throw 'Expected early-exit error'
} catch {
    if ($_.Exception.Message -notmatch 'exit code 42') { throw }
}
exit 0
""".replace("SCRIPT", literal)
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
