# One-time setup for the Kokoro voice engine.
#
# Kokoro's English phonetics need spaCy -> thinc -> blis, and blis has no
# Windows wheel past Python 3.12 (see README.md). This project's
# Python is newer than that, so Kokoro runs in its own Python 3.12 venv
# instead, talked to as a subprocess by KokoroEngine in engines.py.
#
# Requires Python 3.12 to be installed and registered with the `py` launcher
# (py -0p to check; `winget install Python.Python.3.12` if it's missing).
#
#     powershell -ExecutionPolicy Bypass -File setup_kokoro.ps1

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $here "kokoro_venv"

py -3.12 -m venv $venv
& "$venv\Scripts\python.exe" -m pip install --upgrade pip
# Exact versions known to work together - see requirements-kokoro.txt.
& "$venv\Scripts\python.exe" -m pip install -r (Join-Path $here "requirements-kokoro.txt")

Write-Host "`nKokoro venv ready at $venv"
Write-Host "Run:  python talkify.py --engine kokoro"
