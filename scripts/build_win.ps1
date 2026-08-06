# Windows dev-verification build: mock cameras + no serial + ONNX + Web UI.
# Prereq: run scripts/setup_deps_win.ps1 once (downloads ORT/OpenCV, builds
# yaml-cpp). Uses the Visual Studio 2026 generator (MSVC BuildTools) — no
# manual vcvars64 sourcing needed. Falls back to Ninja + vcvars64 if present.
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts\build_win.ps1
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$build = Join-Path $root "build-win"

# ── Locate installed deps ────────────────────────────────────────────────
$ortRoot = Join-Path $root "tools\onnxruntime-win"
if (-not (Test-Path (Join-Path $ortRoot "lib\onnxruntime.dll"))) {
  throw "ONNX Runtime missing at $ortRoot — run scripts\setup_deps_win.ps1"
}

# OpenCV build root = dir containing include/opencv2. The TOP-LEVEL
# OpenCVConfig.cmake only recognizes MSVC up to VS2022 (19.4x); our MSVC
# 14.50 (VS2026) falls through and sets OpenCV_FOUND=FALSE. So we point
# OpenCV_DIR at the concrete x64\vcXX\lib config (vc16 import libs are
# ABI-compatible with newer MSVC) and use that for the DLL dir too.
$ocvRoot = Get-ChildItem (Join-Path $root "tools") -Recurse -Directory -EA SilentlyContinue |
           Where-Object { Test-Path (Join-Path $_.FullName "include\opencv2\core.hpp") } |
           Select-Object -First 1
if (-not $ocvRoot) { throw "OpenCV not found — run scripts\setup_deps_win.ps1" }
$opencvDir = $ocvRoot.FullName
$vcDir = Get-ChildItem (Join-Path $opencvDir "x64") -Directory |
         Where-Object Name -match '^vc' | Sort-Object Name | Select-Object -Last 1
if (-not $vcDir) { throw "No x64\vcXX dir under $opencvDir" }
$opencvCfgDir = Join-Path $vcDir.FullName "lib"
if (-not (Test-Path (Join-Path $opencvCfgDir "OpenCVConfig.cmake"))) {
  throw "OpenCVConfig.cmake missing at $opencvCfgDir"
}

$yamlDir = Join-Path $root "tools\yaml-cpp"
if (-not (Test-Path (Join-Path $yamlDir "include\yaml-cpp\yaml.h"))) {
  throw "yaml-cpp missing at $yamlDir — run scripts\setup_deps_win.ps1"
}
$yamlCfgDir = Get-ChildItem $yamlDir -Recurse -Filter "yaml-cpp-config.cmake" -EA SilentlyContinue |
              Select-Object -First 1
if (-not $yamlCfgDir) {
  throw "yaml-cpp cmake config not found under $yamlDir — re-run setup_deps_win.ps1"
}

# Runtime DLLs for crane_vision AND every ctest exe.
$env:PATH = "$($vcDir.FullName)\bin;$ortRoot\lib;$(Split-Path $yamlCfgDir.FullName -Parent);$env:PATH"
Write-Host "OpenCV DLL dir: $($vcDir.FullName)\bin"
Write-Host "ORT    DLL dir: $ortRoot\lib"

# ── Configure ────────────────────────────────────────────────────────────
$genArgs = @("-S", $root, "-B", $build, "-DCMAKE_BUILD_TYPE=Release",
             "-DONNXRUNTIME_ROOT=$ortRoot",
             "-DOpenCV_DIR=$opencvCfgDir",
             "-Dyaml-cpp_DIR=$(Split-Path $yamlCfgDir.FullName -Parent)",
             "-DCMAKE_PREFIX_PATH=$yamlDir")

$vs2026 = "Visual Studio 18 2026"
$hasVS = (Get-Command cmake).Source -and (cmake --help 2>&1 | Select-String -Quiet "$vs2026")
$hasNinja = Get-Command ninja -EA SilentlyContinue

if ($hasVS) {
  cmake @genArgs -G $vs2026 -A x64
} elseif ($hasNinja) {
  Write-Host "Visual Studio generator unavailable; using Ninja + vcvars64"
  $vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  if (-not (Test-Path $vcvars)) {
    $vcvars = (Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio" -Recurse -Filter "vcvars64.bat" -EA SilentlyContinue | Select-Object -First 1).FullName
  }
  if (-not $vcvars) { throw "vcvars64.bat not found — install VS BuildTools" }
  cmd /c "`"$vcvars`" && cmake @genArgs -G Ninja"
} else {
  throw "Neither the Visual Studio 2026 generator nor Ninja is available."
}
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

# ── Build + test ─────────────────────────────────────────────────────────
cmake --build $build --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "build failed" }

ctest --test-dir $build -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "ctest failed" }

# ── Smoke: --once static images (end-to-end) ─────────────────────────────
$exe = Join-Path $build "Release\crane_vision.exe"
Write-Host "`n==> smoke: --once"
& $exe --once --bean-image (Join-Path $root "tests\data\beans_2.jpg") `
       --digit-image (Join-Path $root "tests\data\num_2.jpg")
if ($LASTEXITCODE -ne 0) { throw "--once smoke failed" }

# ── Smoke: CUDA EP (only when the GPU ORT package is installed) ─────────
# Informational, NOT fatal: the CPU dev-verification path above is the point
# of this script; this only reports whether --backend cuda is usable right
# now. Needs CUDA 12 + cuDNN 9 on PATH. GPU avg is ~5-8ms on an RTX 4060 vs
# ~27ms CPU, so an avg > 15ms means the bench ran on the CPU EP.
if (Test-Path (Join-Path $ortRoot "lib\onnxruntime_providers_cuda.dll")) {
  Write-Host "`n==> smoke: --backend cuda bench (informational)"
  $cudaOut = & $exe --backend cuda --mock-cameras --bench 10 2>&1
  $cudaOut | ForEach-Object { Write-Host "    $_" }
  $cudaOk = $LASTEXITCODE -eq 0
  if ($cudaOk) {
    foreach ($line in $cudaOut) {
      if ($line -match 'avg=([0-9.]+)ms') {
        if ([double]$matches[1] -gt 15.0) { $cudaOk = $false }
      }
    }
  }
  if ($cudaOk) {
    Write-Host "    CUDA EP OK (GPU inference active)"
  } else {
    Write-Host "    WARN: CUDA EP smoke failed (exit $LASTEXITCODE) — install CUDA 12.x + cuDNN 9.x and add both bin dirs to PATH; CPU verification above is unaffected"
  }
} else {
  Write-Host "`n==> CUDA EP smoke skipped (CPU ORT package; run setup_deps_win.ps1 -GPU for GPU)"
}
# The CUDA exe run above leaves $LASTEXITCODE=1 when CUDA is not yet installed;
# it is informational only, so reset it — the script's exit code must reflect
# the actual dev-verification build, not the optional GPU probe.
$LASTEXITCODE = 0

# ── Smoke: Web UI boot (127.0.0.1 avoids the firewall prompt) ────────────
Write-Host "`n==> smoke: Web UI"
$job = Start-Job -ScriptBlock {
  param($e, $root)
  # The config path and assets/*.onnx are relative to the repo root; a job
  # spawned from elsewhere must chdir first or the server exits at load.
  Set-Location $root
  & $e --mock-cameras --no-serial --web-listen 127.0.0.1 --web-port 8080
} -ArgumentList $exe, $root
try {
  Start-Sleep -Seconds 6
  $status = Invoke-WebRequest -Uri "http://127.0.0.1:8080/api/status" -UseBasicParsing
  $ok = $status.Content -match '"serial":"na"' -and
        $status.Content -match '"recognition":false' -and
        $status.Content -match '"bean":"ok"' -and
        $status.Content -match '"digit":"ok"'
  if (-not $ok) {
    throw "Web UI status check failed: $($status.Content)"
  }
  # No-serial default is idle (recognition off) so the process does not pin
  # 100% CPU while calibrating. DEBUG is the only way to force inference for
  # calibration/verification — verify the toggle round-trips.
  $dbg = Invoke-WebRequest -Uri "http://127.0.0.1:8080/api/debug?mode=1" `
                           -Method POST -UseBasicParsing
  $status = Invoke-WebRequest -Uri "http://127.0.0.1:8080/api/status" -UseBasicParsing
  if ($dbg.Content -notmatch '"debug":true' -or
      $status.Content -notmatch '"recognition":true') {
    throw "DEBUG on did not force recognition: $($dbg.Content) / $($status.Content)"
  }
  Invoke-WebRequest -Uri "http://127.0.0.1:8080/api/debug?mode=0" `
                    -Method POST -UseBasicParsing | Out-Null
  $status = Invoke-WebRequest -Uri "http://127.0.0.1:8080/api/status" -UseBasicParsing
  if ($status.Content -notmatch '"recognition":false') {
    throw "DEBUG off did not close the gate: $($status.Content)"
  }
  $html = Invoke-WebRequest -Uri "http://127.0.0.1:8080/" -UseBasicParsing
  if ($html.Content -notmatch "Crane Vision") { throw "GET / did not return the UI" }
  Write-Host "Web UI OK: serial=na, recognition=off default, DEBUG round-trip works"
} finally {
  Stop-Job $job -EA SilentlyContinue
  Remove-Job $job -Force -EA SilentlyContinue
}

Write-Host "`nWindows dev-verification build OK"
# Explicit exit 0: the CUDA smoke and the Web-UI Start-Job leave $LASTEXITCODE
# nonzero (informational probe + job child-process exit code), so without this
# the script "fails" despite every check passing. All real failures throw above
# and never reach here.
exit 0
