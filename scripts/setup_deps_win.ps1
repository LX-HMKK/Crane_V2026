# Windows dependency provisioning for the dev-verification build.
# Downloads/installs into <repo>/tools/ (gitignored), idempotent per step.
#
#   ORT    -> tools/onnxruntime-win   (official win-x64 zip, 1.23.2)
#   OpenCV -> tools/opencv            (official 4.12 prebuilt, self-extract)
#   yaml-cpp -> tools/yaml-cpp        (0.7.0 built from source once, static)
#
# Run from anywhere: `powershell -ExecutionPolicy Bypass -File scripts\setup_deps_win.ps1`
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$tools = Join-Path $root "tools"

$ortVer = "1.23.2"
$opencvVer = "4.12.0"
$yamlVer = "0.7.0"

# ── ONNX Runtime ────────────────────────────────────────────────────────
$ortDir = Join-Path $tools "onnxruntime-win"
if (-not (Test-Path (Join-Path $ortDir "lib\onnxruntime.dll"))) {
  Write-Host "==> ONNX Runtime $ortVer (win-x64)"
  $zip = Join-Path $env:TEMP "onnxruntime-win-x64-$ortVer.zip"
  Invoke-WebRequest -Uri "https://github.com/microsoft/onnxruntime/releases/download/v$ortVer/onnxruntime-win-x64-$ortVer.zip" -OutFile $zip
  New-Item -ItemType Directory -Force $ortDir | Out-Null
  Expand-Archive -Path $zip -DestinationPath $ortDir
  # Hoist the inner onnxruntime-win-*/ folder so layout matches tools/onnxruntime.
  $inner = Get-ChildItem $ortDir -Directory | Where-Object Name -like "onnxruntime-win-*" | Select-Object -First 1
  if ($inner) {
    Get-ChildItem $inner.FullName | Move-Item -Destination $ortDir -Force
    Remove-Item $inner.FullName -Recurse -Force
  }
  Remove-Item $zip -Force
} else {
  Write-Host "==> ONNX Runtime present, skipping"
}

# ── OpenCV prebuilt ─────────────────────────────────────────────────────
$ocvCfg = Get-ChildItem $tools -Recurse -Filter "OpenCVConfig.cmake" -EA SilentlyContinue |
          Select-Object -First 1
if (-not $ocvCfg) {
  Write-Host "==> OpenCV $opencvVer (windows prebuilt)"
  $exe = Join-Path $env:TEMP "opencv-$opencvVer-windows.exe"
  Invoke-WebRequest -Uri "https://github.com/opencv/opencv/releases/download/$opencvVer/opencv-$opencvVer-windows.exe" -OutFile $exe
  # 7-Zip SFX: silent extract with -o<dir> -y -> lands in tools/opencv/build.
  # NOTE: the SFX is a GUI-subsystem exe, so &$exe returns immediately without
  # waiting — Start-Process -Wait is required or the completeness check races
  # the background extraction.
  $p = Start-Process -FilePath $exe -ArgumentList "-o$tools", "-y" -Wait -PassThru
  if ($p.ExitCode -ne 0) {
    throw "OpenCV self-extract exited with code $($p.ExitCode)"
  }
  $ocvCfg = Get-ChildItem $tools -Recurse -Filter "OpenCVConfig.cmake" -EA SilentlyContinue | Select-Object -First 1
  if (-not $ocvCfg) {
    if (Get-Command 7z -EA SilentlyContinue) {
      & 7z x $exe "-o$tools" -y
      $ocvCfg = Get-ChildItem $tools -Recurse -Filter "OpenCVConfig.cmake" -EA SilentlyContinue | Select-Object -First 1
    }
    if (-not $ocvCfg) {
      throw "OpenCV self-extract failed. Extract opencv-$opencvVer-windows.exe manually into tools\ (it needs to contain opencv\build\OpenCVConfig.cmake), or install 7-Zip and re-run."
    }
  }
  Remove-Item $exe -Force
} else {
  Write-Host "==> OpenCV present, skipping"
}

# ── yaml-cpp from source ────────────────────────────────────────────────
$yamlDir = Join-Path $tools "yaml-cpp"
if (-not (Test-Path (Join-Path $yamlDir "include\yaml-cpp\yaml.h"))) {
  Write-Host "==> yaml-cpp $yamlVer (source build)"
  # The release tag is "yaml-cpp-0.7.0" (not "0.7.0"), and the archive
  # unpacks to a folder named yaml-cpp-yaml-cpp-0.7.0/.
  $tag = "yaml-cpp-$yamlVer"
  $tgz = Join-Path $env:TEMP "yaml-cpp-$yamlVer.tar.gz"
  $src = Join-Path $env:TEMP "yaml-cpp-yaml-cpp-$yamlVer"
  $bld = Join-Path $env:TEMP "yaml-cpp-$yamlVer-build"
  Invoke-WebRequest -Uri "https://github.com/jbeder/yaml-cpp/archive/refs/tags/$tag.tar.gz" -OutFile $tgz
  tar -xzf $tgz -C $env:TEMP
  # CMake 4.x dropped compatibility with <3.5; yaml-cpp 0.7.0 requires 3.4.
  cmake -S $src -B $bld "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" `
        -DCMAKE_BUILD_TYPE=Release -DYAML_BUILD_SHARED_LIBS=OFF `
        "-DCMAKE_INSTALL_PREFIX=$yamlDir"
  cmake --build $bld --config Release
  cmake --install $bld --config Release
  Remove-Item $tgz -Force
  Remove-Item $src -Recurse -Force
  Remove-Item $bld -Recurse -Force
} else {
  Write-Host "==> yaml-cpp present, skipping"
}

Write-Host "`nWindows deps ready:"
Write-Host "  ORT:    $ortDir"
Write-Host "  OpenCV: $(Split-Path (Split-Path $ocvCfg.FullName -Parent) -Parent)"
Write-Host "  yaml:   $yamlDir"
