@echo off
rem usage: cpp\scripts\build-windows.bat <OpenCV_DIR>
rem   e.g. F:\Project\Cpp\MaaFramework\source\MaaUtils\MaaDeps\vcpkg\installed\maa-x64-windows\share\opencv4
setlocal
set "HERE=%~dp0.."
if "%~1"=="" (echo usage: build-windows.bat ^<OpenCV_DIR^> & exit /b 1)
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set "VS=%%i"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cmake -S "%HERE%" -B "%HERE%\build\windows" -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="%~1" || exit /b 1
cmake --build "%HERE%\build\windows" || exit /b 1
for %%d in ("%~1\..\..\bin\opencv_world*.dll") do copy /y "%%d" "%HERE%\build\windows\" >nul
endlocal
