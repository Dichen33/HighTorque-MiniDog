@echo off
setlocal
pushd "%~dp0..\mujoco\mjlab" || exit /b 1
call scripts\play_native.bat %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
