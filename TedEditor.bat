@if exist pythonenv.bat @(
  @call pythonenv.bat
)
@python ted-editor.py
@if ERRORLEVEL 1 @(
    @echo python version 3 not found on your system. create the file pythonenv.bat in this directory
    @echo and insert a line PATH=^<PathToPythonInstallation^>;^<PathToPythonInstallation^>\Scripts;^%PATH^%
    @ and try again
    @pause
)
