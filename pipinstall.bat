@if exist pythonenv.bat @(
  @call pythonenv.bat
)
@python -m pip install -r requirements.txt
@if ERRORLEVEL 1 @(
    @echo python not found on your system. create the file pythonenv.bat in this directory
    @echo and insert a line PATH=^%PATH^%;<PathToPythonInstallation>;<PathToPythonInstallation>\Scripts
    @ and try again
    @pause
)
