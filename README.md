# TedEditor
Editor for TED files

##Installation
Download a
[ZIP](https://github.com/tarnheld/ted-editor/archive/master.zip) of
the current repository and extract all contents preserving the folder
structure.

On Windows double click the batch file
```
pipinstall.bat
```
for installation of the requirements and
```
TedEditor.bat
```
for starting the Application.

On Unices use
```
pip3 install -r requirements.txt
```
for installation of the requisites and
```
python3 ted-editor.py
```
for running the program.


## Mouse and Keyboard Cheatsheet

Hopefully not too outdated:
| Button/Key | Action |
|------------|--------|
| Middle Button | Move while pressed to drag Canvas View |
| Mouse Wheel | Zoom Canvas View |
| Left Button | On Control Points (blue)/Joint Points(pink)/Tangent Points(yellow)/Segments(grey) drag to move |
| Double Click | Control Points: remove point, Segments: insert point, Open Track anywhere: append point |
| Right Button | drag to select Control Points |
| Control Right Click | toggle selection of Control Points |
| Control+Left Button | drag to scale selected points |
| Shift+Left Button | drag to rotate selected points |
| Key o | Toggle Track Open/Closed |
| Key r | Reverse Track |
| Key r | Reverse Track |
| Control+Key z | Undo |
| Control+Key r | Redo |


