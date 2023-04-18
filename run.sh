#！/bin/bash

# win10
pyinstaller -D -w C:\Users\js\Documents\myMediapipe\CVideo.py --add-data="C:\Users\js\miniconda3\Lib\site-packages\mediapipe\modules;mediapipe/modules" --add-data="C:\Users\js\Documents\myMediapipe\action.h5;."