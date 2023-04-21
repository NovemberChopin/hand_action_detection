#！/bin/bash

# -F 打包成单个文件
# -D 打包成文件夹
# --add-data: 添加打包文件，注意linux中以":"隔开，win中以";"隔开

# linux
pyinstaller -F -w <project-dir>/CVideo.py --add-data="<virtual-env>/site-packages/mediapipe/modules:mediapipe/modules" --add-data="<project-dir>/Model/action_13.h5:./Model/"
# example:
# pyinstaller -D -w /home/js/myProjects/myMediapipe/CVideo.py --add-data="/home/js/.conda/envs/tf/lib/python3.7/site-packages/mediapipe/modules:mediapipe/modules" --add-data="/home/js/myProjects/myMediapipe/Model/action_13.h5:./Model/"


# win10
pyinstaller -F -w <project-dir>/CVideo.py --add-data="<virtual-env>/site-packages/mediapipe/modules;mediapipe/modules" --add-data="<project-dir>/Model/action_13.h5;./Model/"
# example:
# pyinstaller -D -w C:\Users\js\Documents\myMediapipe\CVideo.py --add-data="C:\Users\js\miniconda3\Lib\site-packages\mediapipe\modules;mediapipe/modules" --add-data="C:\Users\js\Documents\myMediapipe\action13.h5:."