# data/download_dataset.py

# Install Roboflow SDK
!pip install roboflow

from roboflow import Roboflow

# Authenticate
rf = Roboflow(api_key="6nm9viAPF2N0DrpLr3Em")

# Connect to project and version
project = rf.workspace("crime-scene-analys").project("examensarbete-2")
version = project.version(1)

# Download dataset for YOLOv11
dataset = version.download("yolov11")
