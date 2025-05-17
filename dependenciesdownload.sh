#!/bin/bash

# Update package list and install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv

# Install Python dependencies
pip3 install ultralytics opencv-python numpy

# Download YOLOv8n model weights
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"