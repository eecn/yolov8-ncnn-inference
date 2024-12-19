import os
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO("yolo11s-pose.pt")
    results = model.train(data="/mnt/d/workspace/github/yolov8-ncnn-inference/ultralytics/cfg/datasets/wider-face.yaml", batch=16,epochs=100, imgsz=640)