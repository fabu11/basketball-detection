import torch
from ultralytics import YOLO

# dataset from https://universe.roboflow.com/thesis-physics/basketball-1zkue
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def train():
    model = YOLO('yolov8m.pt')
    model.train(
        data='dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        device=DEVICE,
        project='runs/yolo_weights',
        name='train',
    )

if __name__ == '__main__':
    train()
