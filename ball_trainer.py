from ultralytics import YOLO

# dataset from https://universe.roboflow.com/thesis-physics/basketball-1zkue
def train():
    model = YOLO('yolov8m.pt')
    model.train(
        data='dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        device=0,
        project='runs/yolo_weights',
        name='train',
    )

if __name__ == '__main__':
    train()
