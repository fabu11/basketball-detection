import cv2 
from ultralytics import YOLO #pyright: ignore 
from utils import display_frame, load_video

model = YOLO("yolov8n.pt")

def add_bounding_box(f):
    """
    adds human bounding box + foot point to cv2 frame
    """
    h, p = detect_human_in_frame(f, bottom_percent=0.5, conf_min=.20)
    if h is not None:
        x1, y1, x2, y2 = h.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if p is not None:
        cv2.circle(f, p, 6, (0, 0, 255), -1)

    return f

def detect_human_in_frame(f, bottom_percent=0.5, conf_min=0.15):
    """
    takes frame f and returns bounding box for human (with highest confidence)
    also returns a point to determine where the feet of person are

    bottom_percent will filter out confidence scores that lie above the threshold.
    """
    results = model(f, classes=[0], conf=conf_min, verbose=False) # classes=[0] --> human detection
    boxes = results[0].boxes
    if(len(boxes) == 0):
        return None, None
    
    y_min = f.shape[0]  * (1-bottom_percent)
    foot_area = boxes.xyxy[:, 3]
    valid = boxes[foot_area >= y_min]

    if(len(valid) == 0):
        return None, None

    # get foot point
    h = valid[valid.conf.argmax()] # highest conf of filtered results
    x1, _, x2, y2 = h.xyxy[0].cpu().numpy()
    foot_point = (int((x1 + x2) / 2), int(y2)) # the bottom center of the bounding box



    # if(add_bb):
    #     cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.circle(f, foot_point, 6, (0, 0, 255), -1)

    return h, foot_point



if __name__ == "__main__":
    frames = load_video("shots/make/5.mov", display=False)
    for f in frames:
        display_frame(add_bounding_box(f))
    if len(frames) == 0:
        print("Did not extract frames")
        exit(1)



