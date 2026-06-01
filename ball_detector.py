from utils import display_frame, load_video, display_frames_video
import cv2
import numpy as np
import skimage
import torch
from skimage.transform import hough_circle, hough_circle_peaks
from ultralytics import YOLO #pyright: ignore

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

trained_model = YOLO("runs/detect/runs/yolo_weights/train/weights/best.pt")
pretraied_model = YOLO("yolov8l")

def orange_mask(f):
    """
    masks out the non-orange colors from the frame f
    """
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV) # frame --> hsv colors

    # used color picker to narrow range. might still need work
    lower = np.array([10, 130,  80])
    upper = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    # slide 3x3 kernel to first remove any small white pixels
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    # close small black holes in areas of orange (white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    return mask

def detect_ball_hough(f):
    """
    uses hough transform to determine the pixel location of the ball similar to HW2
    """

    # the background is too noisy, so this mask will try to filter out non-orange colors
    mask = orange_mask(f)

    edges = skimage.feature.canny(mask > 0)
    hough_radii = np.arange(10, 20)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    if len(accums) == 0:
        return None, edges
    return (int(cx[0]), int(cy[0]), int(radii[0])), edges

def detect_ball_yolo_trained(f, conf_min=0.01):
    """
    uses yolo trained model to detect the balls pixel location
    """
    results = trained_model(f, imgsz=1280, conf=conf_min, verbose=False, device=DEVICE)
    boxes = results[0].boxes
    if(len(boxes) == 0):
        return None, None

    best_ball = boxes[boxes.conf.argmax()]
    x1, y1, x2, y2 = best_ball.xyxy[0].cpu().numpy()
    # calc circle from bounding box
    cx, cy = int((x1+x2) / 2), int((y1+y2) / 2) # center of box is center of circle
    r = int(max(x2-x1, y2-y1) / 2) # take width of box as 2r
    return (cx, cy, r), None # None is here so that it is the same return type as hough

def detect_ball_yolo_base(f, conf_min=0.01):
    """
    uses yolo pretrained model to detect the balls pixel location
    """
    results = pretraied_model(f, classes=[32], imgsz=1280, conf=conf_min, verbose=False, device=DEVICE)
    boxes = results[0].boxes
    if(len(boxes) == 0):
        return None, None

    best_ball = boxes[boxes.conf.argmax()]
    x1, y1, x2, y2 = best_ball.xyxy[0].cpu().numpy()
    # calc circle from bounding box
    cx, cy = int((x1+x2) / 2), int((y1+y2) / 2) # center of box is center of circle
    r = int(max(x2-x1, y2-y1) / 2) # take width of box as 2r
    return (cx, cy, r), None # None is here so that it is the same return type as hough

# for fallback in detect_ball_both 
last_yolo_bb = None

def detect_ball_both(f, conf_min=0.25, fallback_pad_factor=2.0):
    """
    yolo + hough. yolo finds the ball, then hough refines the circle within
    the yolo bbox. if yolo misses, hough is run on an expanded area around
    the previous yolo detection. if hough also misses on that fallback, we
    just return the frame with no ball
    """
    global last_yolo_bb

    results = trained_model(f, imgsz=1280, conf=conf_min, verbose=False, device=DEVICE)
    boxes = results[0].boxes
    yolo_hit = len(boxes) > 0

    if yolo_hit:
        best = boxes[boxes.conf.argmax()]
        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)
        last_yolo_bb = (x1, y1, x2, y2)
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2
        pad = 20
    elif last_yolo_bb is not None:
        rx1, ry1, rx2, ry2 = last_yolo_bb #pyright: ignore
        # ball has moved since the last detection, so widen the search area
        w, h = rx2 - rx1, ry2 - ry1
        pad = int(max(w, h) * fallback_pad_factor)
    else:
        return None, None

    # create the region of interest using yolo location
    H, W = f.shape[:2]
    cx1 = max(0, rx1 - pad)
    cy1 = max(0, ry1 - pad)
    cx2 = min(W, rx2 + pad)
    cy2 = min(H, ry2 + pad)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    crop = gray[cy1:cy2, cx1:cx2]

    # scale hough radius search to the known ball size from the bbox
    expected_r = max(rx2 - rx1, ry2 - ry1) // 2
    hough_radii = np.arange(max(5, int(expected_r * 0.6)), max(15, int(expected_r * 1.4)))
    edges = skimage.feature.canny(crop, sigma=2.0)
    hough_res = hough_circle(edges, hough_radii)
    accums, peak_x, peak_y, peak_r = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    if len(accums) > 0:
        # offset crop-local coords back to full-frame coords
        return (int(peak_x[0]) + cx1, int(peak_y[0]) + cy1, int(peak_r[0])), edges

    if yolo_hit:
        # hough failed. just use the yolo
        cx, cy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
        return (cx, cy, expected_r), edges

    return None, edges

def draw_ball_on_frame(f, type="hough"):
    """
    detects the ball on frame similar to HW2. 

    type determines which method to find the ball:
        default - hough, trained, pretrained, both
    """
    if type == "trained":
        ball, e = detect_ball_yolo_trained(f)
    elif type == "pretrained":
        ball, e = detect_ball_yolo_base(f)
    elif type == "both":
        ball, e = detect_ball_both(f)
    else: # hough
        ball, e = detect_ball_hough(f)

    if(ball is None):
        return f, None

    cx, cy, r = ball 
    cv2.circle(f, (cx, cy), r, (0, 165, 255), 2)   # orange outline
    cv2.circle(f, (cx, cy), 2,  (0, 0, 255),  3)    # red center dot
    return f, e

def vconcat_mask_and_ball(f, detectionmethod):
    mask = orange_mask(f)
    frame_with_ball, e = draw_ball_on_frame(f.copy(), type=detectionmethod)
    if(e is None):
        return f
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    edge_bgr = cv2.cvtColor(e.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    return cv2.resize(cv2.vconcat([mask_bgr, edge_bgr, frame_with_ball]), None, fx=0.25, fy=0.25)


if __name__ == "__main__":
    filename = "shots/make/14.mov"
    frames = load_video(filename, display=False)

    types = ["hough", "trained", "pretrained", "both"]
    all_annotated = {}
    for t in types:
        all_annotated[t] = [draw_ball_on_frame(f.copy(), type=t)[0] for f in frames]

    import cv2
    n = len(frames)
    while True:
        for i in range(n):
            for t, annotated in all_annotated.items():
                cv2.imshow(t, annotated[i])
            if cv2.waitKey(33) & 0xFF == 27:
                cv2.destroyAllWindows()
                break
        else:
            continue
        break

    # h = "hough"
    # annotated = [vconcat_mask_and_ball(f, h) for f in frames]
    # display_frames_video(annotated, name=filename, loop=True)
