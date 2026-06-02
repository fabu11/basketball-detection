from ball_detector import detect_ball_yolo_trained
from utils import load_video
import os

def makemiss(hoop_pixel_coords: tuple, ball_pixel_coords: list[tuple[int, int]]):
    '''
    DONT DELETE: COORDS OF HOOP: (971, 377)
    '''
    thresh = 10
    hoop_x, hoop_y = hoop_pixel_coords

    saw_above = False
    for ball_x, ball_y in ball_pixel_coords:
        if abs(ball_x - hoop_x) < thresh:
            if ball_y < hoop_y:
                saw_above = True
                thresh = 20
            elif saw_above and (hoop_y + thresh) < ball_y < (hoop_y + thresh + 150):
                return True

    return False

def run():
    miss_videos_dir = "shots/miss"
    make_videos_dir = "shots/make"
    videos = []
    for vidname in os.listdir(make_videos_dir):
        if vidname.endswith(".mov"):
            # if vidname in ["4.mov", "8.mov"]:
            videos.append(os.path.join(make_videos_dir, vidname))
    
    hoop_coords = (971, 377)

    for vid in videos:
        ball_coords = []
        for f in load_video(vid, display = False):
            loc, error = detect_ball_yolo_trained(f)
            if loc:
                ball_coords.append((loc[0], loc[1]))
        
        print(f"video {vid}: {makemiss(hoop_coords, ball_coords)}")

run()