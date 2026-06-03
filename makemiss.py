from ball_detector import detect_ball_yolo_trained
from utils import load_video
import os

def makemiss(hoop_pixel_coords: tuple, ball_positions: list):
    '''
    DONT DELETE: COORDS OF HOOP: (971, 377)
    '''
    thresh = 10
    saw_above = False
    hoop_x, hoop_y = hoop_pixel_coords

    make_frames = []
    for i, ballpos in enumerate(ball_positions):
        if ballpos is not None:
            bx, by = ballpos
            if abs(bx - hoop_x) < thresh:
                if by < hoop_y:
                    saw_above = True
                    thresh = 20
                elif saw_above and (hoop_y + thresh) < by < (hoop_y + thresh + 150):
                    saw_above = False
                    thresh = 10
                    make_frames.append(i)
                elif saw_above:
                    pass
                    # LIL HARDER SO DO LATER: do we wanna make it show a red or something meaning the ball missed

    return make_frames


# note: old code - prolly not gon work
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
