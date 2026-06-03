from human_position import pixel_to_court_coords, compute_homography, overlay_court_coords_on_frame, frame_points, court_rw_points
from ball_detector import draw_ball_on_frame
from utils import load_video, display_frames_video
from makemiss import makemiss
import cv2
import numpy as np

frames = load_video("shots/make/16.mov", display=False)
h = compute_homography(frame_points, court_rw_points)
annotated = []
ballpos_list = []

for i, f in enumerate(frames):
    # draw circle around ball on the frame
    ball_drawn, _, ballpos = draw_ball_on_frame(f.copy(), type = "trained")
    ballpos_list.append(ballpos)
    
    # get person realworld location
    court_coords = pixel_to_court_coords(f, h)
    ball_drawn = overlay_court_coords_on_frame(ball_drawn, court_coords)
    annotated.append(ball_drawn)

# get the frame indices where the ball went through the hoop
hoop_coords = (971, 377)
make_frames = makemiss(hoop_coords, ballpos_list)
for made_frame_i in make_frames:
    # put green filter on that 10 frame range
    green_filter = np.full_like(annotated[made_frame_i], (0, 255, 0))
    for i in range(max(0, made_frame_i - 5), min(len(annotated) - 1, made_frame_i + 5)): 
        
        annotated[i] = cv2.addWeighted(annotated[i], 0.6, green_filter, 0.4, 0)

display_frames_video(annotated, loop=True)
