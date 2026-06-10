import human_detector
from human_position import pixel_to_court_coords, compute_homography, overlay_court_coords_on_frame, smallvids_frame_points, court_rw_points, longvids_frame_points
from ball_detector import draw_ball_on_frame
from utils import load_video, display_frames_video, apply_color_filter
from makemiss import makemiss, makemiss2
from human_detector import add_bounding_box

smallvids_hoop_coords = (971, 377)
longvids_hoop_coords = (647, 245)

frames = load_video("shots/miss/34.mov", display=False)
h = compute_homography(smallvids_frame_points, court_rw_points)
annotated = []
ballpos_list = []

for i, f in enumerate(frames):
    # draw circle around ball on the frame
    ball_drawn, _, ballpos = draw_ball_on_frame(f.copy(), type = "trained")
    ball_drawn = human_detector.add_bounding_box(f)
    ballpos_list.append(ballpos)
    
    # get person realworld location
    court_coords = pixel_to_court_coords(f, h)
    ball_drawn = overlay_court_coords_on_frame(ball_drawn, court_coords)
    annotated.append(ball_drawn)

# get the frame indices where the ball went through the hoop
make_frames, miss_frames = makemiss2(smallvids_hoop_coords, ballpos_list)
apply_color_filter(annotated, make_frames, (0, 255, 0))
apply_color_filter(annotated, miss_frames, (0, 0, 255))

display_frames_video(annotated, loop=True)
