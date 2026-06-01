from human_position import pixel_to_court_coords, compute_homography, overlay_court_coords_on_frame, frame_points, court_rw_points
from ball_detector import draw_ball_on_frame
from utils import load_video, display_frames_video

frames = load_video("shots/make/16.mov", display=False)
h = compute_homography(frame_points, court_rw_points)
annotated = []
for i, f in enumerate(frames):
    all_annotated = draw_ball_on_frame(f.copy(), type = "trained")[0]
    court_coords = pixel_to_court_coords(f, h)
    all_annotated = overlay_court_coords_on_frame(all_annotated, court_coords)
    annotated.append(all_annotated)

display_frames_video(annotated, loop=True)
