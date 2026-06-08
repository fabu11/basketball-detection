import os
from human_position import (pixel_to_court_coords, compute_homography, overlay_court_coords_on_frame, smallvids_frame_points, court_rw_points)
from ball_detector import draw_ball_on_frame
from utils import load_video, display_frames_video, apply_color_filter
from makemiss import makemiss2
from shot_chart import collect_shot_points, plot_shot_chart

FOLDER = "basketball-detection-main\\shots\\smallvids"      # put multiple small videos in this folder
SMALLVIDS_HOOP = (971, 377)     # hoop pixel location for the small vids
DISPLAY = True                  # set False to skip watching each clip

h = compute_homography(smallvids_frame_points, court_rw_points)

def process_video(path):
    """runs the full pipeline on one clip and returns its (make, miss) shot
    points in court coordinates. plays the annotated clip if DISPLAY is on."""
    frames = load_video(path, display=False)
    if not frames:
        return [], []

    annotated = []
    ballpos_list = []
    court_coords_list = []

    for f in frames:
        # draw circle around ball on the frame
        ball_drawn, _, ballpos = draw_ball_on_frame(f.copy(), type="trained")
        ballpos_list.append(ballpos)

        # get person realworld location
        court_coords = pixel_to_court_coords(f, h)

        # store realworld location in list (for shot chart later)
        if court_coords is not None:
            court_coords_list.append((float(court_coords[0][0][0]), float(court_coords[0][0][1])))
        else:
            court_coords_list.append(None)

        ball_drawn = overlay_court_coords_on_frame(ball_drawn, court_coords)
        annotated.append(ball_drawn)

    # frame indices where the ball went through the hoop
    make_frames, miss_frames = makemiss2(SMALLVIDS_HOOP, ballpos_list)
    apply_color_filter(annotated, make_frames, (0, 255, 0))
    apply_color_filter(annotated, miss_frames, (0, 0, 255))

    if DISPLAY:
        # plays one video, press escape for next video
        display_frames_video(annotated, name=os.path.basename(path), loop=False)

    return collect_shot_points(make_frames, miss_frames, court_coords_list, lookback=5)


# loop over every video in the folder, track all makes and misses
all_make_pts = []
all_miss_pts = []

for name in sorted(os.listdir(FOLDER)):
    if name.lower().endswith(".mov"):
        path = os.path.join(FOLDER, name)
        print(f"processing {path}")
        mp, xp = process_video(path)
        all_make_pts.extend(mp)
        all_miss_pts.extend(xp)

print(f"total makes: {len(all_make_pts)}, total misses: {len(all_miss_pts)}")

# one combined shot chart across every clip in the folder
plot_shot_chart(all_make_pts, all_miss_pts, title="All small vids — shot chart")