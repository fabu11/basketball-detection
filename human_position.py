# 47 ft from halfcourt to baseline
## center of rim is 63 inches from baseline
### 47 ft - 63 in = 41.75 ft from halfcourt to rim center
# 25 ft from center of halfcourt to edge of halfcourt

# pythag: sqrt(41.75^2 + 25^2) = 48.66 ft from edge of halfcourt to rim center
# camera details say: 24mm f1.78

# might be better to do it using homogrpahy - don't ahve to worry about any intrinsics
'''
HOMOGRAPHY INFO: relative to right corner seen in frame
Center of Free throw line: 19 feet in front, 25 ft to the left (1307, 771)

Right corner of court: 0ft in front, 0 ft to the left (1422, 661)

Left 3 point line baseline: 0ft in front, 50ft - 51 inches to the left (121, 727)
Right 3 point line baseline: 0ft in front, 51 inches to the left (1368, 665)

Left corner of free throw line: 19 ft in front, 31 ft to the left (1064, 797)
Right corner of the free throw line: 19 ft in front, 19 ft to the left (1463, 760)

Left paint corner at baseline: 0 ft in front, 31 ft to the left (706, 694)
Right paint corner at baseline: 0 ft in front, 19 ft to the left (1029, 679)

Center of 3 point line: 22 ft 1.75 inches in front, 25 ft to the left (1624, 844)

Bottom left block lane marker: 7ft in front, 31 ft to the left (811, 728)
2nd left lane marker from baseline: 11 ft in front, 31 ft to the left (872, 742)
3rd left lane marker from baseline: 14 ft 2 inches in front, 31 ft to the left (939, 762)
4th left lane marker from baseline: 17ft 4 inches in front, 31 ft to the left (1009, 784)

Bottom right block lane marker: 7ft in front, 19 ft to the left - cant see in frame due to glare
2nd right lane marker from baseline: 11 ft in front, 19 ft to the left - cant see in frame due to glare
3rd right lane marker from baseline: 14 ft 2 inches in front, 19 ft to the left (1322, 734)
4th right lane marker from baseline: 17ft 4 inches in front, 19 ft to the left (1412, 752)
'''

import numpy as np
import cv2
from human_detector import detect_human_in_frame
from utils import load_video

frame_points = np.array([
    [1307, 771], # Center of Free throw line
    [1422, 661], # Right corner of court
    [121, 727], # Left 3 point line baseline
    [1368, 665], # Right 3 point line baseline
    [1064, 797], # Left corner of free throw line
    [1463, 760], # Right corner of the free throw line
    [706, 694], # Left paint corner at baseline
    [1029, 679], # Right paint corner at baseline
    [1624, 844], # Center of 3 point line
    [811, 728], # Bottom left block lane marker
    [872, 742], # 2nd left lane marker from baseline
    [939, 762], # 3rd left lane marker from baseline
    [1009, 784], # 4th left lane marker from baseline
    [1322, 734], # 3rd right lane marker from baseline
    [1412, 752], # 4th right lane marker from baseline
])

court_rw_points = np.array([
    [19, 25], # Center of Free throw line
    [0, 0], # Right corner of court
    [0, 50 - 51/12], # Left 3 point line baseline
    [0, 51/12], # Right 3 point line baseline
    [19, 31], # Left corner of free throw line
    [19, 19], # Right corner of the free throw line
    [0, 31], # Left paint corner at baseline
    [0, 19], # Right paint corner at baseline
    [22 + 1.75/12, 25], # Center of 3 point line
    [7, 31], # Bottom left block lane marker
    [11, 31], # 2nd left lane marker from baseline
    [14 + 2/12, 31], # 3rd left lane marker from baseline
    [17 + 4/12, 31], # 4th left lane marker from baseline
    [14 + 2/12, 19], # 3rd right lane marker from baseline
    [17 + 4/12, 19], # 4th right lane marker from baseline
])

def compute_homography(pixel_coords, rw_coords):
    # referenced from Lab 6.2
    src = pixel_coords.reshape(-1, 1, 2).astype(np.float32)
    dst = rw_coords.reshape(-1, 1, 2).astype(np.float32)
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return H

def pixel_to_court_coords(f, homography):
    _, human_foot_pos = detect_human_in_frame(f)
    if human_foot_pos is None:
        return None
    
    human_foot_pos = np.array(human_foot_pos, dtype=np.float32).reshape(-1, 1, 2)
    
    realworld = cv2.perspectiveTransform(human_foot_pos, homography)
    return realworld

def main():
    h = compute_homography(frame_points, court_rw_points)
    frames = load_video("shots/make/16.mov", display=False)
    for i, f in enumerate(frames):
        court_coords = pixel_to_court_coords(f, h)
        if court_coords is not None:
            print(f"Person coordinates in frame {i}: {court_coords}")

main()