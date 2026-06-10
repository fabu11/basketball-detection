import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc


def draw_halfcourt(ax=None, color="black", lw=2):
    """
    draws a half court with matplotlib using the same real-world dimensions
    as court_rw_points in human_position.py

    court coords are (x, y) = (feet from baseline, feet from right corner).
    plotted as: horizontal axis = court width (0-50 ft),
                vertical axis  = distance from baseline (0-47 ft),
    so the hoop sits near the bottom center like a normal shot chart
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    # outer boundary: 50 ft wide, 47 ft from baseline to halfcourt
    ax.add_patch(Rectangle((0, 0), 50, 47, fill=False, color=color, lw=lw))

    # hoop: center 5.25 ft (63 in) from baseline, mid-width; rim radius ~0.75 ft
    ax.add_patch(Circle((25, 5.25), 0.75, fill=False, color=color, lw=lw))

    # backboard: 6 ft wide, 4 ft from baseline
    ax.plot([22, 28], [4, 4], color=color, lw=lw)

    # paint / lane: 12 ft wide (y 19-31), from baseline to free throw line (19 ft)
    ax.add_patch(Rectangle((19, 0), 12, 19, fill=False, color=color, lw=lw))

    # free throw circle: radius 6 ft centered on the free throw line
    ax.add_patch(Circle((25, 19), 6, fill=False, color=color, lw=lw))

    # 3 point line: straight corner segments (51 in from each sideline) + arc
    ax.plot([4.25, 4.25], [0, 13.0], color=color, lw=lw)      # right corner
    ax.plot([45.75, 45.75], [0, 13.0], color=color, lw=lw)    # left corner

    # arc of radius ~22.15 ft centered on the hoop, spanning the two corners
    ax.add_patch(Arc((25, 5.25), 2 * 22.15, 2 * 22.15, angle=0, theta1=20.5, theta2=159.5, color=color, lw=lw))

    ax.set_xlim(-2, 52)
    ax.set_ylim(-2, 49)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def collect_shot_points(make_frames, miss_frames, court_coords_list, lookback=0):
    """
    given the make/miss frame indices and a per-frame list of (x, y) court
    coords (or None), returns two lists of (x, y) shot locations

    lookback: sample the person position this many frames before the event.
    the make/miss frame is when the ball reaches the rim (slightly after
    release), so a small lookback can better capture where the shooter was.
    if the sampled frame has no detection, walks backward to the nearest one.
    """
    def sample(idx):
        i = max(0, idx - lookback)
        while i >= 0:
            if court_coords_list[i] is not None:
                return court_coords_list[i]
            i -= 1
        return None

    makes = []
    for fr in make_frames:
        pt = sample(fr)
        if pt is not None:
            makes.append(pt)

    misses = []
    for fr in miss_frames:
        pt = sample(fr)
        if pt is not None:
            misses.append(pt)

    return makes, misses


def plot_shot_chart(make_points, miss_points, title="Shot Chart"):
    """
    plots makes (green circles) and misses (red x's) on a half court.
    points are (x, y) = (feet from baseline, feet from right corner).
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_halfcourt(ax)

    if make_points:
        # plot x = court width (point[1]), plot y = dist from baseline (point[0])
        mx = [p[1] for p in make_points]
        my = [p[0] for p in make_points]
        ax.scatter(mx, my, marker="o", s=110, facecolors="none", edgecolors="green", linewidths=2, label=f"Make ({len(make_points)})")

    if miss_points:
        xx = [p[1] for p in miss_points]
        xy = [p[0] for p in miss_points]
        ax.scatter(xx, xy, marker="x", s=110, color="red", linewidths=2, label=f"Miss ({len(miss_points)})")

    ax.legend(loc="upper right")
    ax.set_title(title)
    plt.show()
    return fig
