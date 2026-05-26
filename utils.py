import cv2 
import os

def load_video(filename, display=True):
    "loads video from filepath and returns an array of its frames"
    if not os.path.exists(filename):
        print(f"filename: {filename} does not exist")
        return []
    
    video = cv2.VideoCapture(filename)
    frames = []

    while 1:
        r, f = video.read()
        if not r:
            break

            
        if display:
            display_frame(f)

        frames.append(f)

    video.release()
    return frames


def display_frame(f):
    """
    displays frame f, press esc to close
    """
    cv2.imshow('f', f)
    while 1:
        if(cv2.waitKey(1) & 0xFF == 27):
            break

def display_frames_video(frames, ms=33, name="video"):
    """
    takes array of frames and plays them as a video
    """
    for f in frames:
        cv2.imshow(name, f)
        if(cv2.waitKey(ms) & 0xFF == 27):
            break
    cv2.destroyWindow(name)
        




