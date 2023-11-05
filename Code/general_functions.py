
from Liora_and_Orgad2.Code.config import *
import cv2
import numpy as np

def get_video_parameters(capture):

    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}
def get_video_frames(video_path):

    video = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)

    video.release()

    return frames

def write_video_from_frames(frames, output_path, isColor=True):
    """ writes the output video with the given frames"""
    cap = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(output_path, fourcc, params["fps"], (params["width"], params["height"]), isColor)
    for frame in frames:
        video_out.write(frame)
    video_out.release()
