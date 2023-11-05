
# Imports #

from Liora_and_Orgad2.Code.general_functions import *
from Liora_and_Orgad2.Code.config import *
from tqdm import tqdm
#######################################################################################################################################################

# Functions #

def fixBorder(frame: np.ndarray) -> np.ndarray:
    (h, w, channels) = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (w, h))
    return frame


def get_stable_frames(video_frames: list, transforms_smooth: np.ndarray) -> list:
    video_params = get_video_parameters(cv2.VideoCapture(input_video_path))
    stabilized_frames = [fixBorder(video_frames[0])]
    for frame_idx, current_frame in enumerate((video_frames[:-1])):
        transform_matrix = transforms_smooth[frame_idx].reshape((3, 3))
        # Apply homography wrapping to the given frame
        frame_stabilized = cv2.warpPerspective(current_frame, transform_matrix, (video_params['width'], video_params['height']))
        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)
        stabilized_frames.append(frame_stabilized)

    # np.save(stable_frames_path, stabilized_frames)
    return stabilized_frames


def movingAverage(curve: np.ndarray, radius: int) -> np.ndarray:
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size)/window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed

def smooth(trajectory: np.ndarray) -> np.ndarray:
    smoothed_trajectory = np.copy(trajectory)
    SMOOTHING_RADIUS = 5
    for i in range(smoothed_trajectory.shape[1]):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory


def smooth_homography(homography_list: np.ndarray) -> np.ndarray:
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(homography_list, axis=0)
    smoothed_trajectory = smooth(trajectory)
    # Calculate difference between smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory
    # Calculate smooth transformation array
    homography_list_smooth = homography_list + difference
    return homography_list_smooth

def get_homography_list(video_frames: list) -> np.ndarray:
    video_params = get_video_parameters(cv2.VideoCapture(input_video_path))
    homography_list = np.zeros((video_params['frame_count'] - 1, 9), np.float32)
    prev_frame_gray = cv2.cvtColor(video_frames[0], cv2.COLOR_BGR2GRAY)
    for frame_idx, current_frame in enumerate((tqdm(video_frames[1:]))):
        prev_frame_pts = []
        curr_frame_pts = []
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Calculating optical flow and keeping only the valid features points
        detector = cv2.FastFeatureDetector.create(threshold=20)
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        kp1 = detector.detect(prev_frame_gray, None)
        kp2 = detector.detect(current_frame_gray, None)

        kp1, des1 = orb.compute(prev_frame_gray, kp1)
        kp2, des2 = orb.compute(current_frame_gray, kp2)

        matches = bf.match(des1, des2)

        prev_frame_pts.append(np.float32([kp1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2))
        curr_frame_pts.append(np.float32([kp2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2))
        prev_frame_pts = np.squeeze(np.array(prev_frame_pts))
        curr_frame_pts = np.squeeze(np.array(curr_frame_pts))

        homography, mask = cv2.findHomography(prev_frame_pts, curr_frame_pts, cv2.RANSAC, 5.0)
        homography_list[frame_idx] = homography.flatten()

        prev_frame_gray = current_frame_gray
    return homography_list

#######################################################################################################################################################

# Main #

"""
    To stabilize the shaky video, we will calculate homographies betweeen frames at times t and t+1. Using those homographies we will be able to warp the images at time t and get a more stable video
"""

'''
def stabilize_video(video_path):
    print('')
    print("Stabilizing Video:")
    progress = 1
    steps = 5

    print(' '+str(progress)+'/'+str(steps)+" Getting Input Video Frames")
    progress += 1
    video_frames = get_video_frames(video_path)

    print(' '+str(progress)+'/'+str(steps)+" Calculating Homographies")
    progress += 1
    homography_list = get_homography_list(video_frames)

    print(' '+str(progress)+'/'+str(steps)+" Smoothing Homographies")
    progress += 1
    homography_list_smooth = smooth_homography(homography_list)

    print(' '+str(progress)+'/'+str(steps)+" Warping Stable Frames using Homographies")
    progress += 1
    stabilized_frames = get_stable_frames(video_frames, homography_list_smooth)

    print(' '+str(progress)+'/'+str(steps)+" Writting Output Video")
    progress += 1
    write_video_from_frames(stabilized_frames, stabilize_video_path)

    if present_output_files_location:
        print('')
        print('Results ready @:\n', stabilize_video_path)

    return stabilized_frames
'''

def stabilize_video(video_path):
    print('')
    print("Stage 1: Stabilizing Video")
    video_frames = get_video_frames(video_path)
    homography_list = get_homography_list(video_frames)
    homography_list_smooth = smooth_homography(homography_list)
    stabilized_frames = get_stable_frames(video_frames, homography_list_smooth)
    write_video_from_frames(stabilized_frames, stabilize_video_path)

    if present_output_files_location:
        print('')
        print('Results ready @:\n', stabilize_video_path)

    return stabilized_frames