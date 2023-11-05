
# Imports #

from Liora_and_Orgad.Code.general_functions import *
from Liora_and_Orgad.Code.config import *
from scipy.stats import gaussian_kde

#######################################################################################################################################################

# Functions #

def find_indices(src, value, how_many):
    """ find indices for a specific value in the src, returns also *how_many* points"""
    indices = np.where(src == value)
    if len(indices[0]) < how_many:
        print("Sorry, not enough points in src, we will use the amount found")
        how_many = len(indices[0])
    indices_shuffle = np.random.choice(len(indices[0]), how_many)
    return np.column_stack((indices[0][indices_shuffle], indices[1][indices_shuffle])), how_many

def check_if_new(dict, value, pdf):
    """ checking if the value already exists in the dict """
    if value in dict:
        return dict[value]
    else:
        dict[value] = pdf(value)[0]
        return dict[value]

# TODO: Recheck function2
def init_vars(stabilize_video_path):
    np.random.seed(0)
    vid = cv2.VideoCapture(stabilize_video_path)
    Parameters = get_video_parameters(vid)
    frame_count, fps, width, height = Parameters["frame_count"], Parameters["fps"], Parameters["width"], Parameters[
        "height"]

    fgbg = cv2.createBackgroundSubtractorKNN()
    how_many = 20
    how_many_foreground = 30

    frames = get_video_frames(stabilize_video_path)
    extracted_frames = []

    foreground_values = np.empty((how_many_foreground * frame_count, 3))
    background_values = np.empty((how_many * frame_count, 3))

    output_frames = np.zeros((frame_count, height, width), dtype=np.uint8)
    frames_mask = np.zeros((frame_count, height, width), dtype=np.uint8)

    # init
    return fgbg, frames, frames_mask, background_values, foreground_values, extracted_frames, output_frames

def get_initial_mask(fgbg, frames, frames_mask, max_iter):
    for j in range(max_iter):
        # print(f"Iteration_number {j}")
        for index_frame, frame in enumerate(frames):
            transformed_frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_sv = transformed_frame_HSV[:, :, 1:]  # taking into account only the Saturation and Value
            fgMask = fgbg.apply(frame_sv)
            # fgMask = (fgMask > 200).astype(np.uint8)
            fgMask = (fgMask > 130).astype(np.uint8)
            frames_mask[index_frame] = fgMask

    return frames_mask

def improving_mask(frames, frames_mask, background_values, foreground_values, BLUE_MASK_THR):
    start_foreground = 0
    start = 0
    how_many = 20
    how_many_foreground = 30

    for index_frame, frame in enumerate(frames):
        mask = frames_mask[index_frame]
        blue_frame, _, _ = cv2.split(frame)
        # performing morphological operators to clean and restore
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Dilation followed by Erosion
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # finding contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        person_mask = np.zeros(mask.shape)
        cv2.fillPoly(person_mask, pts=[contours[0]], color=1)

        blue_mask = (blue_frame < BLUE_MASK_THR).astype(np.uint8)
        person_mask = (person_mask * blue_mask)
        frames_mask[index_frame] = person_mask

        # there are still noises in the mask, therefore would like to create a distribution for fore/background
        indices_foreground, how_many_foreground = find_indices(person_mask, 1, how_many_foreground)
        indices_background, how_many = find_indices(person_mask, 0, how_many)
        foreground_values[start_foreground:start_foreground + how_many_foreground] = frame[indices_foreground[:, 0],
                                                                                     indices_foreground[:, 1], :]
        background_values[start:start + how_many] = frame[indices_background[:, 0], indices_background[:, 1], :]
        start = start + how_many
        start_foreground = start_foreground + how_many_foreground

    return background_values, foreground_values, frames_mask

def get_binary_and_extracted_frames(background_values, foreground_values, frames, frames_mask, extracted_frames, binary_frames):

    pdf_foreground = gaussian_kde(np.asarray(foreground_values).T, bw_method=0.95)
    pdf_background = gaussian_kde(np.asarray(background_values).T, bw_method=0.95)

    # so we won't have to calculate the same value twice
    pdf_foreground_dict = dict()
    pdf_background_dict = dict()

    pdf_foreground_dict = dict()
    pdf_background_dict = dict()

    for index_frame, frame in (enumerate(frames)):
        mask = frames_mask[index_frame]
        new_mask = np.zeros_like(mask)
        positions = np.where(mask == 1)

        # checking the probability of each pixel in the mask, to which label it is more likely to belong
        check_probability_foreground = np.fromiter(
            map(lambda elem: check_if_new(pdf_foreground_dict, elem, pdf_foreground),
                map(tuple, frame[positions])), dtype=float)
        check_probability_background = np.fromiter(
            map(lambda elem: check_if_new(pdf_background_dict, elem, pdf_background),
                map(tuple, frame[positions])), dtype=float)
        new_mask[positions] = (check_probability_foreground > check_probability_background).astype(np.uint8)

        # more fixes for the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        new_mask = cv2.erode(new_mask, kernel).astype(np.uint8)
        contours, _ = cv2.findContours(new_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        person_mask = np.zeros((new_mask.shape), dtype=np.uint8)
        cv2.fillPoly(person_mask, pts=[contours[0]], color=1)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, np.ones((15, 15)))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             .astype(np.uint8)
        person_mask[person_mask == 1] = 255
        binary_frames[index_frame] = person_mask
        extracted_frames.append(cv2.bitwise_and(frame, frame, mask=person_mask))

    return extracted_frames, binary_frames


#######################################################################################################################################################

# Main #

def subtract_background(stabilize_video_path):
    print('')
    print("Subtracting Video Background:")

    progress = 1
    steps = 4

    print(' '+str(progress)+'/'+str(steps)+" Initializing variables")
    progress += 1
    fgbg, frames, frames_mask, background_values, foreground_values, extracted_frames, output_frames = init_vars(stabilize_video_path)

    print(' '+str(progress)+'/'+str(steps)+" Creating initial masks")
    progress += 1
    initial_mask = get_initial_mask(fgbg, frames, frames_mask, max_iter=5)

    print(' '+str(progress)+'/'+str(steps)+" Improving masks")
    progress += 1
    background_values, foreground_values, frames_mask = improving_mask(frames, initial_mask, background_values, foreground_values, BLUE_MASK_THR=140)

    print(' '+str(progress)+'/'+str(steps)+" Writing output videos")
    progress += 1
    extracted_frames, binary_frames = get_binary_and_extracted_frames(background_values, foreground_values, frames, frames_mask, extracted_frames, output_frames)

    write_video_from_frames(extracted_frames, extracted_video_path)
    write_video_from_frames(binary_frames, binary_video_path, isColor=False)

    if present_output_files_location:
        print('')
        print('Results ready @:\n', extracted_video_path, '\n', binary_video_path)

    return binary_frames, extracted_frames