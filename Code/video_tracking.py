
# Imports #

from Liora_and_Orgad2.Code.general_functions import *
from Liora_and_Orgad2.Code.config import *
import cv2
from tqdm import tqdm

#######################################################################################################################################################

# Functions #

def draw_rect(frame, tl, br):
    img = cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
    return img

'''
def  do_tracking(matted_video_frames, box_list):
    matted_len = len(matted_video_frames)
    box_len = len(box_list)
    n = min(matted_len, box_len)
    new_frames = []

    for i in range(n-1):
        frame = matted_video_frames[i]
        tl, br = box_list[i]
        new_frame = draw_rect(frame, tl, br)
        new_frames.append(new_frame)

    return new_frames
'''

def  do_tracking(matted_video_frames, box_list, output_path):
    cap = cv2.VideoCapture(matted_video_frames)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor = True)
    
    box_len = len(box_list)
    n = min(n_frames, box_len)
    new_frames = []

    for i in range(n-1):
        frame = cap.read()
        tl, br = box_list[i]
        new_frame = draw_rect(frame[1], tl, br)

        out.write(new_frame)
    
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def get_tl_br2(frame):
    # find the indices where the value is 1
    indices = np.argwhere(frame == 255)
    # find the highest index in the y-axis (i.e., row)
    min_index_y = np.min(indices[:, 0])
    max_index_y = np.max(indices[:, 0])
    # find the highest index in the x-axis (i.e., column)
    min_index_x = np.min(indices[:, 1])
    max_index_x = np.max(indices[:, 1])

    tl = (max_index_x, min_index_y)
    br = (min_index_x, max_index_y)

    return tl, br

def get_box_list(binary_frames):
    cap = cv2.VideoCapture(binary_frames)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    box_list = []
    for frame_n in tqdm(range(n_frames)):
        ret1, frame = cap.read()
        (tl, br) = get_tl_br2(frame)
        box_list.append((tl, br))
    
    cap.release()
    cv2.destroyAllWindows()

    return box_list

#######################################################################################################################################################

# Main #



def tracking(binary_frames, matted_frames, output_path):
    print('')
    print("Stage 4: Tracking")


    box_list = get_box_list(binary_frames)

    do_tracking(matted_frames, box_list, output_path)


    if present_output_files_location:
        print('')
        print('Results ready @:\n', output_video_path)
