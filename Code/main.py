import sys
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the path two levels above the current directory
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(parent_dir)


from Liora_and_Orgad2.Code.video_stabilization import *
#from Liora_and_Orgad2.Code.background_subtraction import *
from Liora_and_Orgad2.Code.Binary_Video import *
from Liora_and_Orgad2.Code.video_matting import *
from Liora_and_Orgad2.Code.video_tracking import *
from Liora_and_Orgad2.Code.config import *
import time

start_time = time.time()

stable_frames = stabilize_video(input_video_path)
background_sub(stabilize_video_path, binary_video_path, extracted_video_path, frameId = 91)
#matting(stabilize_video_path, binary_video_path, background_image_path)
matting_func_1(background_image_path, stabilize_video_path, binary_video_path, alpha_video_path, trimap_video_path, matted_video_path)
tracking(binary_video_path, matted_video_path, output_video_path)

print('')
print('Results ready @:\n', output_video_path)

end_time = time.time()
print("\nRun Time:", end_time-start_time)

