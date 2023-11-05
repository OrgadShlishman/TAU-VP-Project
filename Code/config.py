
# Imports
from Liora_and_Orgad2.Code.general_functions import *
import os

# Dirs defintions
project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
inputs_dir = project_dir+'/Inputs/'
outputs_dir = project_dir+'/Outputs/'
temp_dir = project_dir+'/Temp/'
video_format = '.MP4'
# Inputs
input_video_path = inputs_dir+'INPUT'+video_format
background_image_path = inputs_dir+'background.jpg'

# Outputs
stabilize_video_path = outputs_dir+'stabilizei'+video_format
extracted_video_path = outputs_dir+'extracted'+video_format
binary_video_path = outputs_dir+'binary'+video_format
alpha_video_path = outputs_dir+'alpha'+video_format
matted_video_path = outputs_dir+'matted'+video_format
output_video_path = outputs_dir+'OUTPUT'+video_format
trimap_video_path = outputs_dir+'trimap'+video_format

# Temp/Debug
stable_frames_path = temp_dir+'extracted_frames.npy'
extracted_frames_path = temp_dir+'extracted_frames.npy'
binary_frames_path = temp_dir+'binary_frames.npy'
present_output_files_location = 0