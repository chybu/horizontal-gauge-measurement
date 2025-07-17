import pyrealsense2 as rs
import numpy as np
import cv2, os, pickle, shutil


bag_file = "path_to_.bag_file"


output_folder = "output_folder"
if os.path.exists(output_folder): shutil.rmtree(output_folder)
os.makedirs(output_folder)
color_folder = os.path.join(output_folder, "color")
depth_folder = os.path.join(output_folder, "depth")
os.makedirs(color_folder)
os.makedirs(depth_folder)

pipeline = rs.pipeline()
config = rs.config()
align_to = rs.stream.color
align = rs.align(align_to)

config.enable_device_from_file(bag_file, False)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
device = profile.get_device()
playback = device.as_playback()
playback.set_real_time(False)

frame_count = 0
interval = 6
image_count = 1
color_intrin = None

while True:
    # print(frame_count)
    try:
        frames = pipeline.wait_for_frames(timeout_ms=3000)
    except RuntimeError: break
    
    if not frames: break
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
            
    if not depth_frame or not color_frame:
        continue
    
    if not color_intrin: 
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        intrinsics_dict = {
            'width': color_intrin.width,
            'height': color_intrin.height,
            'ppx': color_intrin.ppx,
            'ppy': color_intrin.ppy,
            'fx': color_intrin.fx,
            'fy': color_intrin.fy,
            'model': int(color_intrin.model),
            'coeffs': list(color_intrin.coeffs)
        }
        with open(os.path.join(output_folder, 'color_intrinsics.pkl'), 'wb') as f:
            pickle.dump(intrinsics_dict, f)
            
    if frame_count % interval == 0 and frame_count>100:
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(color_folder, f'{image_count:08d}.jpg'), color_image)
        cv2.imwrite(os.path.join(depth_folder, f'{image_count:08d}.png'), depth_image)
        image_count+=1
    frame_count+=1
