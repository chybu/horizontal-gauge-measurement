
import os
import cv2
import torch
import pickle
import numpy as np
import gc
import pyrealsense2 as rs
import math
from multiprocessing import Process, Manager
from sam2.build_sam import build_sam2_video_predictor
from time import time
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

def get_left_edge_coordinates(mask):
    coords = []
    H, W = mask.shape
    for row in range(H):
        ones = np.where(mask[row] == 1)[0]
        if ones.size > 0:
            leftmost_col = ones[0]
            coords.append((int(leftmost_col), int(row)))
    return coords

def get_right_edge_coordinates(mask):
    coords = []
    H, W = mask.shape
    for row in range(H):
        ones = np.where(mask[row] == 1)[0]
        if ones.size > 0:
            rightmost_col = ones[-1]
            coords.append((int(rightmost_col), int(row)))
    return coords

def denoise_depth(depth_image, y_list, fill=False):
    if fill:
        depth_image = depth_image.astype(np.float32)
        valid_mask = (depth_image != 0) & (depth_image <= 4000)
        valid_mask_float = valid_mask.astype(np.float32)
        depth_filled = np.where(valid_mask, depth_image, 0).astype(np.float32)

        sum_neighbors = cv2.blur(depth_filled, (5, 5))
        count_neighbors = cv2.blur(valid_mask_float, (5, 5))

        average_neighbors = sum_neighbors / (count_neighbors + 1e-5)
        depth_image[~valid_mask] = average_neighbors[~valid_mask]
    for i in y_list:
        depth_image = cv2.GaussianBlur(depth_image, (1, i), sigmaX=1, sigmaY=max(10, i/2))
    return depth_image

def to3d(point, intrin, depth_img):
    depth = depth_img[point[1], point[0]]/1000
    point3d = rs.rs2_deproject_pixel_to_point(intrin, point, depth)
    return point3d

def getDistance(left_point, right_point, depth_path, intrin):
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_image = median_filter(depth_image, (1, 5))
    # filter out plants and unwanted slopes
    baseline = denoise_depth(depth_image, [401], fill=True)
    difference = depth_image - baseline
    spike_mask = difference < -20
    depth_image[spike_mask] = baseline[spike_mask]
    # replace the slopes with the new depth. This would avoid the slope in the final denoise
    depth_image = denoise_depth(depth_image, [31, 51, 101])
    
    points_3d = []
    for point in [left_point, right_point]:
        points_3d.append(to3d(point, intrin, depth_image))
        
    dist_meter = math.sqrt(
        math.pow(points_3d[0][0] - points_3d[1][0], 2)
        + math.pow(points_3d[0][1] - points_3d[1][1],2)
        + math.pow(points_3d[0][2] - points_3d[1][2], 2)
        ) 
    dist_inch = round(dist_meter*39.3701,2)
    return dist_inch

def neighborsGenerator(original_cordinate:list, cordinate_interval = 15, max = 40):
    y_list = list(range(original_cordinate[1]-cordinate_interval, (original_cordinate[1]-cordinate_interval*max-1), -cordinate_interval))
    y_list.reverse()
    y_list.append(original_cordinate[1])
    y_list.extend(list(range(original_cordinate[1]+cordinate_interval, (original_cordinate[1]+cordinate_interval*max+1), cordinate_interval)))
    neighbors = []
    for y in y_list:
        if y<99 or y>379: continue
        neighbors.append([original_cordinate[0],y])
        
    return neighbors

def get_y_bounds(mask):
    ys, xs = np.where(mask)
    return ys.min(), ys.max()

def getCentroid(mask1, mask2, img_width = 640):
    ymin1, ymax1 = get_y_bounds(mask1)
    ymin2, ymax2 = get_y_bounds(mask2)

    crop_ymin = max(ymin1, ymin2)
    crop_ymax = min(ymax1, ymax2)

    y = int(np.round(crop_ymin + ((crop_ymin+crop_ymax)/2)))
    x = img_width/2
    
    return (x, y)

def denoise_mask(mask, min_area=300, size=7):
    # Make sure it's uint8
    mask_uint8 = (mask.astype(np.uint8) * 255)

    # Morphological opening and closing
    kernel = np.ones((size, size), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    # Connected components filtering
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    cleaned = np.zeros_like(mask_uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned > 0

def extract_numframes_width_height(image_folder):
    files = os.listdir(image_folder)
    numframes = len(files)
    img = cv2.imread(os.path.join(image_folder, files[0]))
    height, width = img.shape[:2]
    return numframes, height, width

def load_multi_boxes(txt_path):
    boxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            x, y, w, h = map(int, line.strip().split(','))
            boxes.append((x, y, x + w, y + h))
    return boxes

def run_tracking(obj_id, box, shared_masks, temp_img_folder, width, height, model_cfg, checkpoint):
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")
    state = predictor.init_state(temp_img_folder, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)
    predictor.add_new_points_or_box(state, box=box, frame_idx=0, obj_id=obj_id)
    masks = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for frame_idx, object_ids, pred_masks in predictor.propagate_in_video(state):
            for o_id, mask in zip(object_ids, pred_masks):
                if o_id == obj_id:
                    masks[frame_idx] = mask[0].cpu().numpy() > 0.0
    shared_masks[obj_id] = masks
    del predictor
    del state
    gc.collect()
    torch.cuda.empty_cache()

def main(data_path, txt_path, model_name="base_plus"):
    
    out_path = "output.mp4"
    temp_img_folder = os.path.join(data_path, "color")
    depth_folder = os.path.join(data_path, "depth")
    intrin_path = os.path.join(data_path, "color_intrinsics.pkl")
    
    checkpoint = f"sam2/checkpoints/sam2.1_hiera_{model_name}.pt"
    model_cfg = "configs/samurai/sam2.1_hiera_b+.yaml" if model_name == "base_plus" else f"configs/samurai/sam2.1_hiera_{model_name[0]}.yaml"
    num_frames, width, height = extract_numframes_width_height(temp_img_folder)
    boxes = load_multi_boxes(txt_path)
    
    manager = Manager()
    shared_masks = manager.dict()
    jobs = []

    with open(intrin_path, 'rb') as f:
        loaded_intrinsics_dict = pickle.load(f)
        
    color_intrin = rs.intrinsics()
    color_intrin.width = loaded_intrinsics_dict['width']
    color_intrin.height = loaded_intrinsics_dict['height']
    color_intrin.ppx = loaded_intrinsics_dict['ppx']
    color_intrin.ppy = loaded_intrinsics_dict['ppy']
    color_intrin.fx = loaded_intrinsics_dict['fx']
    color_intrin.fy = loaded_intrinsics_dict['fy']
    color_intrin.model = rs.distortion(loaded_intrinsics_dict['model'])
    color_intrin.coeffs = loaded_intrinsics_dict['coeffs']

    error_rate = 0.56/100
    lower = 56
    upper = 57.75
    lower_limit = round(lower - lower * error_rate, 2)
    upper_limit = round(upper + upper * error_rate, 2)
    print("lower limit", lower_limit)
    print("upper limit", upper_limit)
    print()
    for obj_id, box in enumerate(boxes):
        p = Process(target=run_tracking, args=(obj_id, box, shared_masks, temp_img_folder, width, height, model_cfg, checkpoint))
        jobs.append(p)
        p.start()

    start = time()
    for p in jobs:
        p.join()
    print("joining took ", time()-start)
    # start = time()
    # Compose video
    # out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (height, width))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor = (0, 0, 0)
    lineType = 2
    dist_list = []
    for frame_idx in range(num_frames):
        img_path = os.path.join(temp_img_folder, f"{frame_idx+1:08d}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            continue
        left_mask = None
        right_mask = None
        for obj_id, masks in shared_masks.items():
            mask = masks[frame_idx]
            mask = denoise_mask(mask)
            color = (0, 255, 0) if obj_id == 0 else (0, 0, 255)
            mask_img = np.zeros_like(img)
            mask_img[mask] = color
            img = cv2.addWeighted(img, 1.0, mask_img, 0.2, 0)
            
            if obj_id==0: left_mask = mask 
            else: right_mask = mask
        if left_mask is None or right_mask is None or (left_mask is not None and not np.any(left_mask)) or (right_mask is not None and not np.any(right_mask)):
            print(f"[WARNING] Frame {frame_idx+1}: Empty mask, skipping.")
            cv2.imwrite(f"output/{frame_idx+1}.jpg", img)
            continue
        
        left_edges_points = get_left_edge_coordinates(right_mask)
        right_edges_points = get_right_edge_coordinates(left_mask)
        right_edge_map = { y: (x, y) for x, y in right_edges_points }
        left_edge_map = { y: (x, y) for x, y in left_edges_points }
        centroid = getCentroid(left_mask, right_mask)
        midpoint_list = neighborsGenerator(centroid)
        horizontal_gauge = False
        total_dist = 0.0
        count = 0.0
        for mid_point in midpoint_list:
            left_point = right_edge_map.get(mid_point[1])
            if left_point is None: continue
            right_point = left_edge_map.get(mid_point[1])
            if right_point is None: continue
            
            dist = getDistance(left_point, right_point, os.path.join(depth_folder, f"{frame_idx+1:08d}.png"), color_intrin)
            total_dist+=dist
            count+=1
            if dist<lower_limit or dist>upper_limit:
                cv2.line(img, left_point, right_point, color=(0, 0, 255), thickness=1)
                horizontal_gauge = True
            else: cv2.line(img, left_point, right_point, color=(255, 0, 0), thickness=1)
            
            cv2.putText(img, '{0:.1f} inches'.format(dist), (0, left_point[1]), font, fontScale, fontColor, 2, lineType)
        if horizontal_gauge: print("Horizontal Gauge in", frame_idx+1)
        # out.write(img)
        cv2.imwrite(f"output/{frame_idx+1}.jpg", img)
        if count!=0:
            dist_list.append(total_dist/count)
        
    # out.release()
    frame = list(range(len(dist_list)))
    plt.figure(figsize=(8, 5))
    plt.plot(frame, dist_list, linestyle='-', color='blue')
    # Add labels and title
    plt.xlabel('Frame')
    plt.ylabel('Gauge (inches)')
    plt.title('Track Gage Measurements Over Frame')
    plt.grid(True)

    # Save to file
    plt.savefig('gage_over_time.png', dpi=300)
    print("average gauge", sum(dist_list)/(len(dist_list)*1.0))
    print("Finished exporting combined video.")
    print("exporting took ", time()-start)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="base_plus")
    args = parser.parse_args()
    main(args.data_path, args.txt_path, model_name=args.model_name)



