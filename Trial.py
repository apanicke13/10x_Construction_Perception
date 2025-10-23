1#!/usr/bin/env python3

import os
import sys
import math
import numpy as np
import cv2
from scipy.spatial import ConvexHull
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# BAG EXTRACTION

def extract_depth_frames(bag_file, output_dir, topic="/depth"):
    
    # Checking file and making output folder
    if not os.path.exists(bag_file):
        raise FileNotFoundError(f"Cannot find bag file: {bag_file}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Extract] Reading {bag_file}")

    # Open the rosbag
    # rosbag2_py is used to iterate through messages
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_file, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )

    bridge = CvBridge()
    count = 0

    # Converts messages to images
    # Converting serialized ROS 2 data into a NumPy array using cv_bridge
    while reader.has_next():
        topic_name, data, _ = reader.read_next()
        if topic_name != topic:
            continue
        try:
            msg = deserialize_message(data, Image)
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            print("  >> skipped frame:", e)
            continue

        # Normalizing depth units
        arr = img.astype(np.float32)
        if arr.dtype == np.uint16 or arr.max() > 1000:
            arr /= 1000.0  # convert mm → m

        # Saving frames
        np.save(os.path.join(output_dir, f"frame_{count}.npy"), arr)
        count += 1
    print(f"[Extract] Done. {count} frames written to {output_dir}")
    return output_dir

# PLANE FITTING AND ANALYSIS

def depth_to_xyz(depth, fx, fy, cx, cy, step=2, max_z=5.0):

    # Getting image dimensions
    h, w = depth.shape

    # Generating pixel coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Filtering out invalid and unwanted depth values
    mask = np.isfinite(depth) & (depth > 0.01) & (depth < max_z)

    # Downsampling the depth map to speed up computation
    if step > 1:
        mask &= ((u % step == 0) & (v % step == 0))
    
    # Early exit if no valid points remain
    if not mask.any():
        return np.zeros((0, 3))
    
    # Extracts depth value for valid pixels
    z = depth[mask]

    # Using pinhole camera equations to back-project from 2D pixel coordinates to 3D camera coordinates
    x = (u[mask] - cx) * z / fx
    y = (v[mask] - cy) * z / fy
    return np.column_stack((x, y, z))



def ransac_plane(points, trials=500, eps=0.02, min_pts=80):

    # Checking if enough points are available
    if len(points) < 3:
        return None, None

    # Intializing best model trackers
    best_normal, best_inliers = None, []
    n = len(points)

    # Running the RANSAC loop
    for _ in range(trials):
        # Randomly sampling 3 distinct points
        ids = np.random.choice(n, 3, replace=False)
        p0, p1, p2 = points[ids]

        # Compute the plane's normal vector
        normal = np.cross(p1 - p0, p2 - p0)

        # Skipping collinear points
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue

        # Normalizing the normal vector
        normal /= norm

        # Computing the plane's offset term d using point-normal plane equation
        d = -np.dot(normal, p0)

        # Computing point to plane distances
        dist = np.abs(points @ normal + d)

        # Finding all inliers for this plane
        inliers = points[dist < eps]

        # Checking if the plane is the best so far
        if len(inliers) > len(best_inliers):
            best_inliers, best_normal = inliers, normal

    # Validating the final model
    if len(best_inliers) < min_pts:
        return None, None

    # Refining the plane using all inliers
    centered = best_inliers - best_inliers.mean(axis=0)
    cov = centered.T @ centered
    _, _, vt = np.linalg.svd(cov)
    refined_normal = vt[-1] / np.linalg.norm(vt[-1])
    return refined_normal, best_inliers



def compute_area_2d(pts3, n):

    # Normalizing the normal vector
    n = n / np.linalg.norm(n)

    # Picking a reference vector not normal to the plane normal
    ref = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
    
    # Computing the tangent vectors u and v
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    v = np.cross(n, u)

    # Projecting the 3D points onto the 2D plane basis
    proj = np.dot(pts3 - pts3.mean(0), np.vstack([u, v]).T)

    proj_real = proj[np.linalg.norm(proj, axis=1) < ((np.median(np.linalg.norm(proj, axis=1))) * 2.0)]
    if proj_real.shape[0] < 3:
        return 0.0
    
    return ConvexHull(proj_real).volume



def process_frames(frames_dir):

    # Loading and checking the frame files
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".npy"))
    if not files:
        raise RuntimeError("No frames found for analysis.")
    
    # Loading one sample frame to get the camera parameters
    depth0 = np.load(os.path.join(frames_dir, files[0]))
    H, W = depth0.shape
    fx = fy = 525.0
    cx, cy = (W - 1) / 2, (H - 1) / 2

    # Preparing output container
    csv_lines = ["filename,angle_deg,area_m2\n"]
    normals = []

    # Looping through all depth frames
    for i, name in enumerate(files):

        # Loading depth image and converting to 3D points
        depth = np.load(os.path.join(frames_dir, name)).astype(np.float32)
        pts = depth_to_xyz(depth, fx, fy, cx, cy)

        # Skipping frames with too few valid points
        if pts.shape[0] < 50:
            print(f"[{i}] {name}: insufficient points")
            continue

        # Fitting a plane using RANSAC
        n, inliers = ransac_plane(pts)
        if n is None:
            print(f"[{i}] {name}: no plane")
            continue

        # Ensuring the normal points towards the camera
        if np.dot(n, [0, 0, 1]) < 0:
            n = -n

        # Computing the plane's surface area and orientation angle
        area = compute_area_2d(inliers, n)
        angle = math.degrees(math.acos(np.clip(abs(np.dot(n, [0, 0, 1])), -1, 1)))
        
        # Logging results for this frame
        csv_lines.append(f"{name},{angle:.2f},{area:.2f}\n")
        normals.append(n)

        # Printing progress information
        print(f"[{i}] {name}: angle={angle:.2f}°, area={area:.2f}")

    # Saving results as CSV
    with open("results.csv", "w") as f:
        f.writelines(csv_lines)
    print("Results saved → results.csv")

    # Computing the overall rotation axis
    if len(normals) >= 3:
        M = np.vstack(normals) - np.mean(normals, axis=0)
        _, _, vt = np.linalg.svd(M)
        axis = vt[-1] / np.linalg.norm(vt[-1])

        # Saving results as txt
        np.savetxt("axis_of_rotation.txt", axis[None], fmt="%.4f")
        print("Rotation axis saved → axis_of_rotation.txt")
    return 

# MAIN

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 depth_pipeline.py <bag_file.db3> <depth_topic>")
        sys.exit(0)

    # Extracting data
    bag_path = sys.argv[1]
    topic = sys.argv[2]

    # Defining output directory for extracted frames
    frames_dir = "./frames"
    extract_depth_frames(bag_path, frames_dir, topic)
    process_frames(frames_dir)



if __name__ == "__main__":
    main()