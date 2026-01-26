"""
3D Visualization for FineGrasp Results
Description:
    1. Loads RGB, Depth, and Camera Intrinsics.
    2. Projects the 2D scene into a 3D Point Cloud.
    3. Overlays the detected 6-DoF grasp poses as 'U'-shaped grippers.
    4. Opens an interactive Open3D viewer window.
"""

import numpy as np
import open3d as o3d
import os
from PIL import Image
import scipy.io as scio

def get_gripper_points(width, depth, num_points=100):
    """Generates points forming a 'U' shape representing the gripper."""
    w = width / 2
    d = depth
    
    # Vertices in gripper frame: [x, y, z]
    # x=Approach (Depth), y=Width
    bl = np.array([-d, -w, 0]) # Base Left
    br = np.array([-d, w, 0])  # Base Right
    tl = np.array([0, -w, 0])  # Tip Left
    tr = np.array([0, w, 0])   # Tip Right
    
    points = []
    # Line 1: Gripper Base (Left -> Right)
    for t in np.linspace(0, 1, num_points):
        points.append(bl + (br - bl) * t)
    # Line 2: Left Finger
    for t in np.linspace(0, 1, num_points):
        points.append(bl + (tl - bl) * t)
    # Line 3: Right Finger
    for t in np.linspace(0, 1, num_points):
        points.append(br + (tr - br) * t)
        
    return np.array(points)

def main():
    print("🎨 Preparing 3D Visualization...")
    
    # --- PATHS (Adjusted to your structure) ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "scripts/scene_data")
    
    rgb_path = os.path.join(DATA_DIR, "color.png")
    depth_path = os.path.join(DATA_DIR, "depth.png")
    mat_path = os.path.join(DATA_DIR, "0000.mat")
    npy_path = os.path.join(DATA_DIR, "result_poses.npy")

    # --- CHECK DATA ---
    for p in [rgb_path, depth_path, mat_path, npy_path]:
        if not os.path.exists(p):
            print(f"❌ Missing file: {p}")
            return

    # --- LOAD DATA ---
    print("⏳ Loading images and grasp results...")
    rgb = np.array(Image.open(rgb_path))
    depth = np.array(Image.open(depth_path)).astype(np.float32)
    meta = scio.loadmat(mat_path)
    intrinsic = meta["intrinsic_matrix"]
    
    depth_scale = float(meta["factor_depth"][0][0]) if "factor_depth" in meta else 1000.0
    grasps = np.load(npy_path)
    print(f"   -> Found {len(grasps)} grasp candidates.")

    # --- GENERATE SCENE CLOUD ---
    print("   -> Processing Point Cloud...")
    o3d_rgb = o3d.geometry.Image(rgb)
    o3d_depth = o3d.geometry.Image(depth)
    
    h, w, _ = rgb.shape
    fx, fy = intrinsic[0,0], intrinsic[1,1]
    cx, cy = intrinsic[0,2], intrinsic[1,2]
    
    pinhole_cam = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, 
        depth_scale=depth_scale, 
        depth_trunc=2.0, 
        convert_rgb_to_intensity=False
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_cam)
    
    # Collect points/colors for concatenation
    all_points = [np.asarray(pcd.points)]
    all_colors = [np.asarray(pcd.colors)]
    
    # --- ADD GRASPS ---
    print("   -> Adding gripper visualizations...")
    # Show top grasps to avoid cluttering (limit to 100)
    display_limit = min(500, len(grasps))
    
    for i in range(display_limit):
        g = grasps[i]
        # Format: [score, width, height, depth, R(9), T(3), id]
        width, g_depth = g[1], g[3]
        rot = g[4:13].reshape(3, 3)
        trans = g[13:16]
        
        gripper_pts = get_gripper_points(width, g_depth)
        
        # Transform to camera/world frame
        # P_world = R * P_gripper + T
        gripper_pts_world = (rot @ gripper_pts.T).T + trans
        
        # Color Coding: Green (Best) -> Yellow (Top 10) -> Red (Others)
        if i == 0:
            color = [0, 1, 0] # Best
        elif i < 10:
            color = [1, 1, 0] # High score
        else:
            color = [1, 0, 0] # Low score
            
        gripper_colors = np.tile(color, (len(gripper_pts_world), 1))
        
        all_points.append(gripper_pts_world)
        all_colors.append(gripper_colors)
        
    # --- COMBINE & VISUALIZE ---
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_points, axis=0))
    final_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(all_colors, axis=0))
    
    # Coordinate system frame for orientation (Red=X, Green=Y, Blue=Z)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    print("\n🚀 Starting Interactive Viewer...")
    print("   [Left Click]   Rotate")
    print("   [Right Click]  Pan")
    print("   [Scroll Wheel] Zoom")
    print("   [+/- Keys]     Adjust point size")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="FineGrasp Visualization", width=1280, height=720)
    
    vis.add_geometry(final_pcd)
    vis.add_geometry(axes)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.05, 0.05, 0.05]) # Almost black
    opt.point_size = 3.0 
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
