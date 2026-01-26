import numpy as np
import cv2
import os
import scipy.io as scio
from PIL import Image

def draw_grasp_on_image(img, grasp, K, color=(0, 255, 0)):
    """
    Draws a grasp (gripper) onto the image.
    grasp format: [score, width, height, depth, R(9), T(3), id]
    """
    # 1. Unpack parameters
    width = grasp[1]
    depth = grasp[3]
    rot = grasp[4:13].reshape(3, 3)
    trans = grasp[13:16]
    
    # 2. Gripper geometry (in gripper coordinate system)
    w = width / 2
    d = depth 
    
    # Corner points: BaseL, BaseR, TipL, TipR, Center
    points_g = np.array([
        [-d, -w, 0], [-d, w, 0], [0, -w, 0], [0, w, 0], [0, 0, 0]
    ])
    
    # 3. Transformation into camera coordinates
    points_cam = (rot @ points_g.T).T + trans
    
    # 4. Projection into 2D image
    points_2d_hom = (K @ points_cam.T).T
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:]
    pts = points_2d.astype(np.int32)
    
    # 5. Drawing (U-shape)
    H, W = img.shape[:2]
    def is_valid(p): return 0 <= p[0] < W and 0 <= p[1] < H

    # Lines: TipL->BaseL->BaseR->TipR
    line_seq = [2, 0, 1, 3]
    thickness = 2
    
    for i in range(len(line_seq) - 1):
        p1 = tuple(pts[line_seq[i]])
        p2 = tuple(pts[line_seq[i+1]])
        if is_valid(p1) and is_valid(p2):
            cv2.line(img, p1, p2, color, thickness)
            
    # Center point
    center_pt = tuple(pts[4])
    if is_valid(center_pt):
        cv2.circle(img, center_pt, 4, (0, 0, 255), -1)

def main():
    print("Starting visualization for OWN data...")
    
    # 1. Define paths
    data_dir = os.path.expanduser("~/FINE/scripts/scene_data")
    rgb_path = os.path.join(data_dir, "color.png")
    mat_path = os.path.join(data_dir, "0000.mat")
    npy_path = "/home/rric/FINE/scripts/scene_data/result_poses.npy" 

    # 2. Checks
    if not os.path.exists(rgb_path):
        print(f"Missing image: {rgb_path}")
        return
    if not os.path.exists(mat_path):
        print(f"Missing intrinsics: {mat_path}")
        return
    if not os.path.exists(npy_path):
        print(f"Missing result: {npy_path} (Run my_custom_infer.py first!)")
        return

    # 3. Load data
    print("   -> Loading image and results...")
    rgb = np.array(Image.open(rgb_path))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
    
    intrinsics = scio.loadmat(mat_path)["intrinsic_matrix"]
    grasps = np.load(npy_path)
    print(f"   -> {len(grasps)} grasps loaded.")
    
    # 4. Drawing (Top K)
    top_k = 200 # Number of grasps to display
    print(f"   -> Drawing Top {top_k} grasps...")
    
    for i in range(min(top_k, len(grasps))):
        # Color based on ranking
        if i == 0:
            color = (0, 255, 0) # Best = Green
            thickness = 3
        elif i < 10:
            color = (0, 255, 255) # Top 10 = Yellow
        else:
            color = (0, 165, 255) # Others = Orange
            
        draw_grasp_on_image(bgr, grasps[i], intrinsics, color)
        
    # 5. Save
    out_file = "my_visualization.png"
    cv2.imwrite(out_file, bgr)
    print(f"\nFINISHED! Image saved as: {out_file}")

if __name__ == "__main__":
    main()
