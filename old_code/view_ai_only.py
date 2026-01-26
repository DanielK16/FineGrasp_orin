import numpy as np
import open3d as o3d
import os
from PIL import Image
import scipy.io as scio

def get_gripper_points(width, depth, num_points=100):
    """Generiert Punkte für eine U-Form (Greifer)."""
    w = width / 2
    d = depth
    bl = np.array([-d, -w, 0]) 
    br = np.array([-d, w, 0])  
    tl = np.array([0, -w, 0])  
    tr = np.array([0, w, 0])   
    
    points = []
    for t in np.linspace(0, 1, num_points):
        points.append(bl + (br - bl) * t)
    for t in np.linspace(0, 1, num_points):
        points.append(bl + (tl - bl) * t)
    for t in np.linspace(0, 1, num_points):
        points.append(br + (tr - br) * t)
    return np.array(points)

def main():
    print("🎨 Preparing 3D Visualization (AI-Only Normalization Mode)...")
    
    # --- PFADE ---
    BASE_DIR = os.path.expanduser("~/FINE/scripts/scene_data")
    rgb_path = os.path.join(BASE_DIR, "color.png")
    depth_path = os.path.join(BASE_DIR, "depth_ai.png")
    mat_path = os.path.join(BASE_DIR, "mat.mat") # Prüfe ob 0000.mat oder mat.mat
    npy_path = os.path.join(BASE_DIR, "result_poses.npy")

    # --- DATEN LADEN ---
    if not all([os.path.exists(p) for p in [rgb_path, depth_path, mat_path, npy_path]]):
        print("❌ Fehler: Eine der Dateien fehlt in scene_data!")
        return

    rgb = np.array(Image.open(rgb_path))
    depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
    meta = scio.loadmat(mat_path)
    intrinsic = meta["intrinsic_matrix"]
    grasps = np.load(npy_path)

    # --- DER TRICK: MANUELLE NORMALISIERUNG ---
    print("🛠️ Normalisiere KI-Tiefe (0.4m - 1.0m)...")
    mask = depth_raw > 0
    if np.any(mask):
        d_min = depth_raw[mask].min()
        d_max = depth_raw[mask].max()
        # Werte auf 0.0 - 1.0 bringen
        depth_norm = (depth_raw - d_min) / (d_max - d_min)
        # Auf 0.4m bis 1.0m mappen (künstliche Metrik)
        depth_m = 0.4 + (depth_norm * 0.6)
        depth_m[~mask] = 0
        # In uint16 Millimeter für Open3D konvertieren
        depth_final = (depth_m * 1000).astype(np.uint16)
    else:
        depth_final = depth_raw.astype(np.uint16)

    # --- PUNKTWOLKE ERZEUGEN ---
    o3d_rgb = o3d.geometry.Image(rgb)
    o3d_depth = o3d.geometry.Image(depth_final)
    
    h, w, _ = rgb.shape
    fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
    pinhole_cam = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, 
        depth_scale=1000.0, 
        depth_trunc=2.0, 
        convert_rgb_to_intensity=False
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_cam)
    
    # Transformation damit es nicht "kopfüber" ist
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    all_points = [np.asarray(pcd.points)]
    all_colors = [np.asarray(pcd.colors)]
    
    # --- GRASPS HINZUFÜGEN ---
    print(f"   -> Adding {min(100, len(grasps))} grippers...")
    for i in range(min(100, len(grasps))):
        g = grasps[i]
        width, g_depth = g[1], g[3]
        rot = g[4:13].reshape(3, 3)
        trans = g[13:16]
        
        # Achtung: Falls deine Grasps in mm sind, hier durch 1000 teilen!
        # trans = trans / 1000.0 
        
        gripper_pts = get_gripper_points(width, g_depth)
        # Transformation: P_world = R * P_gripper + T
        gripper_pts_world = (rot @ gripper_pts.T).T + trans
        
        # Invertiere Y und Z der Greifer-Punkte passend zur PCD Transformation
        gripper_pts_world[:, 1] *= -1
        gripper_pts_world[:, 2] *= -1

        color = [0, 1, 0] if i == 0 else ([1, 1, 0] if i < 10 else [1, 0, 0])
        all_colors.append(np.tile(color, (len(gripper_pts_world), 1)))
        all_points.append(gripper_pts_world)
        
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_points, axis=0))
    final_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(all_colors, axis=0))

    # --- VIEWER ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="AI-Only Fixed Preview", width=1280, height=720)
    vis.add_geometry(final_pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    vis.get_render_option().point_size = 3.0
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
