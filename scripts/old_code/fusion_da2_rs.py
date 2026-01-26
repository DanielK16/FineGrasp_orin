import cv2
import numpy as np
import os
import scipy.io as scio

def generate_ply_manually(color_path, depth_path, mat_path, ply_path):
    print("☁️ Berechne Punktwolke mit Numpy (Pure Math Mode)...")
    
    # 1. Daten laden
    color = cv2.imread(color_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mat = scio.loadmat(mat_path)
    intrinsics = mat['intrinsic_matrix']
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # 2. Pixel-Gitter erstellen
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    
    # 3. Maske für gültige Daten (Tiefe > 0)
    mask = depth > 0
    z = depth[mask] / 1000.0  # Konvertierung in Meter
    u = u[mask]
    v = v[mask]
    
    # 4. Rückprojektion in 3D (Lochkamera-Modell)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Farben extrahieren (BGR zu RGB)
    colors = color[mask][:, [2, 1, 0]]
    
    # 5. Punktwolke zusammenfügen
    points = np.stack((x, y, z), axis=-1)
    
    # 6. PLY-Datei schreiben (Header + Daten)
    print(f"💾 Schreibe Daten in {ply_path}...")
    num_points = len(points)
    
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(num_points):
            f.write(f"{points[i,0]:.4f} {points[i,1]:.4f} {points[i,2]:.4f} "
                    f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")

    print(f"✅ PLY-Datei erfolgreich erstellt: {ply_path}")

if __name__ == "__main__":
    BASE = os.path.expanduser("~/FINE/scripts/scene_data")
    COLOR = os.path.join(BASE, "color.png")
    DEPTH = os.path.join(BASE, "depth_fused.png")
    MAT = os.path.join(BASE, "0000.mat")
    OUT_PLY = os.path.join(BASE, "fused_cloud.ply")
    
    if os.path.exists(DEPTH):
        generate_ply_manually(COLOR, DEPTH, MAT, OUT_PLY)
    else:
        print("❌ depth_fused.png nicht gefunden!")