# my_custom_viz.py - Visualisierung für DEINE EIGENEN Daten
import numpy as np
import cv2
import os
import scipy.io as scio
from PIL import Image

def draw_grasp_on_image(img, grasp, K, color=(0, 255, 0)):
    """
    Zeichnet einen Grasp (Greifer) in das Bild.
    grasp-Format: [score, width, height, depth, R(9), T(3), id]
    """
    # 1. Parameter entpacken
    width = grasp[1]
    depth = grasp[3]
    rot = grasp[4:13].reshape(3, 3)
    trans = grasp[13:16]
    
    # 2. Greifer-Geometrie (im Greifer-System)
    w = width / 2
    d = depth 
    
    # Eckpunkte: BasisL, BasisR, SpitzeL, SpitzeR, Center
    points_g = np.array([
        [-d, -w, 0], [-d, w, 0], [0, -w, 0], [0, w, 0], [0, 0, 0]
    ])
    
    # 3. Transformation in Kamera-Koordinaten
    points_cam = (rot @ points_g.T).T + trans
    
    # 4. Projektion ins 2D Bild
    points_2d_hom = (K @ points_cam.T).T
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:]
    pts = points_2d.astype(np.int32)
    
    # 5. Zeichnen (U-Form)
    H, W = img.shape[:2]
    def is_valid(p): return 0 <= p[0] < W and 0 <= p[1] < H

    # Linien: SpitzeL->BasisL->BasisR->SpitzeR
    line_seq = [2, 0, 1, 3]
    thickness = 2
    
    for i in range(len(line_seq) - 1):
        p1 = tuple(pts[line_seq[i]])
        p2 = tuple(pts[line_seq[i+1]])
        if is_valid(p1) and is_valid(p2):
            cv2.line(img, p1, p2, color, thickness)
            
    # Center Punkt
    center_pt = tuple(pts[4])
    if is_valid(center_pt):
        cv2.circle(img, center_pt, 4, (0, 0, 255), -1)

def main():
    print("🎨 Starte Visualisierung für EIGENE Daten...")
    
    # 1. Pfade definieren
    data_dir = os.path.expanduser("~/FINE/scripts/scene_data")
    rgb_path = os.path.join(data_dir, "color.png")
    mat_path = os.path.join(data_dir, "0000.mat")
    npy_path = "/home/rric/FINE/scripts/scene_data/result_poses.npy" 

    # 2. Checks
    if not os.path.exists(rgb_path):
        print(f"❌ Bild fehlt: {rgb_path}")
        return
    if not os.path.exists(mat_path):
        print(f"❌ Intrinsik fehlt: {mat_path}")
        return
    if not os.path.exists(npy_path):
        print(f"❌ Ergebnis fehlt: {npy_path} (Erst my_custom_infer.py ausführen!)")
        return

    # 3. Daten laden
    print(f"   -> Lade Bild und Ergebnisse...")
    rgb = np.array(Image.open(rgb_path))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # OpenCV mag BGR
    
    intrinsics = scio.loadmat(mat_path)["intrinsic_matrix"]
    grasps = np.load(npy_path)
    print(f"   -> {len(grasps)} Grasps geladen.")
    
    # 4. Zeichnen (Top K)
    top_k = 200 # Anzahl der anzuzeigenden Griffe
    print(f"   -> Zeichne Top {top_k} Grasps...")
    
    for i in range(min(top_k, len(grasps))):
        # Farbe basierend auf Ranking
        if i == 0:
            color = (0, 255, 0) # Bester = Grün
            thickness = 3
        elif i < 10:
            color = (0, 255, 255) # Top 10 = Gelb
        else:
            color = (0, 165, 255) # Rest = Orange
            
        draw_grasp_on_image(bgr, grasps[i], intrinsics, color)
        
    # 5. Speichern
    out_file = "my_visualization.png"
    cv2.imwrite(out_file, bgr)
    print(f"\n✅ FERTIG! Bild gespeichert als: {out_file}")

if __name__ == "__main__":
    main()
