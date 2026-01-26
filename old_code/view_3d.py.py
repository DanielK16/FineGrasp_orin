# view_results.py - Starte dies auf deinem PC
import open3d as o3d
import numpy as np
import os

def main():
    filename = "viz_3d_result.ply"
    
    if not os.path.exists(filename):
        print(f"❌ Datei '{filename}' nicht gefunden!")
        print("   Bitte kopiere sie vom Jetson in diesen Ordner.")
        return

    print(f"⏳ Lade {filename}...")
    pcd = o3d.io.read_point_cloud(filename)

    if pcd.is_empty():
        print("⚠️ Punktwolke ist leer oder konnte nicht gelesen werden.")
        return

    print("🚀 Starte Viewer...")
    print("   [Maus Links]   Drehen")
    print("   [Maus Rad]     Zoomen")
    print("   [Maus Rechts]  Verschieben")
    print("   [+/-]          Punkte größer/kleiner")
    
    # Kleines Koordinatensystem am Ursprung (0,0,0) zur Orientierung
    # Rot=X, Grün=Y, Blau=Z
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # Visualisierungs-Optionen setzen
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="FineGrasp Ergebnisse", width=1280, height=720)
    
    vis.add_geometry(pcd)
    vis.add_geometry(axes)
    
    # Optionen für besseres Aussehen (schwarzer Hintergrund, dickere Punkte)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1]) # Dunkelgrau
    opt.point_size = 3.0 # Punkte etwas dicker machen
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
