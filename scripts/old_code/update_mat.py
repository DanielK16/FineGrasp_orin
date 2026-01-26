import numpy as np
import scipy.io as scio
import os

def update_mat_file():
    # ==========================================
    # 1. DEINE WERTE (Hier anpassen!)
    # ==========================================
    
    # Beispiel-Werte (bitte ändern):
    # fx = Brennweite X, fy = Brennweite Y
    # cx = Bildmitte X, cy = Bildmitte Y
    my_intrinsic = np.array([
        [909.72,   0.0, 637.532],  # fx,  0, cx
        [  0.0, 909.482, 359.21],  #  0, fy, cy
        [  0.0,   0.0,   1.0]   #  0,  0,  1
    ], dtype=np.float64)

    # Umrechnung: Wie viel ist 1 Meter in deinem Tiefenbild?
    # Bei Intel RealSense / Kinect ist es meistens 1000.0 (Werte sind mm)
    my_depth_scale = 1000.0 

    # ==========================================
    # 2. DATEI LADEN & BEARBEITEN
    # ==========================================
    
    # Pfad zur Original-Datei (als Vorlage)
    # Falls du die 0000.mat nicht mehr hast, erstellen wir einfach eine neue Struktur.
    original_path = "0000.mat"
    target_path = os.path.expanduser("~/FINE/scene_data/mat.mat")
    
    data = {}
    
    # Versuch, Original zu laden (um poses/cls_indexes zu behalten, falls vorhanden)
    if os.path.exists(original_path):
        print(f"Lade Vorlage: {original_path}")
        data = dict(scio.loadmat(original_path))
    else:
        print("Keine Vorlage gefunden, erstelle neue Datei.")

    # Werte überschreiben
    print("Setze neue Intrinsik...")
    data["intrinsic_matrix"] = my_intrinsic
    
    print(f"Setze Depth Scale auf {my_depth_scale}...")
    data["factor_depth"] = np.array([[my_depth_scale]], dtype=np.float64)

    # Überflüssige Meta-Daten von scipy entfernen (optional, macht es sauberer)
    for key in ['__header__', '__version__', '__globals__']:
        if key in data:
            del data[key]

    # ==========================================
    # 3. SPEICHERN
    # ==========================================
    
    # Sicherstellen, dass der Ordner existiert
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    scio.savemat(target_path, data)
    print(f"✅ Erfolgreich gespeichert: {target_path}")
    print("Inhalt der neuen Datei:")
    print(data["intrinsic_matrix"])

if __name__ == "__main__":
    update_mat_file()
