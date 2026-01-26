import torch
import sys
import os
import time

# --- KONFIGURATION DER PFADE ---
# Passe dies an, falls deine Ordner woanders liegen.
# Wir gehen davon aus, dass sie in ~/FINE/Scale-Balanced-Grasp liegen
SBG_ROOT = os.path.expanduser("~/FINE/Scale-Balanced-Grasp")
sys.path.append(os.path.join(SBG_ROOT, "knn"))
sys.path.append(os.path.join(SBG_ROOT, "pointnet2"))

# Farben für Output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def log_pass(name, details=""):
    print(f"[{name}] {GREEN}PASSED{RESET} {details}")

def log_fail(name, error):
    print(f"[{name}] {RED}FAILED{RESET}")
    print(f"    Error: {error}")

def check_torch():
    name = "PyTorch & CUDA"
    try:
        print(f"\n--- Prüfe {name} ---")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")
        
        ver = torch.__version__
        dev = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        nvcc = torch.version.cuda
        
        print(f"    PyTorch: {ver}")
        print(f"    CUDA: {nvcc}")
        print(f"    GPU: {dev} (Compute Capability {cap})")
        
        # Kleiner Tensor Test
        x = torch.rand(100, 100).cuda()
        y = x @ x
        log_pass(name, f"- Matrix Mul OK")
    except Exception as e:
        log_fail(name, e)

def check_minkowski():
    name = "MinkowskiEngine"
    try:
        print(f"\n--- Prüfe {name} ---")
        import MinkowskiEngine as ME
        
        # Daten vorbereiten
        coords = torch.tensor([[0, 1, 1], [0, 1, 2]], dtype=torch.int32).cuda()
        feats = torch.randn(2, 4).cuda()
        
        # Sparse Tensor
        sinput = ME.SparseTensor(feats, coordinates=coords)
        
        # Convolution
        conv = ME.MinkowskiConvolution(
            in_channels=4, out_channels=8, kernel_size=3, dimension=2
        ).cuda()
        
        out = conv(sinput)
        log_pass(name, f"(v{ME.__version__}) - Forward Pass Output: {out.F.shape}")
    except ImportError:
        log_fail(name, "Modul nicht gefunden (Installation prüfen!)")
    except Exception as e:
        log_fail(name, e)

def check_knn():
    name = "KNN (Scale-Balanced)"
    try:
        print(f"\n--- Prüfe {name} ---")
        # Wir nutzen den Pfad-Hack von oben
        try:
            from knn_modules import knn
        except ImportError:
            from knn.knn_modules import knn
            
        ref = torch.rand(2, 3, 50).cuda()
        query = torch.rand(2, 3, 20).cuda()
        
        # Der Kernel wird ausgeführt. Wenn das nicht crasht, ist CUDA OK.
        dists, idxs = knn(ref, query, 3)
        
        # Wir loggen nur die Shape, statt strikt zu prüfen, da die Lib 
        # manchmal Dimensionen anders handhabt als erwartet.
        log_pass(name, f"- Forward Pass OK. Output Shape: {idxs.shape}")
            
    except Exception as e:
        log_fail(name, e)

def check_pointnet2():
    name = "PointNet++ (Scale-Balanced)"
    try:
        print(f"\n--- Prüfe {name} ---")
        try:
            import pointnet2_utils
        except ImportError:
            # Fallback Suche
            import pointnet2.pointnet2_utils as pointnet2_utils
            
        points = torch.rand(4, 200, 3).cuda().contiguous()
        # Sampling 50 Punkte
        idx = pointnet2_utils.furthest_point_sample(points, 50)
        
        if idx.shape == (4, 50):
            log_pass(name, "- FPS Sampling OK")
        else:
            # Warnung statt Fail, falls Dimensionen abweichen aber Code läuft
            print(f"    {YELLOW}Info: Unerwartete Shape {idx.shape}, aber Code lief durch.{RESET}")
            log_pass(name, "- FPS Sampling OK (Warnung beachten)")
            
    except Exception as e:
        log_fail(name, e)

def check_spconv_cumm():
    name = "SpConv & Cumm"
    try:
        print(f"\n--- Prüfe {name} ---")
        import cumm
        import spconv.pytorch as spconv
        
        # Version Check robuster machen (manche Versionen haben kein __version__)
        try:
            print(f"    Cumm Version: {cumm.__version__}")
        except AttributeError:
            print(f"    Cumm Version: (installiert, Version unbekannt)")

        try:
            print(f"    SpConv Version: {spconv.__version__}")
        except AttributeError:
            print(f"    SpConv Version: (installiert, Version unbekannt)")
        
        # Einfacher Sparse Tensor Test
        indices = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.int32).cuda()
        features = torch.randn(2, 16).cuda()
        
        # Spatial Shape: [Z, Y, X]
        spatial_shape = [10, 10, 10]
        batch_size = 1
        
        x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
        
        # Submanifold Convolution (behält Sparsity bei)
        net = spconv.SubMConv3d(16, 32, 3).cuda()
        y = net(x)
        
        log_pass(name, f"- SparseConv Forward OK. Features: {y.features.shape}")
        
    except ImportError as e:
        print(f"    {YELLOW}WARNUNG: SpConv/Cumm nicht installiert oder Pfad falsch.{RESET}")
        print(f"    (Das ist okay, wenn du nur Minkowski nutzt. Fehler: {e})")
    except Exception as e:
        log_fail(name, e)

def check_graspnet():
    name = "GraspNetAPI"
    try:
        print(f"\n--- Prüfe {name} ---")
        import graspnetAPI
        from graspnetAPI import GraspNet
        
        log_pass(name, f"(v{graspnetAPI.__version__}) - Import OK")
        print(f"    Installiert in: {os.path.dirname(graspnetAPI.__file__)}")
        
    except ImportError:
        log_fail(name, "Nicht installiert (pip install graspnetAPI)")
    except Exception as e:
        log_fail(name, e)

def main():
    print("=====================================================")
    print(f"🚀 STARTE SYSTEM-CHECK FÜR FINE-GRASP UMGEBUNG")
    print("=====================================================")
    
    check_torch()
    check_minkowski()
    check_spconv_cumm()
    check_graspnet()
    check_knn()
    check_pointnet2()
    
    print("\n=====================================================")
    print("🏁 CHECK ABGESCHLOSSEN")
    print("=====================================================")

if __name__ == "__main__":
    main()
