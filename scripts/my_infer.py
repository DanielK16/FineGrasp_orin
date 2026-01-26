"""
FineGrasp Inference - Standard Version
Nutzt die vorgegebene Struktur ohne externe my_config.json.
"""
import time
import sys
import os
import types
import json
import shutil
import logging
import numpy as np
import open3d as o3d
from PIL import Image
import scipy.io as scio

# ==============================================================================
# 1. Configure Environment:idk what exactly this does
# ==============================================================================
print("Configuring Environment...")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # ~/FINE/scripts
FINE_ROOT = os.path.dirname(SCRIPT_DIR)                 # ~/FINE
SBG_ROOT = os.path.join(FINE_ROOT, "Scale-Balanced-Grasp")
ROBO_ROOT = os.path.join(FINE_ROOT, "robo_orchard_lab")

# System-Pfade für Importe setzen
sys.path.extend([
    os.path.join(SBG_ROOT, "knn"),
    os.path.join(SBG_ROOT, "pointnet2"),
    SBG_ROOT, ROBO_ROOT, FINE_ROOT
])

# C++ Module Patching (Nötig für die Funktion von PointNet/KNN)
try:
    import pointnet2_utils
    sys.modules["pointnet2.pointnet2_utils"] = pointnet2_utils
    import knn_modules 
    mod_ops = types.ModuleType("robo_orchard_lab.ops")
    sys.modules.setdefault("robo_orchard_lab.ops", mod_ops)
    mod_knn = types.ModuleType("robo_orchard_lab.ops.knn")
    mod_knn.knn_modules = knn_modules
    sys.modules["robo_orchard_lab.ops.knn"] = mod_knn
    sys.modules["robo_orchard_lab.ops.knn.knn_modules"] = knn_modules
except ImportError: pass

from robo_orchard_lab.models.finegrasp.processor import GraspInput
from robo_orchard_lab.inference.basic import InferencePipeline 
import robo_orchard_core.utils.config as core_config

# Fix für rekursive Import-Fehler
original_string_to_callable = core_config.string_to_callable
def patched_string_to_callable(target_str):
    if target_str == "robo_orchard_lab.inference.basic:InferencePipeline": return InferencePipeline
    return original_string_to_callable(target_str)
core_config.string_to_callable = patched_string_to_callable

# ==============================================================================
# 2. INFERENCE LOGIC
# ==============================================================================

def run():
    # Paths
    # model checkpoint is inside ~/FINE/scripts/config
    model_dir = os.path.join(SCRIPT_DIR, "configs")
    # scene_data is inside ~/FINE/scene_data
    data_dir = os.path.join(FINE_ROOT, "scripts/scene_data")
    # checkpoint file
    checkpoint_file = os.path.join(model_dir, "model.safetensors")
    # config for inference
    inf_cfg_path = os.path.join(model_dir, "inference.config.json")
    if not os.path.exists(inf_cfg_path):
        shutil.copy(os.path.join(model_dir, "model.config.json"), inf_cfg_path)

    print("\n📁 Folder Structure")
    print(f"🔹 model directory:   {model_dir}")
    print(f"   -> Checkpoint:  model.safetensors (weights)")
    print(f"   -> Config:      inference.config.json (config for inference)")
    print(f"🔹 data directory:    {data_dir}")
    print(f"   -> Images:      color.png & depth.png")
    print(f"   -> Intrinsics:  0000.mat")

    # Load Pipeline
    print("\nLoading modell weights into Storage...")
    pipeline = InferencePipeline.load_pipeline(directory=model_dir, device="cuda")
    pipeline.model.eval()

    # Load Data
    start_time = time.time()
    print(f"\nLoading Data from {data_dir}... ")
    rgb = np.array(Image.open(os.path.join(data_dir, "color.png")), dtype=np.float32)
    depth = np.array(Image.open(os.path.join(data_dir, "depth.png")), dtype=np.float32)
    meta = scio.loadmat(os.path.join(data_dir, "0000.mat"))
    intrinsic = meta["intrinsic_matrix"]
    scale = float(meta["factor_depth"][0][0]) if "factor_depth" in meta else 1000.0

    # GraspInput
    inp = GraspInput(
        rgb_image=rgb, 
        depth_image=depth, 
        depth_scale=scale, 
        intrinsic_matrix=intrinsic, 
        grasp_workspace=[-1, 1, -1, 1, -0.05, 2.0], 
        num_sample_points=150000                     
    )

    # Inference
    print("\nStarting Inference...")
    output = pipeline(inp)
    
    # Save Results
    if hasattr(output, 'grasp_poses') and len(output.grasp_poses) > 0:
        gg = output.grasp_poses
        print(f"\nSuccess: {len(gg)} Grasps found")
        save_path = os.path.join(data_dir, "result_poses.npy")
        grasp_data = gg.grasp_group_array if hasattr(gg, 'grasp_group_array') else gg
        np.save(save_path, grasp_data)
        
        print(f"Grasp poses saved to: {save_path}")
    else:
        print("No grasps found!")

    # Time needed for inference
    print(f"\nTime needed for Inference: {time.time() - start_time} seconds")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()