import cv2
import numpy as np
import os

def ultimate_depth_refinement(color_path, depth_path, output_path):
    print("🚀 Starting Ultimate Depth Refinement (Waves + Spikes + Edges)...")

    # 1. LOAD DATA
    color = cv2.imread(color_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if color is None or depth is None:
        print("❌ Error: Files not found. Check your paths.")
        return

    # Convert to float32 for high-precision processing
    depth_f = depth.astype(np.float32)

    # 2. PRE-CLEANING: REMOVE SPIKES (Median Filter)
    # Spikes are impulsive noise. A median filter is the best weapon here.
    # We do this first so the spikes don't get "smeared" by the guided filter.
    depth_no_spikes = cv2.medianBlur(depth_f, 5)

    # 3. HOLE FILLING (Inpainting)
    # Fill small 'zero' pixels where the RealSense failed to get data.
    mask = (depth_no_spikes == 0).astype(np.uint8)
    depth_filled = cv2.inpaint(depth_no_spikes, mask, 3, cv2.INPAINT_NS)

    # 4. DISPARITY TRANSFORMATION
    # Filtering in Disparity space (1/depth) ensures that edges between
    # the chopstick and the table stay razor-sharp.
    with np.errstate(divide='ignore', invalid='ignore'):
        disparity = 1.0 / depth_filled
        disparity[~np.isfinite(disparity)] = 0

    # 5. RGB-GUIDED FILTERING
    # This uses the high-res Color image to 'guide' the low-res Depth image.
    print("   -> Running Edge-Preserving Guided Filter...")
    guide = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    # Sharpen the guide image to make edges even more prominent
    guide = cv2.equalizeHist(guide)
    
    # radius=10: Smooths waves within the stick
    # eps=0.0001: Keeps the edges sharp
    refined_disparity = cv2.ximgproc.guidedFilter(
        guide=guide, 
        src=disparity, 
        radius=12, 
        eps=1e-6
    )

    # 6. BACK TO DEPTH SPACE
    with np.errstate(divide='ignore', invalid='ignore'):
        refined_depth = 1.0 / refined_disparity
        refined_depth[~np.isfinite(refined_depth)] = 0

    # 7. FINAL SURFACE SMOOTHING (Bilateral)
    # This removes the very last 'jitter' on the stick surface.
    print("   -> Final surface smoothing...")
    # d=9: diameter of pixel neighborhood
    # sigmaColor=75: how much color difference is allowed to smooth
    # sigmaSpace=75: how far pixels can be to influence each other
    final_depth = cv2.bilateralFilter(refined_depth, d=9, sigmaColor=0.05, sigmaSpace=5)

    # 8. POST-PROCESS & SAVE
    # Clip to avoid artifacts and convert back to standard 16-bit PNG
    final_uint16 = np.clip(final_depth, 0, 65535).astype(np.uint16)

    cv2.imwrite(output_path, final_uint16)
    print(f"✅ Precision depth map saved to: {output_path}")

    # Generate a visual comparison for your review
    vis_orig = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
    vis_final = cv2.applyColorMap(cv2.convertScaleAbs(final_uint16, alpha=0.03), cv2.COLORMAP_JET)
    comparison = np.hstack((vis_orig, vis_final))
    cv2.imwrite("refinement_comparison.png", comparison)
    print("   -> See 'refinement_comparison.png' to check the improvement.")

if __name__ == "__main__":
    # Your specific paths
    BASE = os.path.expanduser("~/FINE/scripts/scene_data/")
    ultimate_depth_refinement(
        os.path.join(BASE, "color.png"),
        os.path.join(BASE, "depth.png"),
        os.path.join(BASE, "depth_refined.png")
    )