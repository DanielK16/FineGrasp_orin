import pyrealsense2 as rs
import numpy as np
import cv2
import os
import scipy.io as scio
import time

def capture_raw_safe(output_dir):
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

    print("Starting Pipeline...")
    try:

        profile = pipeline.start(config)
    except Exception as e:
        print(f"Pipeline failed to start: {e}")
        print("TIP: Unplug and replug the camera, or try a different USB port.")
        return

    print("Warming up...")
    time.sleep(3)

    try:
        print("Attempting to capture one frame...")
        frames = pipeline.wait_for_frames(timeout_ms=20000)
        
        # Align depth to color
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("Frames received but data is empty.")
            return

        # Data conversion
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Save
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "color.png"), color_image)
        cv2.imwrite(os.path.join(output_dir, "depth.png"), depth_image)
        
        # Save Intrinsics
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        scio.savemat(os.path.join(output_dir, "mat.mat"), {
            "intrinsic_matrix": np.array([[intr.fx, 0, intr.ppx],[0, intr.fy, intr.ppy],[0, 0, 1]]),
            "factor_depth": np.array([[1000.0]])
        })
        
        print(f"Success! Captured raw 720p frame to {output_dir}")

    except Exception as e:
        print(f"TIMEOUT: {e}")
        print("Check if the camera is overheating or if the USB cable is loose.")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    OUT = os.path.expanduser("~/FINE/scripts/scene_data")
    capture_raw_safe(OUT)
