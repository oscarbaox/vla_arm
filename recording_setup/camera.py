import re
import subprocess
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

TARGET_LIFECAM_NAME = "Microsoft® LifeCam Cinema(TM)"
TARGET_USBCAM_NAME = "Intel(R) RealSense(TM)"


def list_v4l2_devices():
    """
    Run 'v4l2-ctl --list-devices' and parse the output.
    """
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Warning: could not list V4L2 devices: {e}")
        return {}

    devices = {}
    current_name = None

    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        if line[0] in (" ", "\t"):
            if current_name is not None:
                devices[current_name].append(line.strip())
        else:
            name = re.sub(r"\s*\([^()]*\):\s*$", "", line).strip()
            current_name = name
            devices[current_name] = []

    return devices


def find_camera_index(devices, target_name):
    """
    Find the lowest-numbered /dev/videoN node for a device name.
    """
    if target_name in devices:
        nodes = devices[target_name]
    else:
        matches = [
            name
            for name in devices
            if name.startswith(target_name) or target_name.startswith(name)
        ]
        if not matches:
            return None
        nodes = devices[matches[0]]

    video_indices = [
        int(re.match(r"/dev/video(\d+)$", node).group(1))
        for node in nodes
        if re.match(r"/dev/video(\d+)$", node)
    ]
    return min(video_indices) if video_indices else None


def list_realsense_serials():
    """Return serial numbers of all connected RealSense devices, performing a hardware reset first."""
    if rs is None:
        print("Warning: pyrealsense2 not installed; skipping RealSense.")
        return []
    try:
        ctx = rs.context()
        devices = ctx.query_devices()

        # 1. HARDWARE RESET STATE RESET
        # Force a programmatic hardware reset to unstick any crashed pipelines from previous runs
        if len(devices) > 0:
            print(
                "Performing hardware reset on RealSense devices to clear stuck states..."
            )
            for dev in devices:
                dev.hardware_reset()
            # Give the Linux kernel time to re-enumerate the USB devices
            time.sleep(2.5)

            # Re-query the context after the reset re-enumeration
            ctx = rs.context()
            devices = ctx.query_devices()

        return [dev.get_info(rs.camera_info.serial_number) for dev in devices]
    except RuntimeError as e:
        print(f"Warning: RealSense query/reset failed: {e}")
        return []


def start_realsense(serials, width=640, height=480, fps=30):
    """Start one color-stream pipeline per serial. Returns a list of pipelines."""
    pipelines = []
    for serial in serials:
        try:
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 15)
            pipeline = rs.pipeline()

            # Start the pipeline
            profile = pipeline.start(config)

            # 2. DISABLE AUTO-EXPOSURE PRIORITY
            # Prevents the firmware from stalling or dropping frames down to 0 in low light
            dev = profile.get_device()
            color_sensor = dev.first_color_sensor()
            if color_sensor.supports(rs.option.auto_exposure_priority):
                color_sensor.set_option(rs.option.auto_exposure_priority, 0)

            pipelines.append((serial, pipeline))
        except RuntimeError as e:
            print(f"Warning: could not start RealSense {serial}: {e}")
    return pipelines


def start_lifecam():
    """Open the LifeCam via V4L2. Returns a VideoCapture or None."""
    index = find_camera_index(list_v4l2_devices(), TARGET_LIFECAM_NAME)
    if index is None:
        print("LifeCam not found.")
        return None
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Warning: LifeCam found at /dev/video{index} but could not be opened.")
        cap.release()
        return None
    print(f"LifeCam started at /dev/video{index}.")
    return cap


def launch_cameras(width=640, height=480, fps=30):
    """Show LifeCam and/or RealSense, whichever are available."""
    # 3. PROBING SEPARATION
    # Initialize the LifeCam first, then rest the thread briefly before firing up the RealSense SDK.
    lifecam = start_lifecam()
    time.sleep(0.5)

    realsense_serials = list_realsense_serials()
    realsense = start_realsense(realsense_serials, width, height, fps)

    if realsense:
        print(f"RealSense started: {[s for s, _ in realsense]}")
    else:
        print("RealSense not found.")

    if lifecam is None and not realsense:
        print("Error: no cameras available.")
        return

    print("Controls: Press 's' to save snapshots, 'q' to quit.")
    img_counter = 0

    try:
        while True:
            frames = {}
            if lifecam is not None:
                ok, frame = lifecam.read()
                if ok:
                    # crop frame to specific height
                    crop_height = 480
                    frame = frame[:crop_height, :]
                    frames["LifeCam"] = frame
                else:
                    print("Warning: failed to grab LifeCam frame.")

            for serial, pipeline in realsense:
                try:
                    # Timeout set to 5000ms
                    frames_obj = pipeline.wait_for_frames(5000)
                    color = frames_obj.get_color_frame()
                except RuntimeError as e:
                    print(f"Warning: RealSense {serial} frame error: {e}")
                    continue
                if color:
                    frames[f"RealSense {serial}"] = np.asanyarray(color.get_data())

            for name, frame in frames.items():
                cv2.imshow(name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Closing cameras...")
                break
            elif key == ord("s"):
                for name, frame in frames.items():
                    img_name = f"{name.replace(' ', '_')}_snapshot_{img_counter}.png"
                    cv2.imwrite(img_name, frame)
                    print(f"Snapshot saved as {img_name}")
                img_counter += 1
    finally:
        print("\nCleaning up resources cleanly...")
        if lifecam is not None:
            lifecam.release()
        for _, pipeline in realsense:
            try:
                pipeline.stop()
            except RuntimeError:
                pass  # Already stopped or crashed
        cv2.destroyAllWindows()


if __name__ == "__main__":
    launch_cameras()
