import cv2
import re
import subprocess

TARGET_LIFECAM_NAME = "Microsoft® LifeCam Cinema(TM)"
TARGET_USBCAM_NAME = "USB Camera: USB Camera"


def list_v4l2_devices():
    """
    Run 'v4l2-ctl --list-devices' and parse the output.

    Returns a dict: {device_name: [list_of_device_nodes]}
    e.g. {"USB Camera: USB Camera": ["/dev/video4", "/dev/video5", "/dev/media2"], ...}
    """
    result = subprocess.run(
        ["v4l2-ctl", "--list-devices"],
        capture_output=True,
        text=True,
        check=True,
    )

    devices = {}
    current_name = None

    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        if line[0] in (" ", "\t"):
            # Indented line -> device node belonging to the current device
            if current_name is not None:
                devices[current_name].append(line.strip())
        else:
            # Device header line, e.g.:
            # "USB Camera: USB Camera (usb-0000:00:14.0-3):"
            # Strip the trailing "(bus-info):" part to get the pure name
            name = re.sub(r"\s*\([^()]*\):\s*$", "", line).strip()
            current_name = name
            devices[current_name] = []

    return devices


def find_camera_index(devices, target_name):
    """
    Find the lowest-numbered /dev/videoN node for a device name.

    Uses exact match first, then falls back to startswith matching
    (useful when the driver truncates names, e.g. 'Integrated_Webcam_FHD: Integrat').

    Returns the integer index (e.g. 6 for /dev/video6), or None if not found.
    """
    # Exact match
    if target_name in devices:
        nodes = devices[target_name]
    else:
        # Fallback: prefix match
        matches = [
            name
            for name in devices
            if name.startswith(target_name) or target_name.startswith(name)
        ]
        if not matches:
            return None
        nodes = devices[matches[0]]

    # Filter to /dev/videoN nodes and return the lowest index
    video_indices = [
        int(re.match(r"/dev/video(\d+)$", node).group(1))
        for node in nodes
        if re.match(r"/dev/video(\d+)$", node)
    ]
    return min(video_indices) if video_indices else None


def launch_webcam(camera_index=[4]):
    # Pass index 4 directly to match your /dev/video4 configuration
    cap = []
    for index in camera_index:
        cap.append(cv2.VideoCapture(index))

    for i in range(len(cap)):
        if not cap[i].isOpened():
            print(
                f"Error: Could not open your Microdia Webcam at index {camera_index[i]}."
            )
            return

    print("Microdia Webcam Vitade AF successfully started!")
    print("Controls: Press 's' to save a snapshot, 'q' to quit.")

    img_counter = 0

    while True:
        ret = []
        frame = []
        for c in cap:
            ret_, frame_ = c.read()
            ret.append(ret_)
            frame.append(frame_)
        if not all(ret):
            print("Error: Failed to grab frame.")
            break

        for i in range(len(cap)):
            cv2.imshow(f"Microdia Webcam Vitade AF ({i})", frame[i])

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Closing webcam...")
            break
        elif key == ord("s"):
            img_name = f"webcam_snapshot_{img_counter}.png"
            cv2.imwrite(img_name, frame[0])  # Save the first camera's frame
            print(f"Snapshot saved successfully as {img_name}!")
            img_counter += 1

    for c in cap:
        c.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    devices = list_v4l2_devices()
    camera_index = [
        find_camera_index(devices, TARGET_LIFECAM_NAME),
        find_camera_index(devices, TARGET_USBCAM_NAME),
    ]
    launch_webcam(camera_index=[index for index in camera_index if index is not None])
