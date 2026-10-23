import torch
import cv2
import numpy as np
import re
import argparse
import json
import socket
import subprocess
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

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


def preprocess_image(frame, target_size=(224, 224)):
    """Resizes, converts to RGB, and normalizes image frames for PyTorch."""
    # Resize to the resolution your SmolVLA vision encoder expects
    resized = cv2.resize(frame, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Convert to (C, H, W) tensor and normalize to [0, 1]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor


def parse_message(raw):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    angles = data.get("angles")
    speed = data.get("speed")
    if not isinstance(angles, list):
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in angles):
        return None
    if not isinstance(speed, (int, float)) or isinstance(speed, bool):
        return None
    return {"angles": [float(v) for v in angles], "speed": float(speed)}


def main():
    parser = argparse.ArgumentParser(description="Run SmolVLA against the arm server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    # 1. Load the SmolVLA policy configuration and weights
    # Replace with your local checkpoint path or Hugging Face repo ID
    policy_path = "lerobot/smolvla_base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading SmolVLA policy from {policy_path} onto {device}...")

    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.to(device)

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path="lerobot/smolvla_base",  # or your fine-tuned checkpoint
    )

    policy.eval()

    # 2. Define your task instruction
    # SmolVLA parses text embeddings to condition its motor sequences
    task_instruction = "pick up the wooden block and place it in the red box"

    # 3. Setup your video streams (e.g., Laptop Cam + External USB Cam)
    devices = list_v4l2_devices()
    lifecam_index = find_camera_index(devices, TARGET_LIFECAM_NAME)  # camera 1
    usbcam_index = find_camera_index(devices, TARGET_USBCAM_NAME)  # camera 2
    cap_top = cv2.VideoCapture(lifecam_index)
    cap_wrist = cv2.VideoCapture(usbcam_index)

    try:
        with socket.create_connection((args.host, args.port)) as sock:
            print(f"Connected to {args.host}:{args.port}")
            print("Starting deployment loop. Press Ctrl+C to stop.")
            with sock.makefile("r") as stream, torch.no_grad():
                for line in stream:
                    raw = line.strip()
                    if not raw:
                        continue
                    state = parse_message(raw)
                    if state is None:
                        print(f"Wrong format: {raw!r}")
                        continue

                    ret_top, frame_top = cap_top.read()
                    ret_wrist, frame_wrist = cap_wrist.read()
                    if not ret_top or not ret_wrist:
                        print("Failed to grab frames from one or more cameras.")
                        break

                    # Shape: (1, 3, 224, 224)
                    tensor_top = preprocess_image(frame_top).unsqueeze(0).to(device)
                    tensor_wrist = preprocess_image(frame_wrist).unsqueeze(0).to(device)
                    arm_state = torch.tensor([state["angles"]], dtype=torch.float32).to(
                        device
                    )

                    # Double check your training config keys
                    observation = {
                        "observation.images.camera1": tensor_top,
                        "observation.images.camera2": tensor_wrist,
                        "task": task_instruction,
                        "observation.state": arm_state,
                    }

                    batch = preprocess(observation)
                    with torch.inference_mode():
                        action = policy.select_action(batch)
                    action = postprocess(action)

                    # Strip batch dimension -> shape: (num_joints,)
                    motor_commands = action.cpu().numpy()[0]
                    print(f"State: {state['angles']} -> Action: {motor_commands}")

                    # TODO: add safety checks / smoothing before sending
                    message = {
                        "angles": [float(v) for v in motor_commands],
                        "speed": state["speed"],
                    }
                    sock.sendall((json.dumps(message) + "\n").encode())

    except KeyboardInterrupt:
        print("\nRollout loop stopped by user.")

    finally:
        cap_top.release()
        cap_wrist.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
