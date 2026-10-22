#!/usr/bin/env python3
"""Convert a raw ``record_session.py`` recording into a single LeRobot episode.

Reads a ``session_*`` directory (two MP4s + ``*_frames.jsonl`` sidecars +
``arm_data.jsonl``) and writes one episode into a LeRobotDataset, resampling all
streams onto a uniform timeline at the session fps (nearest-timestamp match).

Example::

    python3 make_episode.py recordings/session_20261020_194556 \\
        --repo-id local/mycobot_194556 \\
        --root recordings/episodes/session_20261020_194556 \\
        --task "pick up the cube"
"""

import argparse
import bisect
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
]


def read_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class FramePicker:
    """Return the frame nearest a target time, reading the video forward once."""

    def __init__(self, video_path, frame_times):
        self.video_path = video_path
        self.frame_times = frame_times
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
        self.idx = -1
        self.frame = None
        self._advance()

    def _advance(self):
        ok, frame = self.cap.read()
        if not ok:
            return False
        self.idx += 1
        self.frame = frame
        return True

    def at(self, t):
        while self.idx + 1 < len(self.frame_times):
            t_cur = self.frame_times[self.idx]
            t_next = self.frame_times[self.idx + 1]
            if abs(t_next - t) <= abs(t_cur - t):
                if not self._advance():
                    break
            else:
                break
        if self.frame is None:
            raise RuntimeError(f"No frames decoded from {self.video_path}")
        return self.frame

    def release(self):
        self.cap.release()


def features_compatible(existing, new):
    """Check the new session's features fit an existing dataset's schema.

    Ignores auto-added index features and per-feature video ``info`` metadata.
    """
    for key, spec in new.items():
        if key not in existing:
            return False
        old = existing[key]
        if old.get("dtype") != spec.get("dtype"):
            return False
        if tuple(old.get("shape", ())) != tuple(spec.get("shape", ())):
            return False
    return True


def nearest_index(times, t):
    i = bisect.bisect_left(times, t)
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i if abs(times[i] - t) < abs(times[i - 1] - t) else i - 1


def load_session(session):
    meta = json.loads((session / "metadata.json").read_text())
    cameras = []
    for cam in meta["cameras"]:
        cid = int(cam["id"])
        cid_ = cid + 1
        frames = read_jsonl(session / f"cam{cid}_frames.jsonl")
        cameras.append(
            {
                "key": f"observation.images.camera{cid_}",
                "video": session / f"cam{cid}.mp4",
                "times": [f["t"] for f in frames],
                "height": cam["height"],
                "width": cam["width"],
            }
        )
    arm = read_jsonl(session / "arm_data.jsonl")
    return meta, cameras, arm


def build_features(cameras):
    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
        "observation.speed": {"dtype": "float32", "shape": (1,), "names": ["speed"]},
        "action": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
    }
    for cam in cameras:
        features[cam["key"]] = {
            "dtype": "video",
            "shape": (cam["height"], cam["width"], 3),
            "names": ["height", "width", "channels"],
        }
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Path to a session_* directory")
    parser.add_argument("--repo-id", default=None, help="e.g. local/mycobot_194556")
    parser.add_argument(
        "--root", type=Path, default=None, help="Dataset output directory"
    )
    parser.add_argument("--fps", type=int, default=None, help="Override session fps")
    parser.add_argument("--task", default="teleoperate the arm", help="Task string")
    parser.add_argument("--robot-type", default="mycobot280")
    parser.add_argument(
        "--action-from",
        choices=["state", "next"],
        default="state",
        help="'state' -> action = current angles; 'next' -> action = next timestep angles",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Delete --root if it exists"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append this episode to the existing dataset at --root instead of creating a new one",
    )
    args = parser.parse_args()

    if args.append and args.overwrite:
        raise SystemExit("--append and --overwrite are mutually exclusive")

    session = args.session
    meta, cameras, arm = load_session(session)
    fps = args.fps or meta["fps"]
    repo_id = args.repo_id or f"local/{session.name}"
    root = args.root or (session.parent / f"episode_{session.name}")

    if len(arm) < 2:
        raise SystemExit(f"Need at least 2 arm samples, found {len(arm)}")

    arm_times = [a["t"] for a in arm]
    end_t = arm_times[-1]
    n = int(end_t * fps) + 1
    timesteps = [k / fps for k in range(n)]

    states, speeds = [], []
    for t in timesteps:
        sample = arm[nearest_index(arm_times, t)]
        states.append(np.asarray(sample["angles"], dtype=np.float32))
        speeds.append(np.asarray([sample["speed"]], dtype=np.float32))
    states = np.stack(states)
    speeds = np.stack(speeds)
    if args.action_from == "next":
        actions = np.concatenate([states[1:], states[-1:]], axis=0)
    else:
        actions = states

    if args.overwrite and root.exists():
        shutil.rmtree(root)

    features = build_features(cameras)
    if args.append:
        if not root.exists():
            raise SystemExit(f"--append given but dataset root does not exist: {root}")
        dataset = LeRobotDataset(repo_id=repo_id, root=root, vcodec="h264")
        if not features_compatible(dataset.features, features):
            raise SystemExit(
                "Existing dataset features do not match this session:\n"
                f"  existing: {dataset.features}\n"
                f"  session : {features}"
            )
    else:
        if root.exists():
            raise SystemExit(
                f"{root} already exists. Use --append to add an episode "
                f"or --overwrite to replace the dataset."
            )
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=root,
            robot_type=args.robot_type,
            use_videos=True,
            vcodec="h264",
        )

    pickers = {cam["key"]: FramePicker(cam["video"], cam["times"]) for cam in cameras}
    try:
        for k, t in enumerate(timesteps):
            frame = {
                "task": args.task,
                "observation.state": states[k],
                "observation.speed": speeds[k],
                "action": actions[k],
            }
            for cam in cameras:
                bgr = pickers[cam["key"]].at(t)
                frame[cam["key"]] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dataset.add_frame(frame)
    finally:
        for picker in pickers.values():
            picker.release()

    dataset.save_episode()
    dataset.finalize()

    print(f"Wrote episode ({n} frames @ {fps} fps) to {root}")
    print(
        f"dataset now has {dataset.meta.total_episodes} episode(s), {dataset.meta.total_frames} frames"
    )
    print(
        f"repo_id={repo_id} robot_type={args.robot_type} action_from={args.action_from}"
    )


if __name__ == "__main__":
    main()
