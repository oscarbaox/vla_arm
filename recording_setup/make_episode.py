#!/usr/bin/env python3
"""Convert ``record_session.py`` recordings into a LeRobotDataset.

Each ``session_*`` directory (two MP4s + ``*_frames.jsonl`` sidecars +
``arm_data.jsonl``) becomes one episode. All streams are resampled onto a
uniform timeline at the dataset fps (nearest-timestamp match).

Sessions can be given directly, or by passing a parent directory (e.g.
``recordings``) in which case every ``session_*`` subdirectory is used.

Example::

    python3 make_episode.py recordings \\
        --repo-id local/datasetv2 \\
        --root datasetv2 \\
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

GRIPPER_INDEX = JOINT_NAMES.index("gripper")


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
    """Check new session features fit an existing dataset's schema.

    Ignores auto-added index features and per-feature video ``info`` metadata.
    Also rejects video keys present in ``existing`` but absent from ``new`` so
    a session with fewer cameras can't silently drop a stream.
    """
    for key, spec in new.items():
        if key not in existing:
            return False
        old = existing[key]
        if old.get("dtype") != spec.get("dtype"):
            return False
        if tuple(old.get("shape", ())) != tuple(spec.get("shape", ())):
            return False
    for key, spec in existing.items():
        if spec.get("dtype") == "video" and key not in new:
            return False
    return True


def discover_sessions(paths):
    """Expand session directories and parent directories into an ordered list."""
    sessions = []
    for path in paths:
        if (path / "metadata.json").is_file():
            matched = [path]
        elif path.is_dir():
            matched = sorted(
                p
                for p in path.iterdir()
                if p.is_dir() and p.name.startswith("session_") and (p / "metadata.json").is_file()
            )
        else:
            raise SystemExit(f"Not a session directory or parent directory: {path}")
        if not matched:
            raise SystemExit(f"No session_* directories found in {path}")
        sessions.extend(matched)

    seen = set()
    unique = []
    for session in sessions:
        key = session.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(session)
    return unique


def nearest_index(times, t):
    i = bisect.bisect_left(times, t)
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i if abs(times[i] - t) < abs(times[i - 1] - t) else i - 1


def interp_angles(arm, arm_times, t):
    """Linearly interpolate joint angles at ``t``; the gripper stays binary."""
    i = bisect.bisect_left(arm_times, t)
    if i <= 0:
        return np.asarray(arm[0]["angles"], dtype=np.float32)
    if i >= len(arm_times):
        return np.asarray(arm[-1]["angles"], dtype=np.float32)
    t0, t1 = arm_times[i - 1], arm_times[i]
    a0 = np.asarray(arm[i - 1]["angles"], dtype=np.float32)
    a1 = np.asarray(arm[i]["angles"], dtype=np.float32)
    w = (t - t0) / (t1 - t0)
    out = a0 + w * (a1 - a0)
    out[GRIPPER_INDEX] = a0[GRIPPER_INDEX] if w < 0.5 else a1[GRIPPER_INDEX]
    return out


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


def features_from_meta(meta):
    cameras = [
        {
            "key": f"observation.images.camera{int(cam['id']) + 1}",
            "height": cam["height"],
            "width": cam["width"],
        }
        for cam in meta["cameras"]
    ]
    return build_features(cameras)


def add_episode(dataset, session, fps, task, action_from):
    _, cameras, arm = load_session(session)
    if len(arm) < 2:
        raise SystemExit(f"Need at least 2 arm samples in {session}, found {len(arm)}")

    arm_times = [a["t"] for a in arm]
    end_t = arm_times[-1]
    n = int(end_t * fps) + 1
    timesteps = [k / fps for k in range(n)]

    states, speeds = [], []
    for t in timesteps:
        sample = arm[nearest_index(arm_times, t)]
        states.append(interp_angles(arm, arm_times, t))
        speeds.append(np.asarray([sample["speed"]], dtype=np.float32))
    states = np.stack(states)
    speeds = np.stack(speeds)
    if action_from == "next":
        actions = np.concatenate([states[1:], states[-1:]], axis=0)
    else:
        actions = states

    pickers = {cam["key"]: FramePicker(cam["video"], cam["times"]) for cam in cameras}
    try:
        for k, t in enumerate(timesteps):
            frame = {
                "task": task,
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
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sessions",
        nargs="+",
        type=Path,
        help="session_* directories, or parent directories containing them",
    )
    parser.add_argument("--repo-id", default=None, help="e.g. local/datasetv2")
    parser.add_argument("--root", type=Path, default=None, help="Dataset output directory")
    parser.add_argument(
        "--fps", type=int, default=None, help="Override dataset fps (default: first session's fps)"
    )
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
        help="Append these episodes to the existing dataset at --root instead of creating a new one",
    )
    args = parser.parse_args()

    if args.append and args.overwrite:
        raise SystemExit("--append and --overwrite are mutually exclusive")

    sessions = discover_sessions(args.sessions)

    reference = None
    metas = {}
    for session in sessions:
        meta = json.loads((session / "metadata.json").read_text())
        metas[session] = meta
        features = features_from_meta(meta)
        if reference is None:
            reference = features
        elif not features_compatible(reference, features):
            raise SystemExit(
                f"Session features do not match {sessions[0]}: {session}"
            )

    root = args.root or Path("dataset")
    repo_id = args.repo_id or f"local/{root.name}"

    if args.append:
        if not root.exists():
            raise SystemExit(f"--append given but dataset root does not exist: {root}")
        dataset = LeRobotDataset(repo_id=repo_id, root=root, vcodec="h264")
        if not features_compatible(dataset.features, reference):
            raise SystemExit(
                "Existing dataset features do not match the sessions:\n"
                f"  existing: {dataset.features}\n"
                f"  sessions: {reference}"
            )
        if args.fps is not None and args.fps != dataset.fps:
            raise SystemExit(
                f"--fps {args.fps} does not match existing dataset fps {dataset.fps}"
            )
        fps = dataset.fps
    else:
        if root.exists() and not args.overwrite:
            raise SystemExit(
                f"{root} already exists. Use --overwrite to replace it "
                f"or --append to add to it."
            )
        if root.exists():
            shutil.rmtree(root)
        fps = args.fps or metas[sessions[0]]["fps"]
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=reference,
            root=root,
            robot_type=args.robot_type,
            use_videos=True,
            vcodec="h264",
        )

    print(f"Writing {len(sessions)} episode(s) to {root} @ {fps} fps")
    try:
        for i, session in enumerate(sessions, 1):
            n = add_episode(dataset, session, fps, args.task, args.action_from)
            episode_index = dataset.meta.total_episodes - 1
            print(f"  [{i}/{len(sessions)}] {session.name}: {n} frames -> episode {episode_index}")
    finally:
        dataset.finalize()

    print(
        f"dataset now has {dataset.meta.total_episodes} episode(s), "
        f"{dataset.meta.total_frames} frames"
    )
    print(
        f"repo_id={repo_id} robot_type={args.robot_type} action_from={args.action_from}"
    )


if __name__ == "__main__":
    main()
