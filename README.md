# 2026 Fall ISR: Exploring and Deploying Visual-Language-Action Models

This repository contains the data-collection and training pipeline for a
myCobot 280 arm. Raw teleoperation sessions (two webcams + a stream of joint
angles over TCP) are recorded, converted into a LeRobot dataset, and then used
to fine-tune a SmolVLA policy.

## Repository layout

| Path | Description |
| --- | --- |
| `recording_setup/` | Data collection: record raw sessions and convert them into LeRobot episodes |
| `teleoperate.py` | Phone-based teleoperation of the myCobot using LeRobot |
| `my_cobot/` | myCobot 280 LeRobot robot implementation + URDF |
| `xbox_teleop/` | Xbox-controller teleoperation |
| `robot_internal/` | On-robot joystick control and TCP joint-angle server |

---

## `recording_setup/`

This folder is the data-collection pipeline. It has two stages:

1. **`record_session.py`** records synchronized raw data into a `session_*`
   directory.
2. **`make_episode.py`** converts a `session_*` directory into a single
   [LeRobotDataset](https://github.com/huggingface/lerobot) episode.

Supporting modules:

- **`camera.py`** — discovers the two USB webcams by name via
  `v4l2-ctl --list-devices` and resolves them to `/dev/videoN` indices.
- **`client.py`** — parses the JSON lines streamed by the arm's TCP server.

### Prerequisites

- `v4l2-ctl` available on `PATH` (from `v4l-utils`) so the cameras can be
  discovered by name. The two expected cameras are:
  - `Microsoft® LifeCam Cinema(TM)`
  - `USB Camera: USB Camera`
- The arm must be streaming joint angles as newline-delimited JSON over TCP,
  e.g. from `robot_internal/joystick_control.py` (default `0.0.0.0:9999`).
  Each line looks like:

  ```json
  {"angles": [a1, a2, a3, a4, a5, a6, gripper], "speed": 40}
  ```

- Python packages: `opencv-python` (cv2) for `record_session.py`, and
  `lerobot`, `numpy`, `opencv-python` for `make_episode.py`.

---

### `record_session.py`

Records two webcams and the TCP arm stream **in sync**. Every frame and every
received arm sample is stamped against a single `time.monotonic()` clock
(receive time), so the two MP4s and the arm JSONL can be aligned afterwards by
nearest timestamp.

#### Usage

```bash
cd recording_setup
python3 record_session.py --host 192.168.10.2 --port 9999 --fps 15 \
    --width 640 --height 480 --out recordings
```

#### Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--host` | `192.168.10.2` | Arm TCP server host |
| `--port` | `9999` | Arm TCP server port |
| `--fps` | `15` | Recording frame rate |
| `--width` | `640` | Capture width |
| `--height` | `480` | Capture height |
| `--out` | `recordings` | Root directory for session folders |
| `--duration` | `None` | Auto-stop recording after N seconds |

#### Controls

A window is shown per camera with a live overlay of the recording timer, the
latest joint angles, and the arm speed.

| Key | Action |
| --- | --- |
| `s` | Start recording |
| `e` | Stop recording (keeps the app running) |
| `q` | Stop recording and quit |

#### Output

Each recording creates `recordings/session_<YYYYmmdd_HHMMSS>/`:

```
recordings/session_<YYYYmmdd_HHMMSS>/
├── cam0.mp4 / cam1.mp4          # one MP4 per camera (mp4v codec)
├── cam0_frames.jsonl            # {"frame": i, "t": seconds_since_start}
├── cam1_frames.jsonl
├── arm_data.jsonl               # {"t": seconds, "angles": [...], "speed": s}
└── metadata.json                # fps, camera names/indices/sizes, arm host/port, counts, duration
```

The `t` timestamps in the frame sidecars and in `arm_data.jsonl` share the same
zero point (the moment recording started), which is what makes the streams
alignable.

---

### `make_episode.py`

Converts one raw `session_*` directory into a single LeRobot episode. It reads
the two MP4s, their `*_frames.jsonl` timestamp sidecars, and `arm_data.jsonl`,
then **resamples every stream onto a uniform timeline** at the session fps
using nearest-timestamp matching.

Each output frame contains:

- `observation.state` — 7 joint angles (6 joints + gripper)
- `observation.speed` — arm speed
- `action` — either the current angles (`--action-from state`, default) or the
  next timestep's angles (`--action-from next`)
- `observation.images.camera1` / `observation.images.camera2` — RGB video

#### Usage

```bash
cd recording_setup
python3 make_episode.py recordings/session_20261020_194556 \
    --repo-id local/mycobot_194556 \
    --root recordings/episodes/session_20261020_194556 \
    --task "pick up the cube"
```

To add another episode to an existing dataset, use `--append` with the same
`--root`; features are checked for compatibility before appending:

```bash
python3 make_episode.py recordings/session_20261020_195000 \
    --repo-id local/mycobot_194556 \
    --root recordings/episodes/session_20261020_194556 \
    --task "pick up the cube" \
    --append
```

#### Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `session` | — | Path to a `session_*` directory (required) |
| `--repo-id` | `local/<session>` | Dataset repo id, e.g. `local/mycobot_194556` |
| `--root` | `<session parent>/episode_<session>` | Dataset output directory |
| `--fps` | session fps | Override the session fps |
| `--task` | `teleoperate the arm` | Task string stored with the episode |
| `--robot-type` | `mycobot280` | Robot type in the dataset metadata |
| `--action-from` | `state` | `state` → action = current angles; `next` → action = next-step angles |
| `--overwrite` | off | Delete `--root` if it already exists |
| `--append` | off | Append to the dataset at `--root` instead of creating a new one |

`--append` and `--overwrite` are mutually exclusive.

The generated dataset follows the LeRobot v3.0 layout
(`data/`, `videos/`, `meta/info.json`, `meta/tasks.parquet`, `meta/stats.json`),
using `h264` videos and 15 fps.

The repository already contains a converted dataset at
`recording_setup/episodes/` (3 episodes, 1218 frames, `robot_type=mycobot280`).

---

## Training a SmolVLA policy

Training is done with [LeRobot](https://github.com/huggingface/lerobot) and the
[`SmolVLA`](https://huggingface.co/lerobot/smolvla_base) policy. The command
below fine-tunes `smolvla_base` on the dataset produced by `make_episode.py`:

```bash
lerobot-train   --policy.path=lerobot/smolvla_base   --dataset.repo_id=local/cobot   --dataset.root=/home/koroko/Workspace/vla_arm/recording_setup/episodes   --batch_size=32   --steps=20000   --output_dir=outputs/train/smolvla_mycobot   --job_name=smolvla_mycobot   --policy.device=cuda   --policy.push_to_hub=false   --wandb.enable=false   --policy.empty_cameras=1   --dataset.video_backend=pyav
```

Key flags:

| Flag | Meaning |
| --- | --- |
| `--policy.path=lerobot/smolvla_base` | Start from the pretrained SmolVLA base checkpoint |
| `--dataset.repo_id` / `--dataset.root` | Which dataset to train on (repo id + on-disk path) |
| `--batch_size=32` / `--steps=20000` | Training batch size and total optimizer steps |
| `--output_dir` / `--job_name` | Where checkpoints are written and the run name |
| `--policy.device=cuda` | Train on GPU |
| `--policy.push_to_hub=false` | Keep checkpoints local (don't upload to the Hub) |
| `--wandb.enable=false` | Disable Weights & Biases logging |
| `--policy.empty_cameras=1` | Pad with an empty camera slot (needed when the policy expects more cameras than the dataset has) |
| `--dataset.video_backend=pyav` | Decode videos with PyAV (more robust than torchcodec for some files) |

### 1. Clone and install LeRobot

If LeRobot is not already installed, clone the repo and install it (editable
install, with the SmolVLA extra):

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"
```

`pip install -e ".[smolvla]"` installs the SmolVLA policy dependencies
(transformers, etc.). Use `pip install -e ".[all]"` to install every optional
extra. Verify the install provides the `lerobot-train` entry point:

```bash
lerobot-train --help
```

### 2. Point the command at your dataset

The `--dataset.root` in the command above points at
`/home/koroko/Workspace/vla_arm/recording_setup/episodes`. Change it to the
`--root` you passed to `make_episode.py`, and set `--dataset.repo_id` to the
matching `--repo-id` (e.g. `local/mycobot_194556`).

### 3. Train

Run the full command from the top of this section. Checkpoints and logs are
written under `outputs/train/smolvla_mycobot`. Resume or inspect training with
the LeRobot tooling (`lerobot-train --resume=true ...`, TensorBoard on
`output_dir`, or by loading the checkpoint with `--policy.path=<output_dir>`).
