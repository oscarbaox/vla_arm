#!/usr/bin/env python3
"""Record two webcams and a TCP arm stream in sync.

Combines the camera discovery from ``camera.py`` with the TCP parsing from
``client.py``. Each frame and each received arm sample is stamped against a
single ``time.monotonic()`` clock (receive-time), so the two MP4s and the arm
JSONL can be aligned after the fact by nearest timestamp.

Output per session::

    recordings/session_<YYYYmmdd_HHMMSS>/
    |- cam0.mp4 / cam1.mp4
    |- cam0_frames.jsonl / cam1_frames.jsonl   # {"frame": i, "t": sec}
    |- arm_data.jsonl                          # {"t": sec, "angles": [...], "speed": s}
    |- metadata.json

Controls:
    s  start recording
    e  stop recording (keeps the app running)
    q  stop recording and quit
"""

import argparse
import json
import queue
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from camera import (
    TARGET_LIFECAM_NAME,
    TARGET_USBCAM_NAME,
    find_camera_index,
    list_v4l2_devices,
)
from client import parse_message


class Session:
    """Shared recording state guarded by a lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.recording = False
        self.t0 = None
        self.arm_samples = []
        self.latest_arm = None
        self.stop_event = threading.Event()


class CameraWorker(threading.Thread):
    """Continuously grab frames; queue them for writing while recording."""

    def __init__(self, cam_id, index, name, width, height, fps, session):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.index = index
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.session = session
        self.queue = queue.Queue()
        self.cap = None
        self.latest_frame = None

    def open(self):
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera '{self.name}' at index {self.index}."
            )
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        # Read back the size the driver actually negotiated.
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def run(self):
        while not self.session.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                continue
            now = time.monotonic()
            with self.session.lock:
                self.latest_frame = frame
                recording = self.session.recording
                t0 = self.session.t0
            if recording and t0 is not None:
                self.queue.put((now - t0, frame))

    def close(self):
        if self.cap is not None:
            self.cap.release()


class WriterWorker(threading.Thread):
    """Drain a camera queue into an MP4 and record per-frame timestamps."""

    def __init__(self, path, fps, width, height, cam_worker):
        super().__init__(daemon=True)
        self.path = path
        self.fps = fps
        self.width = width
        self.height = height
        self.cam_worker = cam_worker
        self.frames = []

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self.path), fourcc, self.fps, (self.width, self.height)
        )
        idx = 0
        try:
            while True:
                item = self.cam_worker.queue.get()
                if item is None:
                    break
                t, frame = item
                writer.write(frame)
                self.frames.append((idx, t))
                idx += 1
        finally:
            writer.release()


class ArmWorker(threading.Thread):
    """Read JSON lines from the TCP server, stamping each on arrival."""

    def __init__(self, host, port, session):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.session = session
        self.connected = False

    def run(self):
        while not self.session.stop_event.is_set():
            try:
                with socket.create_connection((self.host, self.port)) as sock:
                    self.connected = True
                    print(f"Arm stream connected to {self.host}:{self.port}")
                    with sock.makefile("r") as stream:
                        for line in stream:
                            if self.session.stop_event.is_set():
                                break
                            raw = line.strip()
                            if not raw:
                                continue
                            values = parse_message(raw)
                            if values is None:
                                print(f"Wrong format: {raw!r}")
                                continue
                            now = time.monotonic()
                            with self.session.lock:
                                self.session.latest_arm = values
                                if (
                                    self.session.recording
                                    and self.session.t0 is not None
                                ):
                                    self.session.arm_samples.append(
                                        {"t": now - self.session.t0, **values}
                                    )
            except OSError as exc:
                if self.session.stop_event.is_set():
                    break
                print(f"Arm stream error ({exc}); retrying in 1s...")
                self.connected = False
                self.session.stop_event.wait(1.0)
        self.connected = False


def ascii_title(text):
    """OpenCV's Qt backend cannot set a non-ASCII window title when the locale
    is not UTF-8; it then fails to match the existing window and creates a new
    one on every ``imshow``. Force the title to ASCII (e.g. 'Microsoft® ...')."""
    return text.encode("ascii", "replace").decode("ascii")


def annotate(frame, cam_id, arm, t_elapsed, recording):
    text = f"{cam_id}  {'REC' if recording else 'idle'}"
    if recording:
        text += f"  t={t_elapsed:6.2f}s"
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if recording else (0, 255, 0),
        2,
    )
    if arm is not None:
        angles = " ".join(f"{a:6.1f}" for a in arm["angles"])
        cv2.putText(
            frame,
            f"angles: {angles}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"speed: {arm['speed']:.0f}",
            (10, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
    else:
        cv2.putText(
            frame,
            "angles: (no stream)",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 128, 255),
            2,
        )
    return frame


def discover_cameras():
    devices = list_v4l2_devices()
    found = [
        (TARGET_LIFECAM_NAME, find_camera_index(devices, TARGET_LIFECAM_NAME)),
        (TARGET_USBCAM_NAME, find_camera_index(devices, TARGET_USBCAM_NAME)),
    ]
    return [(name, idx) for name, idx in found if idx is not None]


def main():
    parser = argparse.ArgumentParser(
        description="Record two webcams and a TCP arm stream in sync."
    )
    parser.add_argument("--host", default="192.168.10.2")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out", default="recordings")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Auto-stop recording after this many seconds.",
    )
    args = parser.parse_args()

    cams = discover_cameras()
    if len(cams) < 2:
        raise SystemExit(
            f"Expected 2 cameras, found {len(cams)}: {[n for n, _ in cams]}"
        )

    session = Session()
    cameras = []
    for cam_id, (name, index) in enumerate(cams):
        worker = CameraWorker(
            cam_id, index, name, args.width, args.height, args.fps, session
        )
        worker.open()
        worker.window_title = ascii_title(f"cam{cam_id} ({name})")
        print(f"cam{cam_id}: '{name}' at index {index} ({worker.width}x{worker.height})")
        cameras.append(worker)

    for worker in cameras:
        cv2.namedWindow(worker.window_title, cv2.WINDOW_AUTOSIZE)


    arm = ArmWorker(args.host, args.port, session)
    arm.start()
    for worker in cameras:
        worker.start()

    out_root = Path(args.out)
    writers = []
    record_start = None
    record_dir = None
    wall_start = None

    def start_recording():
        nonlocal writers, record_start, record_dir, wall_start
        if session.recording:
            return
        record_dir = out_root / f"session_{datetime.now():%Y%m%d_%H%M%S}"
        record_dir.mkdir(parents=True, exist_ok=True)
        with session.lock:
            session.arm_samples = []
            session.t0 = time.monotonic()
            record_start = session.t0
        wall_start = datetime.now()
        writers = []
        for worker in cameras:
            writer = WriterWorker(
                record_dir / f"cam{worker.cam_id}.mp4",
                args.fps,
                worker.width,
                worker.height,
                worker,
            )
            writer.start()
            writers.append(writer)
        with session.lock:
            session.recording = True
        print(f"Recording started -> {record_dir}")

    def stop_recording():
        nonlocal writers
        if not session.recording:
            return
        with session.lock:
            session.recording = False
            t0 = session.t0
            arm_samples = list(session.arm_samples)
        for worker in cameras:
            worker.queue.put(None)
        for writer in writers:
            writer.join()
        for writer in writers:
            cam_id = writer.cam_worker.cam_id
            with open(record_dir / f"cam{cam_id}_frames.jsonl", "w") as fh:
                for idx, t in writer.frames:
                    fh.write(json.dumps({"frame": idx, "t": t}) + "\n")
        with open(record_dir / "arm_data.jsonl", "w") as fh:
            for sample in arm_samples:
                fh.write(json.dumps(sample) + "\n")
        metadata = {
            "wall_start": wall_start.isoformat(),
            "fps": args.fps,
            "cameras": [
                {
                    "id": w.cam_id,
                    "name": w.name,
                    "index": w.index,
                    "width": w.width,
                    "height": w.height,
                }
                for w in cameras
            ],
            "arm_host": args.host,
            "arm_port": args.port,
            "arm_samples": len(arm_samples),
            "frame_counts": [len(w.frames) for w in writers],
            "duration": time.monotonic() - t0,
        }
        with open(record_dir / "metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2)
        print(f"Recording stopped -> {record_dir}")
        writers = []

    print("Controls: 's' start  'e' stop  'q' quit")
    try:
        while not session.stop_event.is_set():
            with session.lock:
                recording = session.recording
                t0 = session.t0
                arm_latest = session.latest_arm
            t_elapsed = (time.monotonic() - t0) if (recording and t0) else 0.0
            for worker in cameras:
                frame = worker.latest_frame
                if frame is None:
                    continue
                cv2.imshow(
                    worker.window_title,
                    annotate(
                        frame.copy(),
                        f"cam{worker.cam_id}",
                        arm_latest,
                        t_elapsed,
                        recording,
                    ),
                )
            if (
                recording
                and args.duration is not None
                and time.monotonic() - record_start >= args.duration
            ):
                stop_recording()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                start_recording()
            elif key == ord("e"):
                stop_recording()
            elif key == ord("q"):
                stop_recording()
                session.stop_event.set()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop_event.set()
        if session.recording:
            stop_recording()
        for worker in cameras:
            worker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
