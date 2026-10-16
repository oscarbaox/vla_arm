import logging
import time
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot

from .config_mycobot280 import MyCobot280Config

logger = logging.getLogger(__name__)

JOINT_NAMES = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]

# Joint limits in degrees, from the pymycobot MyCobot280 API docs (send_angle table).
JOINT_LIMITS = {
    "joint2_to_joint1": (-168.0, 168.0),
    "joint3_to_joint2": (-135.0, 135.0),
    "joint4_to_joint3": (-150.0, 150.0),
    "joint5_to_joint4": (-145.0, 145.0),
    "joint6_to_joint5": (-155.0, 160.0),
    "joint6output_to_joint6": (-180.0, 180.0),
}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class MyCobot280(Robot):
    """myCobot 280 Pi wrapped as a LeRobot Robot.

    The servos sit behind the Atom controller's firmware, so LeRobot's FeetechMotorsBus is not usable;
    all I/O goes through pymycobot. Positions are in degrees (joints) and 0-100 (gripper).
    """

    config_class = MyCobot280Config
    name = "mycobot280"

    def __init__(self, config: MyCobot280Config):
        super().__init__(config)
        self.config = config
        self.mc = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self._last_angles: list[float] | None = None
        self._last_gripper: float | None = None
        self._last_gripper_cmd: int | None = None

    # ---------- feature contract (must work while disconnected) ----------

    @property
    def _motors_ft(self) -> dict[str, type]:
        ft = {f"{j}.pos": float for j in JOINT_NAMES}
        if self.config.use_gripper:
            ft["gripper.pos"] = float
        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            name: (cfg.height, cfg.width, 3)
            for name, cfg in self.config.cameras.items()
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._motors_ft

    # ---------- connection ----------

    @property
    def is_connected(self) -> bool:
        return self.mc is not None and all(
            cam.is_connected for cam in self.cameras.values()
        )

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected.")

        # Lazy import so observation_features etc. work on machines without pymycobot installed.
        from pymycobot import MyCobot280 as _MyCobot280
        from pymycobot import MyCobot280Socket

        if self.config.transport == "serial":
            self.mc = _MyCobot280(self.config.port, self.config.baudrate)
        elif self.config.transport == "socket":
            self.mc = MyCobot280Socket(self.config.host, self.config.socket_port)
        else:
            raise ValueError(f"Unknown transport {self.config.transport!r}")
        time.sleep(0.5)  # let the Atom settle after the port opens

        if calibrate and not self.is_calibrated:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        # Prime the state cache so the first send_action can apply the relative clamp.
        self._read_angles()
        logger.info(f"{self} connected.")

    # ---------- calibration / configuration ----------

    @property
    def is_calibrated(self) -> bool:
        # The firmware stores joint zeros and reports calibrated degrees, so there is nothing
        # for LeRobot to persist. Re-zeroing is done with mc.set_servo_calibration(id), which is
        # destructive and deliberately not automated here.
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.mc.power_on()
        self.mc.focus_all_servos()
        # Refresh mode: the newest command pre-empts queued ones. Essential for streaming
        # policy actions at 10-30 Hz; in queue mode the arm lags further and further behind.
        self.mc.set_fresh_mode(1)

    # ---------- I/O ----------

    def _read_angles(self) -> list[float]:
        for _ in range(3):
            angles = self.mc.get_angles()
            if isinstance(angles, list) and len(angles) == 6:
                self._last_angles = [float(a) for a in angles]
                return self._last_angles
        if self._last_angles is None:
            raise RuntimeError(f"{self}: get_angles() returned no valid data.")
        logger.warning(f"{self}: get_angles() failed, reusing last reading.")
        return self._last_angles

    def _read_gripper(self) -> float:
        kwargs = (
            {}
            if self.config.gripper_type is None
            else {"gripper_type": self.config.gripper_type}
        )
        value = self.mc.get_gripper_value(**kwargs)
        if isinstance(value, (int, float)) and 0 <= value <= 100:
            self._last_gripper = float(value)
        elif self._last_gripper is None:
            self._last_gripper = float(self._last_gripper_cmd or 0)
        return self._last_gripper

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        angles = self._read_angles()
        obs: dict[str, Any] = {f"{j}.pos": a for j, a in zip(JOINT_NAMES, angles)}
        if self.config.use_gripper:
            obs["gripper.pos"] = self._read_gripper()

        for key, cam in self.cameras.items():
            obs[key] = cam.async_read()
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        goal = []
        for i, j in enumerate(JOINT_NAMES):
            target = _clip(float(action[f"{j}.pos"]), *JOINT_LIMITS[j])
            if (
                self.config.max_relative_target is not None
                and self._last_angles is not None
            ):
                cur = self._last_angles[i]
                m = self.config.max_relative_target
                target = cur + _clip(target - cur, -m, m)
            goal.append(round(target, 2))

        self.mc.send_angles(goal, self.config.speed)
        sent = {f"{j}.pos": g for j, g in zip(JOINT_NAMES, goal)}

        if self.config.use_gripper and "gripper.pos" in action:
            g = int(round(_clip(float(action["gripper.pos"]), 0, 100)))
            # Skip redundant gripper commands to save serial bandwidth.
            if self._last_gripper_cmd is None or abs(g - self._last_gripper_cmd) >= 2:
                kwargs = (
                    {}
                    if self.config.gripper_type is None
                    else {"gripper_type": self.config.gripper_type}
                )
                self.mc.set_gripper_value(g, self.config.speed, **kwargs)
                self._last_gripper_cmd = g
            sent["gripper.pos"] = float(self._last_gripper_cmd)

        return sent

    def disconnect(self) -> None:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")
        try:
            if self.config.home_on_disconnect:
                self.mc.sync_send_angles(self.config.home_angles, 30, timeout=10)
            if self.config.release_on_disconnect:
                self.mc.release_all_servos()
        finally:
            port = getattr(self.mc, "_serial_port", None)
            if port is not None and getattr(port, "is_open", False):
                port.close()
            for cam in self.cameras.values():
                cam.disconnect()
            self.mc = None
            logger.info(f"{self} disconnected.")
