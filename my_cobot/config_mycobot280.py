from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("mycobot280")
@dataclass
class MyCobot280Config(RobotConfig):
    # "serial": LeRobot runs on the arm's own Raspberry Pi and talks to the Atom over UART.
    # "socket": LeRobot runs on a PC; the Pi runs pymycobot's demo/Server_280.py (TCP, default port 9000).
    transport: str = "socket"
    port: str = "/dev/ttyAMA0"
    baudrate: int = 1_000_000
    host: str = "192.168.10.2"
    socket_port: int = 9000

    # pymycobot speed argument for send_angles (1-100).
    speed: int = 80

    use_gripper: bool = True
    # None -> firmware default (adaptive gripper). 3 = parallel, 4 = flexible (see pymycobot docs).
    gripper_type: int | None = None

    # Safety: max change per send_action call, in degrees. None disables the clamp.
    max_relative_target: float | None = 10.0

    # What to do on disconnect. Releasing servos makes the arm fall, so it is off by default.
    home_on_disconnect: bool = False
    home_angles: list[float] = field(default_factory=lambda: [0.0] * 6)
    release_on_disconnect: bool = False

    # e.g. {"wrist": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)}
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
