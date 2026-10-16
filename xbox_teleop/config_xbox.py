from dataclasses import dataclass, field

from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xbox")
@dataclass
class XboxTeleopConfig(TeleoperatorConfig):
    # pygame joystick index, 0 = first controller found.
    joystick_index: int = 0

    # Stick values below this magnitude are treated as zero (sticks rarely rest at exactly 0.0).
    deadzone: float = 0.2

    # End-effector speed at full stick deflection and speed_scale = 1.0.
    # 0.04 m/s and 20 deg/s match the old Pi script (2 mm/tick and 1 deg/tick at 20 Hz).
    linear_speed: float = 0.04  # m/s
    angular_speed: float = 0.35  # rad/s (~20 deg/s)

    # LB/RB change speed_scale by speed_scale_step, clamped to [min_speed_scale, max_speed_scale].
    speed_scale: float = 1.0
    speed_scale_step: float = 0.25
    min_speed_scale: float = 0.25
    max_speed_scale: float = 2.5

    # Cap on how far (m) the accumulated target may drift from the anchor pose on each axis.
    # Prevents "windup": holding a stick against a workspace bound would otherwise keep growing
    # the target, and the arm would not respond to the opposite direction until it unwound.
    max_offset_m: float = 0.3

    # Axis index -> (target key, sign). Axis numbering depends on the OS driver and on USB vs
    # Bluetooth, so VERIFY these on the machine you teleoperate from. Defaults are the Linux
    # xpad (USB) layout: 0/1 = left stick X/Y, 3/4 = right stick X/Y (2/5 are the triggers).
    # The old Pi script used 2/3 for the right stick.
    axis_map: dict[int, tuple[str, int]] = field(
        default_factory=lambda: {
            0: ("y", -1),  # left stick X  -> base Y
            1: ("x", -1),  # left stick Y  -> base X (push forward = +X)
            3: ("wz", +1),  # right stick X -> rotate about the tool z axis
            4: ("z", -1),  # right stick Y -> base Z (push forward = up)
        }
    )

    # Button indices (Linux xpad layout).
    button_gripper_open: int = 0  # A
    button_gripper_close: int = 1  # B
    button_reanchor: int = 3  # Y
    button_slower: int = 4  # LB
    button_faster: int = 5  # RB
