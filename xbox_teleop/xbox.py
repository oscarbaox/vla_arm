import logging
import time
from typing import Any

from lerobot.teleoperators import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_xbox import XboxTeleopConfig

logger = logging.getLogger(__name__)

_LINEAR = ("x", "y", "z")


def _apply_deadzone(value: float, deadzone: float) -> float:
    """Zero inside the deadzone, then rescale so output ramps smoothly from 0 to +-1."""
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


class XboxTeleop(Teleoperator):
    """Xbox controller as an end-effector *rate* teleoperator.

    Sticks command velocities; this class integrates them into a target displacement from an
    anchor pose. The output keys (`enabled`, `target_*`, `gripper_vel`) are exactly what
    `EEReferenceAndDelta` consumes, so the controller plugs into the same
    EE -> bounds -> IK -> joints pipeline as the phone teleop (drop `MapPhoneActionToRobotAction`
    and use `end_effector_step_sizes` of 1.0, since targets are already in metres).

    The anchor is the robot's pose when the pipeline first sees `enabled=True`. Pressing Y
    re-anchors: offsets reset to zero and the reference is re-latched at the arm's current pose,
    which clears accumulated IK/tracking error.
    """

    config_class = XboxTeleopConfig
    name = "xbox"

    def __init__(self, config: XboxTeleopConfig):
        super().__init__(config)
        self.config = config
        self.joystick = None
        self._offset = {"x": 0.0, "y": 0.0, "z": 0.0, "wz": 0.0}
        self._speed_scale = config.speed_scale
        self._last_t: float | None = None
        self._reanchor_pending = False

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "enabled": bool,
            "target_x": float,
            "target_y": float,
            "target_z": float,
            "target_wx": float,
            "target_wy": float,
            "target_wz": float,
            "gripper_vel": float,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.joystick is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        # Lazy import so the config/features work on machines without pygame installed.
        import pygame

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= self.config.joystick_index:
            raise ConnectionError(f"{self}: no controller at index {self.config.joystick_index}.")

        self.joystick = pygame.joystick.Joystick(self.config.joystick_index)
        self.joystick.init()
        logger.info(
            f"{self} connected to {self.joystick.get_name()} "
            f"({self.joystick.get_numaxes()} axes, {self.joystick.get_numbuttons()} buttons)."
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _handle_button_presses(self) -> None:
        import pygame

        # Edge-triggered buttons come from the event queue so one press = one step.
        for event in pygame.event.get():
            if event.type != pygame.JOYBUTTONDOWN:
                continue
            if event.button == self.config.button_slower:
                self._speed_scale = max(
                    self.config.min_speed_scale, self._speed_scale - self.config.speed_scale_step
                )
                logger.info(f"{self} speed scale = {self._speed_scale:.2f}")
            elif event.button == self.config.button_faster:
                self._speed_scale = min(
                    self.config.max_speed_scale, self._speed_scale + self.config.speed_scale_step
                )
                logger.info(f"{self} speed scale = {self._speed_scale:.2f}")
            elif event.button == self.config.button_reanchor:
                self._reanchor_pending = True

    @check_if_not_connected
    def get_action(self) -> dict[str, Any]:
        self._handle_button_presses()

        # Integrate with measured wall-clock dt so speed does not depend on the loop rate.
        # dt is capped so a stalled loop (e.g. a slow network round trip) cannot cause a jump.
        now = time.perf_counter()
        dt = 0.0 if self._last_t is None else min(now - self._last_t, 0.1)
        self._last_t = now

        if self._reanchor_pending:
            # One disabled frame makes EEReferenceAndDelta re-latch its reference on the next
            # enabled frame; zeroed offsets then mean "hold the current pose".
            self._reanchor_pending = False
            self._offset = dict.fromkeys(self._offset, 0.0)
            logger.info(f"{self} re-anchored at the current pose.")
            return self._make_action(enabled=False, gripper_vel=0.0)

        for axis, (key, sign) in self.config.axis_map.items():
            value = _apply_deadzone(self.joystick.get_axis(axis), self.config.deadzone)
            speed = self.config.linear_speed if key in _LINEAR else self.config.angular_speed
            self._offset[key] += sign * value * speed * self._speed_scale * dt

        m = self.config.max_offset_m
        for key in _LINEAR:
            self._offset[key] = max(-m, min(m, self._offset[key]))

        gripper_vel = float(self.joystick.get_button(self.config.button_gripper_open)) - float(
            self.joystick.get_button(self.config.button_gripper_close)
        )
        return self._make_action(enabled=True, gripper_vel=gripper_vel)

    def _make_action(self, enabled: bool, gripper_vel: float) -> dict[str, Any]:
        return {
            "enabled": enabled,
            "target_x": self._offset["x"],
            "target_y": self._offset["y"],
            "target_z": self._offset["z"],
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": self._offset["wz"],
            "gripper_vel": gripper_vel,
        }

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # A controller has no feedback channel here (rumble could go here later).
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        import pygame

        self.joystick.quit()
        self.joystick = None
        pygame.joystick.quit()
        logger.info(f"{self} disconnected.")
