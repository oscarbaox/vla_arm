#!/usr/bin/env python3
"""Smooth streaming joystick teleop for myCobot 280 (pymycobot, direct serial)."""

import sys, time, threading
import pygame
from pymycobot import MyCobot280
from gpiozero import AngularServo

servo = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023)

PORT, BAUD = "/dev/ttyAMA0", 1000000
HOME = [0.0, 8.0, -127.0, 40.0, 0.0, 0.0]
LIMITS = [(-281, 281), (-281, 281), (-70, 412), (-180, 180), (-180, 180), (-180, 180)]
DEADZONE = 0.2
TICK_HZ = 20

mc = MyCobot280(PORT, BAUD)
mc.set_fresh_mode(1)
time.sleep(0.5)

lock = threading.Lock()
target = mc.get_coords()
step = 2.0  # mm per tick (XYZ) at full deflection; rotations use step / 2
speed = 40  # 10-100
going_home = False
running = True


def clamp(t):
    return [max(lo, min(hi, v)) for v, (lo, hi) in zip(t, LIMITS)]


def streamer():
    """Background loop: keep pushing the current target at ~10 Hz."""
    while running:
        if going_home:
            time.sleep(0.1)
            continue
        with lock:
            t = target[:]
        mc.send_coords(t, speed, 1, _async=True)
        time.sleep(0.1)


threading.Thread(target=streamer, daemon=True).start()

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller found.")
    running = False
    sys.exit(1)

controller = pygame.joystick.Joystick(0)
controller.init()
print(f"Connected to: {controller.get_name()}")

AXES = {  # axis index -> (coord index, sign)
    0: (1, -1),  # left stick X  -> Y
    1: (0, -1),  # left stick Y  -> X
    2: (5, +1),  # right stick X -> RZ
    3: (2, -1),  # right stick Y -> Z
}

BUTTONS = {
    0: "grip_open",
    1: "grip_close",
    2: "step_down",
    3: "home",
    4: "speed_down",
    5: "speed_up",
    9: "quit",
}

print("""Streaming joystick teleop (hold sticks to move, release to stop):
  left stick: X/Y   right stick: Z/RZ
  A: gripper open   B: gripper close
  X: step -0.5 (now %.1f)   Y: go home
  LB/RB: speed -/+10 (now %d)   Start: quit
Current target: """ % (step, speed))

clock = pygame.time.Clock()
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                action = BUTTONS.get(event.button)
                if action == "grip_open":
                    servo.angle = -90
                elif action == "grip_close":
                    servo.angle = 0
                elif action == "step_down":
                    step = max(0.5, step - 0.5)
                    print("\nstep = %.1f" % step)
                elif action == "home":
                    going_home = True
                    mc.send_angles(HOME, 50)  # blocking; arm goes home
                    time.sleep(0.3)
                    going_home = False
                    with lock:
                        target = clamp(mc.get_coords())
                    print("\nhome. target reset.")
                elif action == "speed_down":
                    speed = max(10, speed - 10)
                    print("\nspeed = %d" % speed)
                elif action == "speed_up":
                    speed = min(100, speed + 10)
                    print("\nspeed = %d" % speed)
                elif action == "quit":
                    running = False
        for i, (idx, sign) in AXES.items():
            val = controller.get_axis(i)
            if abs(val) > DEADZONE:
                with lock:
                    target[idx] += sign * val * (step if idx < 3 else step / 2)
                    target = clamp(target)
        print("\rtarget: " + str([round(c, 1) for c in target]), end="")
        clock.tick(TICK_HZ)
finally:
    running = False
    time.sleep(0.1)  # let streamer fire one last command
    pygame.quit()
    print("\nbye.")
