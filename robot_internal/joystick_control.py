#!/usr/bin/env python3
"""Smooth streaming joystick teleop for myCobot 280 (pymycobot, direct serial)."""

import sys, time, threading, socket, json
import pygame
from pymycobot import MyCobot280
from gpiozero import AngularServo

servo = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023)

PORT, BAUD = "/dev/ttyAMA0", 1000000
TCP_HOST, TCP_PORT = "0.0.0.0", 9999
HOME = [0.0, 8.0, -127.0, 40.0, 0.0, 0.0]
LIMITS = [(-281, 281), (-281, 281), (-70, 412), (-180, 180), (-180, 180), (-180, 180)]
DEADZONE = 0.2
TICK_HZ = 30

mc = MyCobot280(PORT, BAUD)
mc.set_fresh_mode(1)
time.sleep(0.5)

mc.send_angles(HOME, 50)  # blocking; arm goes home
time.sleep(3)

lock = threading.Lock()
target = mc.get_coords()
step = 2.0  # mm per tick (XYZ) at full deflection; rotations use step / 2
speed = 40  # 10-100
going_home = False
running = True
angles = None  # last angles read from the arm; used for streaming to clients
mc_lock = threading.Lock()  # serializes serial access across threads
ARRIVE_TOL = 3.0  # degrees; per-joint tolerance for "arm reached the commanded pose"
ARRIVE_TIMEOUT = 5.0  # seconds to wait for arrival before giving up

clients = []
clients_lock = threading.Lock()


def clamp(t):
    return [max(lo, min(hi, v)) for v, (lo, hi) in zip(t, LIMITS)]


def broadcast(payload):
    data = (json.dumps(payload) + "\n").encode()
    with clients_lock:
        dead = []
        for c in clients:
            try:
                c.sendall(data)
            except OSError:
                dead.append(c)
        for c in dead:
            clients.remove(c)
            try:
                c.close()
            except OSError:
                pass


def handle_client(conn):
    """Read newline-delimited JSON {"angles": [...], "speed": n} and send_angles."""
    buf = b""
    while running:
        try:
            chunk = conn.recv(4096)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
        # coalesce: only the newest complete message matters while the arm is busy
        *lines, buf = buf.split(b"\n")
        latest = None
        for line in lines:
            try:
                msg = json.loads(line)
                cmd = [float(v) for v in msg["angles"]]
                spd = int(max(1, min(100, msg.get("speed", speed))))
            except (ValueError, KeyError, TypeError):
                continue
            if len(cmd) >= 6:
                latest = (cmd, spd)
        if latest is None:
            continue
        cmd, spd = latest
        try:
            mc.send_angles(cmd[:6], spd, _async=True)
        except Exception:
            pass
        if len(cmd) > 6:  # optional 7th value is the gripper servo angle
            servo.angle = max(-90, min(90, cmd[6]))
    with clients_lock:
        if conn in clients:
            clients.remove(conn)
    try:
        conn.close()
    except OSError:
        pass


def accept_clients():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((TCP_HOST, TCP_PORT))
    srv.listen(5)
    print(f"TCP server listening on {TCP_HOST}:{TCP_PORT}")
    while running:
        try:
            conn, addr = srv.accept()
        except OSError:
            break
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with clients_lock:
            clients.append(conn)
        print(f"\nclient connected: {addr}")
        threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
    srv.close()


def streamer():
    """Background loop: stream joint angles to clients (send_coords disabled)."""
    global angles
    while running:
        if going_home:
            time.sleep(0.1)
            continue
        with mc_lock:
            try:
                angles = mc.get_angles()
            except Exception:
                angles = None
        if angles and isinstance(angles, (list, tuple)) and len(angles) == 6:
            broadcast(
                {
                    "angles": [round(a, 2) for a in angles] + [servo.angle],
                    "speed": speed,
                }
            )
        time.sleep(0.1)


threading.Thread(target=accept_clients, daemon=True).start()
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
    3: (5, +1),  # right stick X -> RZ
    4: (2, -1),  # right stick Y -> Z
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
Streaming joint angles over TCP at %d Hz. Current target: """ % (step, speed, TICK_HZ))

clock = pygame.time.Clock()
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                action = BUTTONS.get(event.button)
                if action == "grip_open":
                    servo.angle = -90
                elif action == "grip_close":
                    servo.angle = 90
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
        print("target: " + str([round(c, 1) for c in target]))
        print("angles: " + str([round(a, 1) for a in angles]) if angles else "")
        clock.tick(TICK_HZ)
finally:
    running = False
    time.sleep(0.1)  # let streamer fire one last command
    with clients_lock:
        for c in clients:
            try:
                c.close()
            except OSError:
                pass
        clients.clear()
    pygame.quit()
    print("\nbye.")
