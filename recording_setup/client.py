#!/usr/bin/env python3
"""TCP client that reads a list of 6 floats from a server."""

import argparse
import json
import socket


def parse_message(raw):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    angles = data.get("angles")
    speed = data.get("speed")
    if not isinstance(angles, list):
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in angles):
        return None
    if not isinstance(speed, (int, float)) or isinstance(speed, bool):
        return None
    return {"angles": [float(v) for v in angles], "speed": float(speed)}


def main():
    parser = argparse.ArgumentParser(description="Read a list of 6 floats over TCP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    with socket.create_connection((args.host, args.port)) as sock:
        print(f"Connected to {args.host}:{args.port}")
        with sock.makefile("r") as stream:
            for line in stream:
                raw = line.strip()
                if not raw:
                    continue
                values = parse_message(raw)
                if values is None:
                    print(f"Wrong format: {raw!r}")
                    continue
                print(values)


if __name__ == "__main__":
    main()
