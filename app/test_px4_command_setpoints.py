import sys
import time
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from adapters.px4_offboard_adapter import PX4OffboardAdapter


with open(BASE_DIR / "config" / "real.yaml", "r") as f:
    cfg = yaml.safe_load(f)


TEST_COMMANDS = [
    "HOVER",
    "FORWARD",
    "HOVER",
    "BACKWARD",
    "HOVER",
    "RIGHT",
    "HOVER",
    "LEFT",
    "HOVER",
    "UP",
    "HOVER",
    "DOWN",
    "HOVER",
    "YAW_RIGHT",
    "HOVER",
    "YAW_LEFT",
    "HOVER",
]


def main():

    print("==============================================")
    print(" DRONOSAUR PX4 COMMAND SETPOINT BENCH TEST")
    print(" PROPS OFF | DISARMED | NO MODE CHANGE")
    print("==============================================")

    if cfg["safety"]["allow_arm"]:
        raise RuntimeError(
            "SAFETY STOP: allow_arm must be false."
        )

    if cfg["safety"]["allow_takeoff"]:
        raise RuntimeError(
            "SAFETY STOP: allow_takeoff must be false."
        )

    adapter = PX4OffboardAdapter(
        device=cfg["connection"]["device"],
        baud=cfg["connection"]["baud"],
        send_rate_hz=10
    )

    try:

        adapter.connect()

        print()
        print(f"PX4 armed: {adapter.is_armed()}")

        if adapter.is_armed():
            raise RuntimeError(
                "SAFETY STOP: vehicle is armed."
            )

        print()
        print("Starting setpoint stream...")
        print("Vehicle remains DISARMED.")
        print("No OFFBOARD mode request will be made.")
        print()

        adapter.set_zero()
        adapter.start_stream()

        time.sleep(2)

        for command in TEST_COMMANDS:

            if adapter.is_armed():
                raise RuntimeError(
                    "SAFETY STOP: unexpected armed state."
                )

            result = adapter.set_command(
                command,
                cfg["velocity"]
            )

            print(
                f"{command:10} -> "
                f"vx={result['vx']:+.2f}  "
                f"vy={result['vy']:+.2f}  "
                f"vz={result['vz']:+.2f}  "
                f"yaw={result['yaw_rate_rad_s']:+.3f} rad/s"
            )

            time.sleep(2)

        adapter.set_zero()

        print()
        print("==============================================")
        print(" COMMAND SETPOINT TEST COMPLETE")
        print("==============================================")
        print(f"PX4 armed: {adapter.is_armed()}")
        print("Final command: HOVER / ZERO SETPOINT")

    finally:
        adapter.cleanup()


if __name__ == "__main__":
    main()
