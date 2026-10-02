import sys
import time
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from adapters.px4_offboard_adapter import PX4OffboardAdapter


with open(BASE_DIR / "config" / "real.yaml", "r") as f:
    cfg = yaml.safe_load(f)


def main():

    print("==========================================")
    print(" DRONOSAUR PX4 ZERO-SETPOINT TEST")
    print(" PROPS OFF / DISARMED / BENCH ONLY")
    print("==========================================")

    if cfg["safety"]["allow_arm"]:
        raise RuntimeError(
            "Safety stop: allow_arm must be false."
        )

    if cfg["safety"]["allow_takeoff"]:
        raise RuntimeError(
            "Safety stop: allow_takeoff must be false."
        )

    adapter = PX4OffboardAdapter(
        device=cfg["connection"]["device"],
        baud=cfg["connection"]["baud"],
        send_rate_hz=10
    )

    try:
        adapter.connect()

        print()
        print("Sending ZERO velocity setpoints for 10 seconds...")
        print("Aircraft must remain DISARMED.")

        adapter.start_zero_stream()

        for remaining in range(10, 0, -1):
            print(f"Zero stream: {remaining}s remaining")
            time.sleep(1)

        print()
        print("ZERO-SETPOINT TEST PASSED")

    finally:
        adapter.cleanup()


if __name__ == "__main__":
    main()
