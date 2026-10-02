import sys
import time
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from adapters.pixhawk_adapter import PixhawkAdapter


with open(BASE_DIR / "config" / "real.yaml", "r") as f:
    cfg = yaml.safe_load(f)


COMMANDS = [
    "HOVER",
    "FORWARD",
    "BACKWARD",
    "RIGHT",
    "LEFT",
    "YAW_RIGHT",
    "YAW_LEFT",
    "UP",
    "DOWN",
]


def main():
    adapter = PixhawkAdapter(
        device=cfg["connection"]["device"],
        baud=cfg["connection"]["baud"],
    )

    adapter.connect()
    adapter.print_status()

    print("COMMAND MAPPING TEST")
    print("PROPS MUST BE REMOVED. RC TRANSMITTER MUST BE ON.")
    print("No arming. No takeoff. RC override only.")

    for command in COMMANDS:
        print(f"\nSending command: {command}")

        for _ in range(15):
            adapter.send_command(command)
            time.sleep(0.1)

        print("Back to HOVER/neutral")
        for _ in range(10):
            adapter.send_command("HOVER")
            time.sleep(0.1)

    print("\nClearing RC override...")
    adapter.clear_rc_override()
    adapter.cleanup()

    print("Command mapping test finished")


if __name__ == "__main__":
    main()
