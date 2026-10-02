import sys
import time
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from adapters.pixhawk_adapter import PixhawkAdapter


with open(BASE_DIR / "config" / "real.yaml", "r") as f:
    cfg = yaml.safe_load(f)


def send_rc(adapter, roll=1500, pitch=1500, throttle=1000, yaw=1500):
    adapter.master.mav.rc_channels_override_send(
        adapter.master.target_system,
        adapter.master.target_component,
        roll,
        pitch,
        throttle,
        yaw,
        0, 0, 0, 0
    )


def main():
    adapter = PixhawkAdapter(
        device=cfg["connection"]["device"],
        baud=cfg["connection"]["baud"],
    )

    adapter.connect()
    adapter.print_status()

    print("RC override test starting.")
    print("PROPS MUST BE REMOVED.")
    print("Sending neutral channels...")

    for _ in range(20):
        send_rc(adapter)
        time.sleep(0.1)

    print("Pitch forward test...")
    for _ in range(20):
        send_rc(adapter, pitch=1600)
        time.sleep(0.1)

    print("Back to neutral...")
    for _ in range(20):
        send_rc(adapter)
        time.sleep(0.1)

    print("Roll right test...")
    for _ in range(20):
        send_rc(adapter, roll=1600)
        time.sleep(0.1)

    print("Back to neutral...")
    for _ in range(20):
        send_rc(adapter)
        time.sleep(0.1)

    print("Test done. Clearing override.")
    adapter.master.mav.rc_channels_override_send(
        adapter.master.target_system,
        adapter.master.target_component,
        0, 0, 0, 0, 0, 0, 0, 0
    )

    adapter.cleanup()


if __name__ == "__main__":
    main()
