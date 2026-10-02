import sys
import time
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from adapters.pixhawk_adapter import PixhawkAdapter


with open(BASE_DIR / "config" / "real.yaml", "r") as f:
    cfg = yaml.safe_load(f)


def main():
    adapter = PixhawkAdapter(
        device=cfg["connection"]["device"],
        baud=cfg["connection"]["baud"],
    )

    adapter.connect()

    print("Reading telemetry for 10 seconds...")
    start = time.time()

    while time.time() - start < 10:
        msg = adapter.master.recv_match(
            type=["HEARTBEAT", "ATTITUDE", "SYS_STATUS", "VFR_HUD"],
            blocking=True,
            timeout=1,
        )

        if msg is None:
            continue

        msg_type = msg.get_type()

        if msg_type == "HEARTBEAT":
            print("MODE:", adapter.get_mode())

        elif msg_type == "ATTITUDE":
            print(
                f"ATTITUDE roll={msg.roll:.2f}, pitch={msg.pitch:.2f}, yaw={msg.yaw:.2f}"
            )

        elif msg_type == "SYS_STATUS":
            print(
                f"BATTERY voltage={msg.voltage_battery / 1000:.2f}V, "
                f"remaining={msg.battery_remaining}%"
            )

        elif msg_type == "VFR_HUD":
            print(
                f"ALT={msg.alt:.2f}m, heading={msg.heading}, throttle={msg.throttle}%"
            )

    adapter.cleanup()
    print("Telemetry test finished")


if __name__ == "__main__":
    main()
