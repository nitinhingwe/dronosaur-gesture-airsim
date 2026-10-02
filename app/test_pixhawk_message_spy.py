import sys
import time
from pathlib import Path
import yaml
from collections import Counter

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

    print("Requesting all MAVLink data streams...")

    for stream_id in range(0, 13):
        adapter.master.mav.request_data_stream_send(
            adapter.master.target_system,
            adapter.master.target_component,
            stream_id,
            5,
            1
        )

    counts = Counter()
    start = time.time()

    print("Listening for 15 seconds...")

    while time.time() - start < 15:
        msg = adapter.master.recv_match(blocking=True, timeout=1)

        if msg is None:
            continue

        msg_type = msg.get_type()
        counts[msg_type] += 1

        if msg_type in ["RC_CHANNELS", "RC_CHANNELS_RAW", "SERVO_OUTPUT_RAW", "HEARTBEAT", "ATTITUDE", "VFR_HUD"]:
            print(msg)

    print("\nMessage types received:")
    for msg_type, count in counts.most_common():
        print(msg_type, count)

    adapter.cleanup()


if __name__ == "__main__":
    main()
