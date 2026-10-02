import sys
import time
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from adapters.pixhawk_adapter import PixhawkAdapter


with open(BASE_DIR / "config" / "real.yaml", "r") as f:
    cfg = yaml.safe_load(f)


def request_message_interval(adapter, message_id, hz):
    interval_us = int(1_000_000 / hz)

    adapter.master.mav.command_long_send(
        adapter.master.target_system,
        adapter.master.target_component,
        511,  # MAV_CMD_SET_MESSAGE_INTERVAL
        0,
        message_id,
        interval_us,
        0, 0, 0, 0, 0
    )


def send_rc(adapter, roll=1500, pitch=1500, throttle=1000, yaw=1500):
    adapter.master.mav.rc_channels_override_send(
        adapter.master.target_system,
        adapter.master.target_component,
        roll, pitch, throttle, yaw,
        0, 0, 0, 0
    )


def read_any_rc(adapter, label):
    end = time.time() + 1.0

    while time.time() < end:
        msg = adapter.master.recv_match(
            type=["RC_CHANNELS", "RC_CHANNELS_RAW"],
            blocking=True,
            timeout=0.2
        )

        if msg is None:
            continue

        t = msg.get_type()

        if t == "RC_CHANNELS":
            print(
                f"{label} RC_CHANNELS: "
                f"ch1={msg.chan1_raw}, "
                f"ch2={msg.chan2_raw}, "
                f"ch3={msg.chan3_raw}, "
                f"ch4={msg.chan4_raw}"
            )
            return

        if t == "RC_CHANNELS_RAW":
            print(
                f"{label} RC_RAW: "
                f"ch1={msg.chan1_raw}, "
                f"ch2={msg.chan2_raw}, "
                f"ch3={msg.chan3_raw}, "
                f"ch4={msg.chan4_raw}"
            )
            return

    print(f"{label}: No RC message received")


def main():
    adapter = PixhawkAdapter(
        device=cfg["connection"]["device"],
        baud=cfg["connection"]["baud"],
    )

    adapter.connect()

    print("Requesting RC channel messages...")
    request_message_interval(adapter, 65, 5)   # RC_CHANNELS
    request_message_interval(adapter, 35, 5)   # RC_CHANNELS_RAW
    time.sleep(1)

    print("Neutral...")
    for _ in range(5):
        send_rc(adapter)
        read_any_rc(adapter, "neutral")
        time.sleep(0.1)

    print("Pitch forward...")
    for _ in range(5):
        send_rc(adapter, pitch=1600)
        read_any_rc(adapter, "pitch")
        time.sleep(0.1)

    print("Roll right...")
    for _ in range(5):
        send_rc(adapter, roll=1600)
        read_any_rc(adapter, "roll")
        time.sleep(0.1)

    print("Clearing override...")
    adapter.master.mav.rc_channels_override_send(
        adapter.master.target_system,
        adapter.master.target_component,
        0, 0, 0, 0, 0, 0, 0, 0
    )

    adapter.cleanup()


if __name__ == "__main__":
    main()
