import sys
from pathlib import Path

import yaml
from pymavlink import mavutil


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

with open(BASE_DIR / "config" / "real.yaml", "r") as f:
    cfg = yaml.safe_load(f)


DEVICE = cfg["connection"]["device"]
BAUD = cfg["connection"]["baud"]


def read_param(master, name, timeout=3):
    master.mav.param_request_read_send(
        master.target_system,
        master.target_component,
        name.encode("utf-8"),
        -1
    )

    while True:
        msg = master.recv_match(
            type="PARAM_VALUE",
            blocking=True,
            timeout=timeout
        )

        if msg is None:
            return None

        param_id = msg.param_id

        if isinstance(param_id, bytes):
            param_id = param_id.decode("utf-8")

        param_id = param_id.rstrip("\x00")

        if param_id == name:
            return msg.param_value


def main():
    print("==========================================")
    print(" DRONOSAUR PX4 OFFBOARD FAILSAFE PARAMS")
    print(" READ ONLY")
    print("==========================================")

    master = mavutil.mavlink_connection(
        DEVICE,
        baud=BAUD
    )

    print("Waiting for heartbeat...")
    master.wait_heartbeat(timeout=15)

    print("PX4 connected.")
    print()

    for name in [
        "COM_OF_LOSS_T",
        "COM_OBL_RC_ACT",
    ]:
        value = read_param(master, name)

        print(f"{name:16} : {value}")

    print()
    print("NO PARAMETERS CHANGED")


if __name__ == "__main__":
    main()
