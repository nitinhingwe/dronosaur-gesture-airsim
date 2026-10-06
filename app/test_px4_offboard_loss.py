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

    print("==============================================")
    print(" DRONOSAUR PX4 OFFBOARD-LOSS FAILSAFE TEST")
    print(" PROPS OFF | DISARMED | BENCH ONLY")
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

    initial_mode = None

    try:
        adapter.connect()

        initial_mode = adapter.get_mode()

        print()
        print(f"Initial mode : {initial_mode}")
        print(f"Armed        : {adapter.is_armed()}")

        if adapter.is_armed():
            raise RuntimeError(
                "SAFETY STOP: vehicle is armed."
            )

        # ---------------------------------------
        # Establish healthy Offboard stream
        # ---------------------------------------

        adapter.set_zero()
        adapter.start_stream()

        print()
        print("Pre-streaming ZERO setpoints for 3 seconds...")

        time.sleep(3)

        # ---------------------------------------
        # Enter OFFBOARD
        # ---------------------------------------

        print()
        print("Requesting OFFBOARD...")

        adapter.set_mode("OFFBOARD")

        if not adapter.wait_for_mode(
            "OFFBOARD",
            timeout=8
        ):
            raise RuntimeError(
                "Could not enter OFFBOARD."
            )

        print()
        print("OFFBOARD confirmed.")
        print("Holding healthy stream for 3 seconds...")

        time.sleep(3)

        # ---------------------------------------
        # DELIBERATE LINK/SETPOINT LOSS
        # ---------------------------------------

        print()
        print("==============================================")
        print(" DELIBERATELY STOPPING OFFBOARD SETPOINTS")
        print("==============================================")

        adapter.stop_stream()

        loss_start = time.monotonic()

        previous_mode = "OFFBOARD"

        for _ in range(15):

            mode = adapter.get_mode(timeout=1)

            elapsed = time.monotonic() - loss_start

            print(
                f"T+{elapsed:5.2f}s | "
                f"PX4 mode: {mode} | "
                f"Armed: {adapter.is_armed()}"
            )

            adapter.print_statustext()

            if (
                mode != "UNKNOWN"
                and mode != "OFFBOARD"
            ):
                print()
                print("==============================================")
                print(" OFFBOARD-LOSS FAILSAFE DETECTED")
                print("==============================================")
                print(f"PX4 left OFFBOARD after ~{elapsed:.2f} sec")
                print(f"Fallback mode: {mode}")
                print(f"Armed        : {adapter.is_armed()}")
                print()
                print("FAILSAFE TEST PASSED")
                return

            previous_mode = mode

        print()
        print("==============================================")
        print(" FAILSAFE NOT OBSERVED WITHIN TEST WINDOW")
        print("==============================================")

    finally:

        # Do NOT restart Offboard stream here.
        # Vehicle must remain disarmed.

        print()
        print(f"Final armed state: {adapter.is_armed()}")

        if adapter.running:
            adapter.stop_stream()


if __name__ == "__main__":
    main()
