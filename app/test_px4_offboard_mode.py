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
    print(" DRONOSAUR PX4 OFFBOARD MODE TRANSITION TEST")
    print(" PROPS OFF | DISARMED | ZERO SETPOINT ONLY")
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

        initial_mode = adapter.get_mode()

        print()
        print(f"Initial PX4 mode : {initial_mode}")
        print(f"Armed            : {adapter.is_armed()}")

        if adapter.is_armed():
            raise RuntimeError(
                "SAFETY STOP: vehicle is armed."
            )

        # ------------------------------------------
        # Begin zero-setpoint stream
        # ------------------------------------------

        adapter.set_zero()
        adapter.start_stream()

        print()
        print("Pre-streaming ZERO setpoints for 3 seconds...")

        for remaining in range(3, 0, -1):
            print(f"Pre-stream: {remaining}s")
            time.sleep(1)

        # ------------------------------------------
        # Request OFFBOARD
        # ------------------------------------------

        print()
        print("Requesting OFFBOARD mode...")

        adapter.set_mode("OFFBOARD")

        offboard_ok = adapter.wait_for_mode(
            "OFFBOARD",
            timeout=8
        )

        adapter.print_statustext()

        if not offboard_ok:
            print()
            print("OFFBOARD MODE TEST FAILED")
            print("Vehicle remains DISARMED.")
            return

        print()
        print("==============================================")
        print(" OFFBOARD MODE ENTERED SUCCESSFULLY")
        print("==============================================")
        print(f"Armed : {adapter.is_armed()}")
        print("Holding ZERO velocity setpoint for 5 seconds.")

        for remaining in range(5, 0, -1):
            print(f"OFFBOARD zero hold: {remaining}s")
            time.sleep(1)

        # ------------------------------------------
        # Return safely to previous mode
        # ------------------------------------------

        print()
        print(f"Returning to original mode: {initial_mode}")

        if initial_mode != "UNKNOWN":
            adapter.set_mode(initial_mode)

            restored = adapter.wait_for_mode(
                initial_mode,
                timeout=8
            )

            if not restored:
                print(
                    "WARNING: original mode was not confirmed."
                )

        adapter.print_statustext()

        print()
        print("==============================================")
        print(" OFFBOARD MODE TRANSITION TEST COMPLETE")
        print("==============================================")
        print(f"Armed : {adapter.is_armed()}")
        print("No arm command sent.")
        print("No takeoff command sent.")
        print("No non-zero velocity command sent.")

    finally:
        adapter.cleanup()


if __name__ == "__main__":
    main()
