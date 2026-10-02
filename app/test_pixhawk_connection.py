import sys
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))


from adapters.pixhawk_adapter import PixhawkAdapter


with open(BASE_DIR / "config" / "real.yaml", "r") as f:
    cfg = yaml.safe_load(f)


def main():
    print("======================================")
    print(" Dronosaur Pixhawk Connection Test")
    print(" BENCH READ-ONLY MODE")
    print("======================================")

    if not cfg["safety"]["bench_read_only"]:
        raise RuntimeError(
            "Safety stop: bench_read_only must be true."
        )

    if cfg["safety"]["allow_arm"]:
        raise RuntimeError(
            "Safety stop: allow_arm must be false."
        )

    if cfg["safety"]["allow_takeoff"]:
        raise RuntimeError(
            "Safety stop: allow_takeoff must be false."
        )

    adapter = PixhawkAdapter(
        device=cfg["connection"]["device"],
        baud=cfg["connection"]["baud"],
    )

    try:
        adapter.connect()
        adapter.print_status()

    finally:
        adapter.cleanup()


if __name__ == "__main__":
    main()
