import sys
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
    adapter.print_status()

    target_mode = cfg["control"]["mode"]
    print(f"Testing safe mode change to: {target_mode}")

    adapter.set_mode(target_mode)
    adapter.wait_for_mode(target_mode)

    adapter.print_status()
    adapter.cleanup()


if __name__ == "__main__":
    main()
