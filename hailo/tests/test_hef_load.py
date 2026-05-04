from pathlib import Path
from hailo_platform import HEF

MODEL_DIR = Path(__file__).resolve().parents[1] / "hef_models"

hef_files = list(MODEL_DIR.glob("*.hef"))

if not hef_files:
    print("No .hef model found in hailo/hef_models")
    raise SystemExit(1)

hef_path = hef_files[0]
print(f"Loading HEF: {hef_path}")

hef = HEF(str(hef_path))

print("HEF loaded successfully")
print("Input infos:")
for info in hef.get_input_vstream_infos():
    print(info)

print("Output infos:")
for info in hef.get_output_vstream_infos():
    print(info)
