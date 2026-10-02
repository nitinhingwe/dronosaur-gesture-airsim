import sys
import time
import math
from pathlib import Path

import yaml
from pymavlink import mavutil


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))


with open(BASE_DIR / "config" / "real.yaml", "r") as f:
    cfg = yaml.safe_load(f)


DEVICE = cfg["connection"]["device"]
BAUD = cfg["connection"]["baud"]


def safe_battery_voltage(msg):
    if msg is None:
        return "N/A"

    if msg.voltage_battery in (0, 65535):
        return "N/A"

    return f"{msg.voltage_battery / 1000.0:.2f} V"


def safe_battery_remaining(msg):
    if msg is None:
        return "N/A"

    if msg.battery_remaining < 0:
        return "N/A"

    return f"{msg.battery_remaining}%"


def main():
    print("============================================")
    print(" DRONOSAUR PIXHAWK READ-ONLY TELEMETRY TEST")
    print("============================================")
    print(f"Device : {DEVICE}")
    print(f"Baud   : {BAUD}")
    print()
    print("NO ARMING")
    print("NO MODE CHANGES")
    print("NO RC OVERRIDES")
    print("NO OFFBOARD")
    print("NO FLIGHT COMMANDS")
    print("============================================")
    print()

    master = mavutil.mavlink_connection(
        DEVICE,
        baud=BAUD,
    )

    print("Waiting for PX4 heartbeat...")

    heartbeat = master.wait_heartbeat(timeout=15)

    if heartbeat is None:
        raise TimeoutError("No heartbeat received within 15 seconds.")

    print(
        f"CONNECTED: system={master.target_system}, "
        f"component={master.target_component}"
    )

    latest = {}

    last_heartbeat_time = time.monotonic()
    last_print_time = 0.0

    try:
        while True:

            msg = master.recv_match(
                blocking=True,
                timeout=1,
            )

            now = time.monotonic()

            if msg is not None:

                msg_type = msg.get_type()

                if msg_type != "BAD_DATA":
                    latest[msg_type] = msg

                if msg_type == "HEARTBEAT":
                    last_heartbeat_time = now

            heartbeat_age = now - last_heartbeat_time
            connected = heartbeat_age < 3.0

            if now - last_print_time < 1.0:
                continue

            last_print_time = now

            print()
            print("--------------------------------------------")
            print(f"MAVLink connected : {connected}")
            print(f"Heartbeat age     : {heartbeat_age:.2f} sec")

            # HEARTBEAT
            hb = latest.get("HEARTBEAT")

            if hb is not None:

                armed = bool(
                    hb.base_mode
                    & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )

                mode = mavutil.mode_string_v10(hb)

                print(f"Armed             : {armed}")
                print(f"PX4 mode          : {mode}")

            else:
                print("Armed             : N/A")
                print("PX4 mode          : N/A")

            # BATTERY
            sys_status = latest.get("SYS_STATUS")

            print(
                f"Battery voltage   : "
                f"{safe_battery_voltage(sys_status)}"
            )

            print(
                f"Battery remaining : "
                f"{safe_battery_remaining(sys_status)}"
            )

            # GPS
            gps = latest.get("GPS_RAW_INT")

            if gps is not None:
                print(f"GPS fix type      : {gps.fix_type}")
                print(f"Satellites        : {gps.satellites_visible}")
            else:
                print("GPS fix type      : N/A")
                print("Satellites        : N/A")

            # GLOBAL POSITION
            global_pos = latest.get("GLOBAL_POSITION_INT")

            if global_pos is not None:
                print(
                    f"Latitude          : "
                    f"{global_pos.lat / 1e7:.7f}"
                )

                print(
                    f"Longitude         : "
                    f"{global_pos.lon / 1e7:.7f}"
                )

                print(
                    f"Relative altitude : "
                    f"{global_pos.relative_alt / 1000.0:.2f} m"
                )

                print(
                    "Global velocity    : "
                    f"vx={global_pos.vx / 100.0:.2f}, "
                    f"vy={global_pos.vy / 100.0:.2f}, "
                    f"vz={global_pos.vz / 100.0:.2f} m/s"
                )

            # ATTITUDE
            attitude = latest.get("ATTITUDE")

            if attitude is not None:
                print(
                    "Attitude           : "
                    f"roll={math.degrees(attitude.roll):.1f} deg, "
                    f"pitch={math.degrees(attitude.pitch):.1f} deg, "
                    f"yaw={math.degrees(attitude.yaw):.1f} deg"
                )

            # LOCAL POSITION
            local = latest.get("LOCAL_POSITION_NED")

            if local is not None:
                print(
                    "Local velocity     : "
                    f"vx={local.vx:.2f}, "
                    f"vy={local.vy:.2f}, "
                    f"vz={local.vz:.2f} m/s"
                )

    except KeyboardInterrupt:
        print()
        print("Read-only telemetry test stopped.")

    finally:
        try:
            master.close()
        except Exception:
            pass

        print("MAVLink connection closed.")


if __name__ == "__main__":
    main()