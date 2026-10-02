from pymavlink import mavutil


class PixhawkAdapter:
    """
    Dronosaur Pixhawk adapter - Phase 1 BENCH READ-ONLY version.

    This version:
    - connects to the flight controller
    - receives MAVLink heartbeat/status
    - identifies firmware
    - reads flight mode

    It DOES NOT:
    - arm
    - disarm
    - change mode
    - send RC overrides
    - send velocity setpoints
    - send actuator commands
    """

    def __init__(self, device="/dev/ttyACM0", baud=57600):
        self.device = device
        self.baud = baud
        self.master = None
        self.heartbeat = None

    def connect(self):
        print(
            f"Connecting to flight controller on "
            f"{self.device} at {self.baud}..."
        )

        self.master = mavutil.mavlink_connection(
            self.device,
            baud=self.baud
        )

        print("Waiting for heartbeat...")

        self.heartbeat = self.master.wait_heartbeat(timeout=15)

        if self.heartbeat is None:
            raise TimeoutError(
                "No MAVLink heartbeat received within 15 seconds."
            )

        print(
            f"Heartbeat received from system "
            f"{self.master.target_system}, "
            f"component {self.master.target_component}"
        )

    def get_firmware_info(self):
        if self.heartbeat is None:
            return "UNKNOWN", "UNKNOWN"

        autopilot_id = self.heartbeat.autopilot
        vehicle_id = self.heartbeat.type

        autopilot_entry = mavutil.mavlink.enums[
            "MAV_AUTOPILOT"
        ].get(autopilot_id)

        vehicle_entry = mavutil.mavlink.enums[
            "MAV_TYPE"
        ].get(vehicle_id)

        autopilot_name = (
            autopilot_entry.name
            if autopilot_entry
            else f"UNKNOWN ({autopilot_id})"
        )

        vehicle_name = (
            vehicle_entry.name
            if vehicle_entry
            else f"UNKNOWN ({vehicle_id})"
        )

        return autopilot_name, vehicle_name

    def get_mode(self):
        msg = self.master.recv_match(
            type="HEARTBEAT",
            blocking=True,
            timeout=3
        )

        if msg is None:
            return "UNKNOWN"

        self.heartbeat = msg

        return mavutil.mode_string_v10(msg)

    def is_armed(self):
        if self.heartbeat is None:
            return False

        return bool(
            self.heartbeat.base_mode
            & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        )

    def print_status(self):
        autopilot, vehicle = self.get_firmware_info()
        mode = self.get_mode()

        print()
        print("====================================")
        print(" DRONOSAUR FLIGHT CONTROLLER STATUS")
        print("====================================")
        print(f"Autopilot     : {autopilot}")
        print(f"Vehicle type  : {vehicle}")
        print(f"Flight mode   : {mode}")
        print(f"Armed         : {self.is_armed()}")
        print("Control output: DISABLED")
        print("====================================")

    def cleanup(self):
        if self.master is not None:
            try:
                self.master.close()
            except Exception:
                pass

        print("Pixhawk adapter cleanup done.")
