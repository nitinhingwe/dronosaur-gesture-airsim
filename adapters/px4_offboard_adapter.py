import time
import threading
from pymavlink import mavutil


class PX4OffboardAdapter:
    """
    Dronosaur PX4 Offboard adapter.

    Current safety stage:
    - connect to PX4
    - stream velocity setpoints
    - change flight mode
    - NEVER arm automatically
    - NEVER take off automatically
    """

    def __init__(self, device, baud=921600, send_rate_hz=10):
        self.device = device
        self.baud = baud
        self.send_rate_hz = send_rate_hz

        self.master = None

        self.running = False
        self.worker = None
        self.lock = threading.Lock()

        # Body-frame velocity setpoint
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_rate = 0.0

    # --------------------------------------------------
    # CONNECTION
    # --------------------------------------------------

    def connect(self):
        print(
            f"Connecting PX4 Offboard adapter "
            f"to {self.device} @ {self.baud}..."
        )

        self.master = mavutil.mavlink_connection(
            self.device,
            baud=self.baud
        )

        print("Waiting for PX4 heartbeat...")

        heartbeat = self.master.wait_heartbeat(timeout=15)

        if heartbeat is None:
            raise RuntimeError("PX4 heartbeat timeout.")

        print(
            f"Heartbeat received: "
            f"system={self.master.target_system}, "
            f"component={self.master.target_component}"
        )

    # --------------------------------------------------
    # VEHICLE STATE
    # --------------------------------------------------

    def get_mode(self, timeout=2):
        msg = self.master.recv_match(
            type="HEARTBEAT",
            blocking=True,
            timeout=timeout
        )

        if msg is None:
            return "UNKNOWN"

        return mavutil.mode_string_v10(msg)

    def is_armed(self):
        return self.master.motors_armed()

    def wait_for_mode(self, expected_mode, timeout=8):
        print(f"Waiting for mode: {expected_mode}")

        start = time.time()

        while time.time() - start < timeout:
            mode = self.get_mode(timeout=1)

            if mode != "UNKNOWN":
                print(f"PX4 mode: {mode}")

            if mode == expected_mode:
                print(f"Mode confirmed: {expected_mode}")
                return True

        print(f"Mode {expected_mode} NOT confirmed.")
        return False

    # --------------------------------------------------
    # MODE CONTROL
    # --------------------------------------------------

    def set_mode(self, mode_name):
        if self.master is None:
            raise RuntimeError("PX4 is not connected.")

        mapping = self.master.mode_mapping()

        if not mapping:
            raise RuntimeError("PX4 mode mapping unavailable.")

        if mode_name not in mapping:
            raise RuntimeError(
                f"Mode {mode_name} unavailable. "
                f"Available modes: {list(mapping.keys())}"
            )

        print(f"Requesting PX4 mode: {mode_name}")

        # IMPORTANT:
        # Pass the PX4 mode NAME, not mapping[mode_name].
        # pymavlink handles the PX4 main/sub-mode fields internally.
        self.master.set_mode(mode_name)

    # --------------------------------------------------
    # SETPOINT CONTROL
    # --------------------------------------------------

    def set_velocity_body(
        self,
        forward=0.0,
        right=0.0,
        down=0.0,
        yaw_rate=0.0
    ):
        with self.lock:
            self.vx = float(forward)
            self.vy = float(right)
            self.vz = float(down)
            self.yaw_rate = float(yaw_rate)

    def set_zero(self):
        self.set_velocity_body(
            forward=0.0,
            right=0.0,
            down=0.0,
            yaw_rate=0.0
        )
        
    def set_command(self, command, velocity_cfg):
        """
        Convert Dronosaur semantic commands into PX4 BODY_NED
        velocity setpoints.

        BODY_NED:
            +X = forward
            +Y = right
            +Z = down

        Vehicle remains controlled only by the existing
        Offboard stream. This function does NOT arm or change mode.
        """

        import math

        forward = 0.0
        right = 0.0
        down = 0.0
        yaw_rate = 0.0

        if command == "FORWARD":
            forward = abs(float(velocity_cfg["forward"]))

        elif command == "BACKWARD":
            forward = -abs(float(velocity_cfg["backward"]))

        elif command == "RIGHT":
            right = abs(float(velocity_cfg["right"]))

        elif command == "LEFT":
            right = -abs(float(velocity_cfg["left"]))

        elif command == "UP":
            down = -abs(float(velocity_cfg["up"]))

        elif command == "DOWN":
            down = abs(float(velocity_cfg["down"]))

        elif command == "YAW_RIGHT":
            yaw_rate = math.radians(
                abs(float(velocity_cfg["yaw_rate"]))
            )

        elif command == "YAW_LEFT":
            yaw_rate = -math.radians(
                abs(float(velocity_cfg["yaw_rate"]))
            )

        elif command in ("HOVER", "NONE", "UNKNOWN"):
            pass

        else:
            print(
                f"Unknown command '{command}' -> HOVER"
            )

        self.set_velocity_body(
            forward=forward,
            right=right,
            down=down,
            yaw_rate=yaw_rate
        )

        return {
            "command": command,
            "vx": forward,
            "vy": right,
            "vz": down,
            "yaw_rate_rad_s": yaw_rate
        }

    def start_stream(self):
        if self.master is None:
            raise RuntimeError(
                "PX4 connection has not been established."
            )

        if self.running:
            return

        self.set_zero()

        print(
            f"Starting PX4 setpoint stream "
            f"at {self.send_rate_hz} Hz."
        )

        self.running = True

        self.worker = threading.Thread(
            target=self._stream_loop,
            daemon=True
        )

        self.worker.start()

    def _stream_loop(self):
        period = 1.0 / self.send_rate_hz

        while self.running:
            with self.lock:
                vx = self.vx
                vy = self.vy
                vz = self.vz
                yaw_rate = self.yaw_rate

            self._send_velocity_body(
                vx,
                vy,
                vz,
                yaw_rate
            )

            time.sleep(period)

    def _send_velocity_body(
        self,
        vx,
        vy,
        vz,
        yaw_rate
    ):
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        )

        self.master.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,

            self.master.target_system,
            self.master.target_component,

            mavutil.mavlink.MAV_FRAME_BODY_NED,

            type_mask,

            # Position ignored
            0.0,
            0.0,
            0.0,

            # Velocity
            vx,
            vy,
            vz,

            # Acceleration ignored
            0.0,
            0.0,
            0.0,

            # Yaw ignored
            0.0,

            # Yaw rate
            yaw_rate
        )

    # --------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------

    def print_statustext(self):
        while True:
            msg = self.master.recv_match(
                type="STATUSTEXT",
                blocking=False
            )

            if msg is None:
                break

            try:
                text = msg.text
            except Exception:
                text = str(msg)

            print(f"PX4: {text}")

    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------

    def stop_stream(self):
        if not self.running:
            return

        self.running = False

        if self.worker is not None:
            self.worker.join(timeout=2)

        self.worker = None

        print("PX4 setpoint stream stopped.")

    def cleanup(self):
        self.set_zero()
        self.stop_stream()
