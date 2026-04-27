"""
Gesture-Controlled Drone — AirSim Simulation (v2)
==================================================
Improvements over v1:
  - Angle-based finger detection (orientation-invariant)
  - Yaw left/right + descend (down) commands added
  - Closed fist now = EMERGENCY HOVER (safer default)
  - Auto-hover + auto-land failsafes on hand loss / camera freeze
  - Velocity ramping (smooth acceleration) for hardware-realism
  - Proportional throttle via thumb-index pinch distance
  - Threaded camera capture (decouples vision FPS from main loop)
  - Structured logging to CSV for post-flight analysis
  - Confidence + stability gating before any command is sent

Author: Jagavision
"""

import time
import math
import csv
import threading
import collections
from dataclasses import dataclass, field
from typing import Optional, List

import cv2
import numpy as np
import mediapipe as mp
import airsim


# =============================
# Configuration
# =============================
@dataclass
class Config:
    # Camera
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    # MediaPipe
    detection_confidence: float = 0.75
    tracking_confidence: float = 0.7
    model_complexity: int = 1

    # Gesture stability
    history_size: int = 7
    majority_threshold: int = 5  # of history_size

    # Failsafes
    no_hand_hover_seconds: float = 1.5     # hover after this long without a hand
    no_hand_land_seconds: float = 6.0      # auto-land after this long without a hand
    frame_stale_seconds: float = 1.0       # consider feed dead if no new frame

    # Movement (m/s) — target velocities; ramping handles acceleration
    v_forward: float = 1.0
    v_lateral: float = 0.8
    v_vertical: float = 0.6
    yaw_rate_deg: float = 35.0

    # Velocity ramping
    accel_per_sec: float = 2.0  # m/s^2 — how fast we approach target velocity
    yaw_accel_per_sec: float = 90.0  # deg/s^2

    # Command send rate
    send_interval: float = 0.10  # seconds between commands

    # Proportional throttle (pinch-based scaling)
    pinch_min: float = 0.04  # normalized distance considered "0%"
    pinch_max: float = 0.25  # normalized distance considered "100%"

    # Geofence (relative to takeoff point in NED frame)
    max_altitude_m: float = 15.0  # max climb height (NED z = -altitude)
    min_altitude_m: float = 1.0   # never descend below this AGL

    # Logging
    log_path: str = "flight_log.csv"


CFG = Config()


# =============================
# Threaded camera capture
# =============================
class CameraThread:
    """Reads frames in a background thread so MediaPipe never blocks capture."""

    def __init__(self, index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # DSHOW = Windows-friendly
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_time: float = 0.0
        self._running = True

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.time()
            else:
                time.sleep(0.005)

    def read(self):
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._frame_time

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# =============================
# Geometry helpers
# =============================
def angle_between(p1, p2, p3) -> float:
    """Angle at p2 formed by p1-p2-p3 (in degrees). Each p is (x,y) or (x,y,z)."""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def finger_extended(lm, mcp_i: int, pip_i: int, tip_i: int, thresh_deg: float = 160.0) -> bool:
    """A finger is 'extended' if the MCP-PIP-TIP angle is close to 180°.
    This is rotation-invariant — works regardless of hand orientation."""
    return angle_between(lm[mcp_i], lm[pip_i], lm[tip_i]) > thresh_deg


def thumb_extended(lm, handedness: str) -> bool:
    """Thumb uses CMC(1)-MCP(2)-IP(3) angle for extension check."""
    return angle_between(lm[1], lm[2], lm[3]) > 150.0


def pinch_distance(lm) -> float:
    """Normalized 3D distance between thumb tip (4) and index tip (8)."""
    p1, p2 = lm[4], lm[8]
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


# =============================
# Finger state + classifier
# =============================
def fingers_state(hand_landmarks, handedness: str) -> List[int]:
    """Returns [thumb, index, middle, ring, pinky] using angle-based detection."""
    lm = hand_landmarks.landmark
    return [
        int(thumb_extended(lm, handedness)),
        int(finger_extended(lm, 5, 6, 8)),
        int(finger_extended(lm, 9, 10, 12)),
        int(finger_extended(lm, 13, 14, 16)),
        int(finger_extended(lm, 17, 18, 20)),
    ]


def classify_gesture(fingers: List[int], hand_landmarks, handedness: str) -> str:
    """
    Gesture vocabulary (v2):
      - Open palm (5)        -> HOVER
      - Closed fist (0)      -> EMERGENCY (immediate hover, breaks any motion)
      - Victory (I+M)        -> FORWARD
      - Three fingers (I+M+R)-> BACKWARD
      - Index only           -> RIGHT
      - Pinky only           -> LEFT
      - Thumbs up            -> UP
      - Thumbs down (4 closed, thumb extended downward) -> DOWN
      - Index + Pinky ("rock") -> YAW_RIGHT
      - Index + Middle + Pinky -> YAW_LEFT
    """
    thumb, index, middle, ring, pinky = fingers
    lm = hand_landmarks.landmark

    # Emergency takes priority
    if fingers == [0, 0, 0, 0, 0]:
        return "EMERGENCY"

    if fingers == [1, 1, 1, 1, 1]:
        return "HOVER"

    if fingers == [0, 1, 1, 0, 0]:
        return "FORWARD"

    if fingers == [0, 1, 1, 1, 0]:
        return "BACKWARD"

    if fingers == [0, 1, 0, 0, 0]:
        return "RIGHT"

    if fingers == [0, 0, 0, 0, 1]:
        return "LEFT"

    # Thumb up vs down — use thumb tip Y vs wrist Y
    if thumb == 1 and index == 0 and middle == 0 and ring == 0 and pinky == 0:
        if lm[4].y < lm[0].y - 0.05:
            return "UP"
        elif lm[4].y > lm[0].y + 0.05:
            return "DOWN"

    # Yaw gestures
    if fingers == [0, 1, 0, 0, 1]:  # rock sign
        return "YAW_RIGHT"

    if fingers == [0, 1, 1, 0, 1]:
        return "YAW_LEFT"

    return "UNKNOWN"


# =============================
# Stability filter
# =============================
class GestureStabilizer:
    def __init__(self, size: int, threshold: int):
        self.history = collections.deque(maxlen=size)
        self.threshold = threshold

    def update(self, gesture: str) -> str:
        self.history.append(gesture)
        if len(self.history) < self.history.maxlen:
            return "NONE"
        most_common, count = collections.Counter(self.history).most_common(1)[0]
        return most_common if count >= self.threshold else "NONE"

    def clear(self):
        self.history.clear()


# =============================
# Drone controller with ramping + geofence
# =============================
class DroneController:
    def __init__(self, client: airsim.MultirotorClient, cfg: Config):
        self.client = client
        self.cfg = cfg

        # Current and target velocities (body frame)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_rate = 0.0

        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_vz = 0.0
        self.target_yaw_rate = 0.0

        self._last_send = 0.0
        self._last_step = time.time()

    def set_command(self, command: str, throttle_scale: float = 1.0):
        """Set target velocities based on a high-level command + scale (0..1)."""
        s = max(0.2, min(throttle_scale, 1.0))  # clamp; never less than 20%

        # Reset all targets
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_vz = 0.0
        self.target_yaw_rate = 0.0

        if command == "FORWARD":
            self.target_vx = self.cfg.v_forward * s
        elif command == "BACKWARD":
            self.target_vx = -self.cfg.v_forward * 0.7 * s  # slower backward
        elif command == "RIGHT":
            self.target_vy = self.cfg.v_lateral * s
        elif command == "LEFT":
            self.target_vy = -self.cfg.v_lateral * s
        elif command == "UP":
            self.target_vz = -self.cfg.v_vertical * s  # NED: up is negative
        elif command == "DOWN":
            self.target_vz = self.cfg.v_vertical * s
        elif command == "YAW_RIGHT":
            self.target_yaw_rate = self.cfg.yaw_rate_deg * s
        elif command == "YAW_LEFT":
            self.target_yaw_rate = -self.cfg.yaw_rate_deg * s
        # HOVER and EMERGENCY both = all zeros

    def _ramp(self, current: float, target: float, accel: float, dt: float) -> float:
        """Move 'current' toward 'target' at most by accel*dt."""
        delta = target - current
        max_step = accel * dt
        if abs(delta) <= max_step:
            return target
        return current + math.copysign(max_step, delta)

    def _check_geofence(self):
        """Block ascent above max altitude or descent below min altitude."""
        try:
            state = self.client.getMultirotorState()
            altitude = -state.kinematics_estimated.position.z_val  # NED -> AGL
            if altitude >= self.cfg.max_altitude_m and self.target_vz < 0:
                self.target_vz = 0.0
            if altitude <= self.cfg.min_altitude_m and self.target_vz > 0:
                self.target_vz = 0.0
            return altitude
        except Exception:
            return None

    def step(self):
        """Called every loop iteration — ramps velocities and sends to AirSim."""
        now = time.time()
        dt = now - self._last_step
        self._last_step = now

        altitude = self._check_geofence()

        # Ramp toward targets
        self.vx = self._ramp(self.vx, self.target_vx, self.cfg.accel_per_sec, dt)
        self.vy = self._ramp(self.vy, self.target_vy, self.cfg.accel_per_sec, dt)
        self.vz = self._ramp(self.vz, self.target_vz, self.cfg.accel_per_sec, dt)
        self.yaw_rate = self._ramp(self.yaw_rate, self.target_yaw_rate,
                                    self.cfg.yaw_accel_per_sec, dt)

        # Rate-limit actual sends to AirSim
        if now - self._last_send < self.cfg.send_interval:
            return altitude
        self._last_send = now

        try:
            self.client.moveByVelocityBodyFrameAsync(
                self.vx, self.vy, self.vz,
                duration=self.cfg.send_interval * 2,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=self.yaw_rate),
            )
        except Exception as e:
            print(f"[Controller] Send failed: {e}")

        return altitude

    def emergency_hover(self):
        """Instantly zero everything and hover."""
        self.target_vx = self.target_vy = self.target_vz = self.target_yaw_rate = 0.0
        self.vx = self.vy = self.vz = self.yaw_rate = 0.0
        try:
            self.client.hoverAsync()
        except Exception:
            pass

    def land(self):
        try:
            self.client.landAsync().join()
        except Exception as e:
            print(f"[Controller] Land failed: {e}")


# =============================
# Logger
# =============================
class FlightLogger:
    def __init__(self, path: str):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "timestamp", "raw_gesture", "stable_gesture", "command",
            "throttle", "vx", "vy", "vz", "yaw_rate", "altitude", "fps",
        ])

    def log(self, raw, stable, cmd, throttle, vx, vy, vz, yaw, alt, fps):
        self.writer.writerow([
            f"{time.time():.3f}", raw, stable, cmd,
            f"{throttle:.2f}", f"{vx:.2f}", f"{vy:.2f}", f"{vz:.2f}",
            f"{yaw:.1f}", f"{alt:.2f}" if alt is not None else "",
            f"{fps:.1f}",
        ])

    def close(self):
        self.file.close()


# =============================
# Main
# =============================
def main():
    # AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(0.5)
    client.moveByVelocityBodyFrameAsync(0, 0, -1.0, 1.5).join()

    # MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=CFG.model_complexity,
        min_detection_confidence=CFG.detection_confidence,
        min_tracking_confidence=CFG.tracking_confidence,
    )

    # Camera (threaded)
    camera = CameraThread(CFG.camera_index, CFG.frame_width, CFG.frame_height)

    # Helpers
    stabilizer = GestureStabilizer(CFG.history_size, CFG.majority_threshold)
    controller = DroneController(client, CFG)
    logger = FlightLogger(CFG.log_path)

    cv2.namedWindow("Dronosaur Labs | Gesture Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dronosaur Labs | Gesture Control", 550, 375)

    last_hand_seen = time.time()
    auto_landed = False
    fps_history = collections.deque(maxlen=30)
    last_loop = time.time()

    try:
        while True:
            loop_start = time.time()
            frame, frame_ts = camera.read()

            # Failsafe: dead camera feed
            if frame is None or (time.time() - frame_ts) > CFG.frame_stale_seconds:
                if not auto_landed:
                    print("[Failsafe] Camera feed stale — emergency hover")
                    controller.emergency_hover()
                cv2.waitKey(50)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            raw_gesture = "No Hand"
            stable_gesture = "NONE"
            command = "HOVER"
            throttle = 1.0
            fingers = None
            hand_label = "—"

            if results.multi_hand_landmarks and results.multi_handedness:
                last_hand_seen = time.time()
                hand_landmarks = results.multi_hand_landmarks[0]
                hand_label = results.multi_handedness[0].classification[0].label

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fingers = fingers_state(hand_landmarks, hand_label)
                raw_gesture = classify_gesture(fingers, hand_landmarks, hand_label)
                stable_gesture = stabilizer.update(raw_gesture)

                # Proportional throttle from pinch (only meaningful for movement commands)
                pinch = pinch_distance(hand_landmarks.landmark)
                throttle = (pinch - CFG.pinch_min) / (CFG.pinch_max - CFG.pinch_min)
                throttle = max(0.2, min(throttle, 1.0))

                if stable_gesture == "EMERGENCY":
                    controller.emergency_hover()
                    command = "EMERGENCY"
                elif stable_gesture not in ("NONE", "UNKNOWN"):
                    command = stable_gesture
                    controller.set_command(command, throttle)
            else:
                stabilizer.clear()
                time_since_hand = time.time() - last_hand_seen

                if time_since_hand > CFG.no_hand_land_seconds and not auto_landed:
                    print("[Failsafe] No hand for >6s — auto-landing")
                    controller.land()
                    auto_landed = True
                elif time_since_hand > CFG.no_hand_hover_seconds:
                    controller.set_command("HOVER")

            altitude = controller.step()

            # FPS
            now = time.time()
            fps_history.append(1.0 / max(now - last_loop, 1e-3))
            last_loop = now
            fps = sum(fps_history) / len(fps_history)

            # Log
            logger.log(raw_gesture, stable_gesture, command, throttle,
                       controller.vx, controller.vy, controller.vz,
                       controller.yaw_rate, altitude, fps)

            # ===== HUD =====
            def put(text, y, color=(0, 255, 0), scale=0.7):
                cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, color, 2, cv2.LINE_AA)

            put(f"FPS: {fps:.1f}", 30, (200, 200, 200))
            put(f"Hand: {hand_label}  Fingers: {fingers}", 60, (255, 255, 0))
            put(f"Raw: {raw_gesture}", 90, (0, 255, 0))
            put(f"Stable: {stable_gesture}", 120, (0, 200, 255))
            put(f"Command: {command}  Throttle: {throttle:.2f}", 150, (255, 200, 0))
            put(f"Velocity: vx={controller.vx:+.2f} vy={controller.vy:+.2f} "
                f"vz={controller.vz:+.2f} yaw={controller.yaw_rate:+.1f}",
                180, (180, 255, 180), 0.6)
            if altitude is not None:
                put(f"Altitude: {altitude:.2f} m", 210, (200, 200, 255))
            if auto_landed:
                put("AUTO-LANDED", 240, (0, 0, 255), 0.9)
            put("q=quit | l=land | r=rearm", frame.shape[0] - 25,
                (0, 200, 255), 0.6)

            cv2.imshow("Dronosaur Labs | Gesture Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l"):
                print("Manual land")
                controller.land()
                auto_landed = True
            elif key == ord("r") and auto_landed:
                print("Re-arming + takeoff")
                client.armDisarm(True)
                client.takeoffAsync().join()
                auto_landed = False
                last_hand_seen = time.time()

    finally:
        print("Cleaning up...")
        camera.stop()
        hands.close()
        cv2.destroyAllWindows()
        logger.close()
        try:
            controller.emergency_hover()
            time.sleep(0.5)
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
        except Exception:
            pass
        print(f"Done. Flight log saved to: {CFG.log_path}")


if __name__ == "__main__":
    main()
