"""
Gesture-Controlled Drone — AirSim Simulation (v3)
==================================================
Changes vs v2:
  - S-curve (smoothstep) velocity ramping — zero jerk at transitions
  - Lower acceleration values — softer pitch on start/stop
  - Slower yaw rate (18°/s) for precise angular targeting
  - Longer command duration (0.5s) — AirSim interpolates smoothly
    instead of PID-chasing new setpoints every 100ms
"""

import time
import math
import csv
import threading
import collections
from dataclasses import dataclass
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
    majority_threshold: int = 5

    # Failsafes
    no_hand_hover_seconds: float = 1.5
    no_hand_land_seconds: float = 6.0
    frame_stale_seconds: float = 1.0

    # Movement (m/s)
    v_forward: float = 1.0
    v_lateral: float = 0.8
    v_vertical: float = 0.5
    yaw_rate_deg: float = 18.0          # SLOWER for precise turns

    # Velocity ramping — LOWERED for smoothness
    accel_per_sec: float = 0.9          # was 2.0
    yaw_accel_per_sec: float = 40.0     # was 90

    # Command send rate
    send_interval: float = 0.08
    command_duration: float = 0.5       # NEW: longer hold = smoother motion

    # Proportional throttle
    pinch_min: float = 0.04
    pinch_max: float = 0.25

    # Geofence
    max_altitude_m: float = 15.0
    min_altitude_m: float = 1.0

    # Logging
    log_path: str = "flight_log.csv"


CFG = Config()


# =============================
# Threaded camera capture
# =============================
class CameraThread:
    def __init__(self, index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
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
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def finger_extended(lm, mcp_i, pip_i, tip_i, thresh_deg=160.0):
    return angle_between(lm[mcp_i], lm[pip_i], lm[tip_i]) > thresh_deg


def thumb_extended(lm, handedness):
    return angle_between(lm[1], lm[2], lm[3]) > 150.0


def pinch_distance(lm):
    p1, p2 = lm[4], lm[8]
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


# =============================
# Finger state + classifier
# =============================
def fingers_state(hand_landmarks, handedness: str) -> List[int]:
    lm = hand_landmarks.landmark
    return [
        int(thumb_extended(lm, handedness)),
        int(finger_extended(lm, 5, 6, 8)),
        int(finger_extended(lm, 9, 10, 12)),
        int(finger_extended(lm, 13, 14, 16)),
        int(finger_extended(lm, 17, 18, 20)),
    ]


def classify_gesture(fingers, hand_landmarks, handedness: str) -> str:
    thumb, index, middle, ring, pinky = fingers
    lm = hand_landmarks.landmark

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

    if thumb == 1 and index == 0 and middle == 0 and ring == 0 and pinky == 0:
        if lm[4].y < lm[0].y - 0.05:
            return "UP"
        elif lm[4].y > lm[0].y + 0.05:
            return "DOWN"

    if fingers == [0, 1, 0, 0, 1]:
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
# Drone controller with S-curve ramping
# =============================
class DroneController:
    def __init__(self, client: airsim.MultirotorClient, cfg: Config):
        self.client = client
        self.cfg = cfg

        self.vx = self.vy = self.vz = self.yaw_rate = 0.0
        self.target_vx = self.target_vy = self.target_vz = self.target_yaw_rate = 0.0

        self._last_send = 0.0
        self._last_step = time.time()

    def set_command(self, command: str, throttle_scale: float = 1.0):
        s = max(0.2, min(throttle_scale, 1.0))
        self.target_vx = self.target_vy = self.target_vz = self.target_yaw_rate = 0.0

        if command == "FORWARD":
            self.target_vx = self.cfg.v_forward * s
        elif command == "BACKWARD":
            self.target_vx = -self.cfg.v_forward * 0.7 * s
        elif command == "RIGHT":
            self.target_vy = self.cfg.v_lateral * s
        elif command == "LEFT":
            self.target_vy = -self.cfg.v_lateral * s
        elif command == "UP":
            self.target_vz = -self.cfg.v_vertical * s
        elif command == "DOWN":
            self.target_vz = self.cfg.v_vertical * s
        elif command == "YAW_RIGHT":
            self.target_yaw_rate = self.cfg.yaw_rate_deg * s
        elif command == "YAW_LEFT":
            self.target_yaw_rate = -self.cfg.yaw_rate_deg * s

    def _ramp(self, current: float, target: float, accel: float, dt: float) -> float:
        """
        S-curve (smoothstep) ramping — zero jerk at transitions.
        Linear ramping causes a discontinuous derivative at start/end
        (instant acceleration sign flip), which the drone's pitch response
        amplifies into visible shake. Smoothstep eases in and out.
        """
        delta = target - current
        if abs(delta) < 1e-4:
            return target

        max_step = accel * dt

        # Normalize progress toward target
        ref_distance = max(abs(target), 0.3)
        progress = min(abs(delta) / ref_distance, 1.0)

        # Smoothstep: 3x² - 2x³
        # Slow near target (small delta), full speed far from target
        ease = progress * progress * (3.0 - 2.0 * progress)
        step = max_step * (0.3 + 0.7 * ease)  # floor at 30% so we don't stall

        if abs(delta) <= step:
            return target
        return current + math.copysign(step, delta)

    def _check_geofence(self):
        try:
            state = self.client.getMultirotorState()
            altitude = -state.kinematics_estimated.position.z_val
            if altitude >= self.cfg.max_altitude_m and self.target_vz < 0:
                self.target_vz = 0.0
            if altitude <= self.cfg.min_altitude_m and self.target_vz > 0:
                self.target_vz = 0.0
            return altitude
        except Exception:
            return None

    def step(self):
        now = time.time()
        dt = now - self._last_step
        self._last_step = now

        altitude = self._check_geofence()

        self.vx = self._ramp(self.vx, self.target_vx, self.cfg.accel_per_sec, dt)
        self.vy = self._ramp(self.vy, self.target_vy, self.cfg.accel_per_sec, dt)
        self.vz = self._ramp(self.vz, self.target_vz, self.cfg.accel_per_sec, dt)
        self.yaw_rate = self._ramp(self.yaw_rate, self.target_yaw_rate,
                                    self.cfg.yaw_accel_per_sec, dt)

        if now - self._last_send < self.cfg.send_interval:
            return altitude
        self._last_send = now

        try:
            # Longer duration = smoother AirSim interpolation, less PID chasing
            self.client.moveByVelocityBodyFrameAsync(
                self.vx, self.vy, self.vz,
                duration=self.cfg.command_duration,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=self.yaw_rate),
            )
        except Exception as e:
            print(f"[Controller] Send failed: {e}")

        return altitude

    def emergency_hover(self):
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
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(0.5)
    client.moveByVelocityBodyFrameAsync(0, 0, -1.0, 1.5).join()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=CFG.model_complexity,
        min_detection_confidence=CFG.detection_confidence,
        min_tracking_confidence=CFG.tracking_confidence,
    )

    camera = CameraThread(CFG.camera_index, CFG.frame_width, CFG.frame_height)
    stabilizer = GestureStabilizer(CFG.history_size, CFG.majority_threshold)
    controller = DroneController(client, CFG)
    logger = FlightLogger(CFG.log_path)

    cv2.namedWindow("Gesture Control v3", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gesture Control v3", 1100, 750)

    last_hand_seen = time.time()
    auto_landed = False
    fps_history = collections.deque(maxlen=30)
    last_loop = time.time()

    try:
        while True:
            frame, frame_ts = camera.read()

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

            now = time.time()
            fps_history.append(1.0 / max(now - last_loop, 1e-3))
            last_loop = now
            fps = sum(fps_history) / len(fps_history)

            logger.log(raw_gesture, stable_gesture, command, throttle,
                       controller.vx, controller.vy, controller.vz,
                       controller.yaw_rate, altitude, fps)

            def put(text, y, color=(0, 255, 0), scale=0.7):
                cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, color, 2, cv2.LINE_AA)

            put(f"FPS: {fps:.1f}", 30, (200, 200, 200))
            put(f"Hand: {hand_label}  Fingers: {fingers}", 60, (255, 255, 0))
            put(f"Raw: {raw_gesture}", 90, (0, 255, 0))
            put(f"Stable: {stable_gesture}", 120, (0, 200, 255))
            put(f"Command: {command}  Throttle: {throttle:.2f}", 150, (255, 200, 0))
            put(f"v: vx={controller.vx:+.2f} vy={controller.vy:+.2f} "
                f"vz={controller.vz:+.2f} yaw={controller.yaw_rate:+.1f}",
                180, (180, 255, 180), 0.6)
            if altitude is not None:
                put(f"Altitude: {altitude:.2f} m", 210, (200, 200, 255))
            if auto_landed:
                put("AUTO-LANDED", 240, (0, 0, 255), 0.9)
            put("q=quit | l=land | r=rearm", frame.shape[0] - 20,
                (0, 200, 255), 0.6)

            cv2.imshow("Gesture Control v3", frame)
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
