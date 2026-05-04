import cv2
import time
import csv
import math
from collections import deque, Counter

import airsim
import mediapipe as mp


# =========================
# CONFIG
# =========================
PC_IP = "172.26.76.45"   # change this to your Windows PC IP

CAM_INDEX = 0
FRAME_W = 480
FRAME_H = 360
TARGET_FPS = 30
FLIP_IMAGE = True

LOG_PATH = "gesture_airsim_v2_log.csv"

MAX_HANDS = 1
DETECTION_CONF = 0.65
TRACKING_CONF = 0.65
FINGER_EXTEND_MIN = 0.02

# Gesture stability
SMOOTH_WINDOW = 8
STABLE_MIN_RATIO = 0.65
INTENT_HOLD_TIME = 0.45       # gesture must stay stable this long before changing intent
LOST_HAND_TIMEOUT = 0.60      # if hand lost/unstable this long -> hover

# Control streaming
CONTROL_INTERVAL = 0.10       # 10 Hz control loop
COMMAND_DURATION = 0.20       # each velocity command lasts 0.2 sec

# Speeds
XY_SPEED = 0.65
Z_SPEED = 0.45
YAW_RATE = 25.0               # deg/sec, for later yaw

# =========================
# GESTURE HELPERS
# =========================
def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def finger_extended(lm, tip_id, pip_id, mcp_id):
    tip = lm[tip_id]
    pip = lm[pip_id]
    mcp = lm[mcp_id]
    return (dist(tip, mcp) - dist(pip, mcp)) > FINGER_EXTEND_MIN


def thumb_extended(lm):
    tip = lm[4]
    ip = lm[3]
    mcp = lm[2]
    return (dist(tip, mcp) - dist(ip, mcp)) > FINGER_EXTEND_MIN


def thumb_direction(lm):
    """
    Simple thumb up/down check.
    Returns: UP, DOWN, SIDE
    """
    tip = lm[4]
    ip = lm[3]

    if tip.y < ip.y - 0.04:
        return "UP"
    if tip.y > ip.y + 0.04:
        return "DOWN"
    return "SIDE"


def classify_gesture(lm):
    thumb = thumb_extended(lm)
    index = finger_extended(lm, 8, 6, 5)
    middle = finger_extended(lm, 12, 10, 9)
    ring = finger_extended(lm, 16, 14, 13)
    pinky = finger_extended(lm, 20, 18, 17)

    fingers = [thumb, index, middle, ring, pinky]
    count_ext = sum(fingers)

    # Open palm
    if count_ext == 5:
        return "OPEN_PALM", 0.95

    # Fist
    if count_ext == 0:
        return "FIST", 0.95

    # Victory / two fingers
    if index and middle and (not ring) and (not pinky):
        return "TWO_FINGERS", 0.90

    # Point / index only
    if index and (not middle) and (not ring) and (not pinky):
        return "POINT", 0.85

    # Pinky only
    if pinky and (not index) and (not middle) and (not ring):
        return "PINKY", 0.85

    # Thumb only: up/down
    if thumb and (not index) and (not middle) and (not ring) and (not pinky):
        direction = thumb_direction(lm)
        if direction == "UP":
            return "THUMBS_UP", 0.88
        if direction == "DOWN":
            return "THUMBS_DOWN", 0.88
        return "THUMB_SIDE", 0.70

    # Spiderman-ish: thumb + index + pinky
    if thumb and index and pinky and (not middle) and (not ring):
        return "SPIDERMAN", 0.85

    return "UNKNOWN", 0.40

def stabilized_label(buffer):
    if not buffer:
        return "NONE", 0.0

    c = Counter(buffer)
    label, cnt = c.most_common(1)[0]
    ratio = cnt / len(buffer)

    if ratio >= STABLE_MIN_RATIO and label not in ["UNKNOWN", "NONE"]:
        return label, ratio

    return "UNSTABLE", ratio

def send_intent(client, intent):
    if intent == "HOVER":
        client.hoverAsync().join()

    elif intent == "FORWARD":
        client.moveByVelocityBodyFrameAsync(1.0, 0, 0, 0.3).join()

    elif intent == "RIGHT":
        client.moveByVelocityBodyFrameAsync(0, 1.0, 0, 0.3).join()

    elif intent == "LEFT":
        client.moveByVelocityBodyFrameAsync(0, -1.0, 0, 0.3).join()

    elif intent == "UP":
        client.moveByVelocityAsync(0, 0, -1.0, 0.3).join()

    elif intent == "DOWN":
        client.moveByVelocityAsync(0, 0, 1.0, 0.3).join()

def gesture_to_intent(gesture):
    """
    First V2 mapping.
    Safe and simple.
    """
    mapping = {
        "OPEN_PALM": "HOVER",
        "TWO_FINGERS": "FORWARD",
        "SPIDERMAN": "BACKWARD",
        "POINT": "RIGHT",
        "PINKY": "LEFT",
        "THUMBS_UP": "UP",
        "THUMBS_DOWN": "DOWN",
        "FIST": "HOVER",
    }
    return mapping.get(gesture, "HOVER")

# =========================
# MAIN
# =========================
def main():
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient(ip=PC_IP)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)
    client.hoverAsync().join()

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        print("ERROR: Camera not opened.")
        client.landAsync().join()
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        model_complexity=0,
        min_detection_confidence=DETECTION_CONF,
        min_tracking_confidence=TRACKING_CONF,
    )

    gesture_buffer = deque(maxlen=SMOOTH_WINDOW)

    current_intent = "HOVER"
    candidate_intent = "HOVER"
    candidate_start_time = time.time()
    last_valid_gesture_time = time.time()
    last_control_time = 0.0

    prev_time = time.time()
    fps = 0.0

    print("Running V2 gesture AirSim control")
    print("Press 'q' to land and quit.")

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "timestamp",
                "raw_gesture",
                "stable_gesture",
                "stable_ratio",
                "current_intent",
                "candidate_intent",
                "fps"
            ])

        try:
            while True:
                now = time.time()

                ret, frame = cap.read()
                if not ret:
                    continue

                if FLIP_IMAGE:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                raw_label = "NONE"
                raw_conf = 0.0

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    lm = hand_landmarks.landmark

                    raw_label, raw_conf = classify_gesture(lm)
                    gesture_buffer.append(raw_label)
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                else:
                    gesture_buffer.append("NONE")

                stable_gesture, stable_ratio = stabilized_label(gesture_buffer)

                # =========================
                # INTENT STATE MACHINE
                # =========================
                if stable_gesture != "UNSTABLE":
                    new_intent = gesture_to_intent(stable_gesture)
                    last_valid_gesture_time = now

                    if new_intent != candidate_intent:
                        candidate_intent = new_intent
                        candidate_start_time = now

                    # change actual intent only after hold time
                    if (now - candidate_start_time) >= INTENT_HOLD_TIME:
                        current_intent = candidate_intent

                else:
                    # If hand/gesture lost for too long, return to hover
                    if (now - last_valid_gesture_time) >= LOST_HAND_TIMEOUT:
                        current_intent = "HOVER"
                        candidate_intent = "HOVER"
                        candidate_start_time = now

                # =========================
                # CONTROL STREAMING LOOP
                # =========================
                if (now - last_control_time) >= CONTROL_INTERVAL:
                    send_intent(client, current_intent)
                    last_control_time = now

                # FPS
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                # Display
                cv2.putText(frame, f"RAW: {raw_label} ({raw_conf:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(frame, f"STABLE: {stable_gesture} ({stable_ratio:.2f})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(frame, f"INTENT: {current_intent}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                bar_x, bar_y = 10, 140
                bar_w, bar_h = 200, 16
                fill = int(bar_w * clamp(stable_ratio, 0.0, 1.0))
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (255, 255, 255), -1)

                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    raw_label,
                    stable_gesture,
                    f"{stable_ratio:.2f}",
                    current_intent,
                    candidate_intent,
                    f"{fps:.1f}"
                ])
                f.flush()

                cv2.imshow("Dronosaur Gesture AirSim Bridge V2", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Landing...")
                    client.landAsync().join()
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            client.hoverAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
            print("Exited cleanly.")


if __name__ == "__main__":
    main()
