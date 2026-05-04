import cv2
import time
import csv
import math
from collections import deque, Counter
import mediapipe as mp
import airsim

# =========================
# CONFIG
# =========================
PC_IP = "172.26.76.45"

CAM_INDEX = 0
FRAME_W = 640
FRAME_H = 480
TARGET_FPS = 30
FLIP_IMAGE = True

SMOOTH_WINDOW = 10
STABLE_MIN_RATIO = 0.6

LOG_PATH = "gesture_airsim_log.csv"

MAX_HANDS = 1
DETECTION_CONF = 0.6
TRACKING_CONF = 0.6

FINGER_EXTEND_MIN = 0.02

COMMAND_INTERVAL = 0.8   # minimum time between repeated commands
MOVE_DURATION = 0.25     # seconds for each movement burst


# =========================
# HELPERS
# =========================
def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def finger_extended(lm, tip_id, pip_id, mcp_id):
    tip = lm[tip_id]
    pip = lm[pip_id]
    mcp = lm[mcp_id]
    d_tip_mcp = dist(tip, mcp)
    d_pip_mcp = dist(pip, mcp)
    return (d_tip_mcp - d_pip_mcp) > FINGER_EXTEND_MIN

def thumb_extended(lm):
    tip = lm[4]
    ip = lm[3]
    mcp = lm[2]
    return (dist(tip, mcp) - dist(ip, mcp)) > FINGER_EXTEND_MIN

def classify_gesture(lm):
    thumb = thumb_extended(lm)
    index = finger_extended(lm, 8, 6, 5)
    middle = finger_extended(lm, 12, 10, 9)
    ring = finger_extended(lm, 16, 14, 13)
    pinky = finger_extended(lm, 20, 18, 17)

    fingers = [thumb, index, middle, ring, pinky]
    count_ext = sum(fingers)

    if count_ext == 5:
        return "OPEN_PALM", 0.95
    if count_ext == 0:
        return "FIST", 0.95
    if index and middle and (not ring) and (not pinky):
        return "TWO_FINGERS", 0.90
    if index and (not middle) and (not ring) and (not pinky):
        return "POINT", 0.85

    return "UNKNOWN", 0.40

def stabilized_label(buffer, min_ratio=0.6):
    if not buffer:
        return "NONE", 0.0

    c = Counter(buffer)
    label, cnt = c.most_common(1)[0]
    ratio = cnt / len(buffer)

    if ratio >= min_ratio and label != "UNKNOWN":
        return label, ratio
    return "UNSTABLE", ratio

def send_airsim_command(client, gesture):
    if gesture == "OPEN_PALM":
        print("Command: HOVER")
        client.hoverAsync().join()

    elif gesture == "TWO_FINGERS":
        print("Command: FORWARD")
        client.moveByVelocityBodyFrameAsync(
            1.0, 0.0, 0.0, MOVE_DURATION
        ).join()

    elif gesture == "POINT":
        print("Command: RIGHT")
        client.moveByVelocityBodyFrameAsync(
            0.0, 1.0, 0.0, MOVE_DURATION
        ).join()

    elif gesture == "FIST":
        print("Command: HOVER (safe)")
        client.hoverAsync().join()
        
# =========================
# MAIN
# =========================
def main():
    # ---------- AirSim setup ----------
    client = airsim.MultirotorClient(ip=PC_IP)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)
    client.hoverAsync().join()

    # ---------- CSV log ----------
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "timestamp",
                "raw_gesture",
                "raw_conf",
                "stable_gesture",
                "stable_ratio"
            ])

        # ---------- Camera ----------
        cap = cv2.VideoCapture(CAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        if not cap.isOpened():
            print("ERROR: Camera not opened. Try changing CAM_INDEX.")
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
            return

        # ---------- MediaPipe ----------
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils

        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=DETECTION_CONF,
            min_tracking_confidence=TRACKING_CONF,
        )

        gesture_buffer = deque(maxlen=SMOOTH_WINDOW)

        prev_time = time.time()
        fps = 0.0

        last_sent_gesture = "NONE"
        last_command_time = 0

        print("Running gesture + AirSim control")
        print("Press 'q' to land and quit.")
 
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("WARNING: Frame not received.")
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

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                else:
                    gesture_buffer.append("NONE")

                stable_label, stable_ratio = stabilized_label(
                    gesture_buffer,
                    STABLE_MIN_RATIO
                )

                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                # send command only if stable and enough time passed
                if stable_label in ["OPEN_PALM", "TWO_FINGERS", "POINT", "FIST"]:
                    if (stable_label != last_sent_gesture) or ((now - last_command_time) > COMMAND_INTERVAL):
                        send_airsim_command(client, stable_label)
                        last_sent_gesture = stable_label
                        last_command_time = now

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([
                    timestamp,
                    raw_label,
                    f"{raw_conf:.2f}",
                    stable_label,
                    f"{stable_ratio:.2f}"
                ])
                f.flush()

                cv2.putText(frame, f"RAW: {raw_label} ({raw_conf:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.putText(frame, f"STABLE: {stable_label} ({stable_ratio:.2f})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                bar_x, bar_y = 10, 110
                bar_w, bar_h = 200, 18
                fill = int(bar_w * clamp(stable_ratio, 0.0, 1.0))
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (255, 255, 255), -1)

                cv2.imshow("Dronosaur Gesture AirSim Bridge", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Landing...")
                    client.landAsync().join()
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            client.armDisarm(False)
            client.enableApiControl(False)
            print("Exited cleanly.")


if __name__ == "__main__":
    main()
