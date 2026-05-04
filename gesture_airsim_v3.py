import time
import collections
import cv2
import mediapipe as mp
import airsim


# =============================
# CONFIG
# =============================
PC_IP = "172.26.76.63"   # change this to current Windows PC IP

CAMERA_INDEX = 0
FRAME_W = 480
FRAME_H = 360
TARGET_FPS = 30

# Gesture smoothing
gesture_history = collections.deque(maxlen=7)
stable_gesture = "NONE"
current_command = "HOVER"
last_sent_display = "NONE"

# Command timing
last_send_time = 0.0
send_interval = 0.10   # 10Hz command streaming

# Safety
last_hand_seen_time = time.time()
HAND_LOST_TIMEOUT = 0.25

# Speed modes
speed_mode = "PRECISION"

SPEED_PROFILES = {
    "PRECISION": {
        "vx_forward": 0.65,
        "vx_backward": -0.55,
        "vy_right": 0.55,
        "vy_left": -0.55,
        "vz_up": -0.45,
        "vz_down": 0.45,
        "yaw_rate": 15,
        "duration": 0.12,
    },
    "FAST": {
        "vx_forward": 1.2,
        "vx_backward": -1.0,
        "vy_right": 0.9,
        "vy_left": -0.9,
        "vz_up": -0.8,
        "vz_down": 0.8,
        "yaw_rate": 28,
        "duration": 0.12,
    },
}

# =============================
# AIRSIM SETUP
# =============================
client = airsim.MultirotorClient(ip=PC_IP)
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()
time.sleep(1.0)

# small lift after takeoff
client.moveByVelocityBodyFrameAsync(0, 0, -0.5, 0.7).join()
client.hoverAsync().join()


# =============================
# MEDIAPIPE SETUP
# =============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
)

# =============================
# CAMERA SETUP
# =============================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

if not cap.isOpened():
    raise RuntimeError("Could not open USB camera. Try CAMERA_INDEX = 1")

cv2.namedWindow("Dronosaur Gesture AirSim Bridge V3", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Dronosaur Gesture AirSim Bridge V3", 720, 520)


# =============================
# GESTURE HELPERS
# =============================
def fingers_state(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y

    # Thumb depends on detected hand side
    if handedness_label == "Right":
        thumb_open = lm[4].x < lm[3].x
    else:
        thumb_open = lm[4].x > lm[3].x

    return [
        int(thumb_open),
        int(index_up),
        int(middle_up),
        int(ring_up),
        int(pinky_up),
    ]


def classify_gesture(fingers, hand_landmarks):
    thumb, index, middle, ring, pinky = fingers
    lm = hand_landmarks.landmark

    thumb_tip_above_base = lm[4].y < lm[2].y
    thumb_tip_below_base = lm[4].y > lm[2].y
    others_closed = index == 0 and middle == 0 and ring == 0 and pinky == 0

    # Open Palm -> Hover
    if fingers == [1, 1, 1, 1, 1]:
        return "HOVER"

    # Victory -> Forward
    if fingers == [0, 1, 1, 0, 0]:
        return "FORWARD"

    # Closed Fist -> Backward
    if fingers == [0, 0, 0, 0, 0]:
        return "BACKWARD"

    # Index Only -> Right
    if fingers == [0, 1, 0, 0, 0]:
        return "RIGHT"

    # Pinky Only -> Left
    if fingers == [0, 0, 0, 0, 1]:
        return "LEFT"

    # L Shape, thumb + index -> Left fallback
    if thumb == 1 and index == 1 and middle == 0 and ring == 0 and pinky == 0:
        return "LEFT"

    # Thumbs Up -> Up
    if thumb == 1 and thumb_tip_above_base and others_closed:
        return "UP"

    # Thumbs Down -> Down
    if thumb == 1 and thumb_tip_below_base and others_closed:
        return "DOWN"

    return "UNKNOWN"

def get_stable_gesture(current_gesture):
    gesture_history.append(current_gesture)

    if len(gesture_history) < gesture_history.maxlen:
        return "NONE"

    most_common = collections.Counter(gesture_history).most_common(1)[0]

    # 5 out of 7 frames need to agree
    if most_common[1] >= 5:
        return most_common[0]

    return "NONE"


def update_current_command(stable):
    global current_command

    valid_commands = [
        "HOVER",
        "FORWARD",
        "BACKWARD",
        "RIGHT",
        "LEFT",
        "UP",
        "DOWN",
    ]

    if stable in valid_commands:
        current_command = stable


# =============================
# AIRSIM COMMANDS
# =============================
def send_motion_command():
    global last_send_time, last_sent_display

    now = time.time()
    if now - last_send_time < send_interval:
        return

    last_send_time = now

    profile = SPEED_PROFILES[speed_mode]
    duration = profile["duration"]

    if current_command == "HOVER":
        client.moveByVelocityBodyFrameAsync(0, 0, 0, duration)
        last_sent_display = "HOVER"

    elif current_command == "FORWARD":
        client.moveByVelocityBodyFrameAsync(profile["vx_forward"], 0, 0, duration)
        last_sent_display = "FORWARD"

    elif current_command == "BACKWARD":
        client.moveByVelocityBodyFrameAsync(profile["vx_backward"], 0, 0, duration)
        last_sent_display = "BACKWARD"

    elif current_command == "RIGHT":
        client.moveByVelocityBodyFrameAsync(0, profile["vy_right"], 0, duration)
        last_sent_display = "RIGHT"

    elif current_command == "LEFT":
        client.moveByVelocityBodyFrameAsync(0, profile["vy_left"], 0, duration)
        last_sent_display = "LEFT"

    elif current_command == "UP":
        client.moveByVelocityBodyFrameAsync(0, 0, profile["vz_up"], duration)
        last_sent_display = "UP"

    elif current_command == "DOWN":
        client.moveByVelocityBodyFrameAsync(0, 0, profile["vz_down"], duration)
        last_sent_display = "DOWN"


def send_keyboard_command(key):
    global last_sent_display, current_command, speed_mode

    profile = SPEED_PROFILES[speed_mode]
    duration = profile["duration"]

    if key == ord("a"):
        client.rotateByYawRateAsync(-profile["yaw_rate"], duration)
        last_sent_display = "YAW LEFT"

    elif key == ord("d"):
        client.rotateByYawRateAsync(profile["yaw_rate"], duration)
        last_sent_display = "YAW RIGHT"

    elif key == ord("m"):
        speed_mode = "FAST" if speed_mode == "PRECISION" else "PRECISION"
        print("Speed mode:", speed_mode)

    elif key == ord("l"):
        print("Manual land triggered")
        client.landAsync().join()
        current_command = "HOVER"
        last_sent_display = "LAND"

# =============================
# MAIN LOOP
# =============================
prev_time = time.time()
fps = 0.0

try:
    print("Running V3 gesture AirSim control")
    print("Controls: q=quit | l=land | a/d=yaw | m=speed mode")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read from USB camera")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        raw_gesture = "No Hand"
        hand_label = "None"
        fingers = None

        now = time.time()

        if results.multi_hand_landmarks and results.multi_handedness:
            last_hand_seen_time = now

            hand_landmarks = results.multi_hand_landmarks[0]
            hand_label = results.multi_handedness[0].classification[0].label

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )

            fingers = fingers_state(hand_landmarks, hand_label)
            raw_gesture = classify_gesture(fingers, hand_landmarks)
            stable_gesture = get_stable_gesture(raw_gesture)

            update_current_command(stable_gesture)

        else:
            gesture_history.clear()
            stable_gesture = "NONE"

            # if hand disappears, hover quickly
            if now - last_hand_seen_time > HAND_LOST_TIMEOUT:
                current_command = "HOVER"

        # continuous command streaming
        send_motion_command()

        # FPS calculation
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Overlay
        cv2.putText(frame, f"Raw Gesture: {raw_gesture}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Stable Gesture: {stable_gesture}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 200, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Active Command: {current_command}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 200, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Last Sent: {last_sent_display}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (180, 255, 180), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Speed Mode: {speed_mode}", (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 100, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

        if fingers is not None:
            cv2.putText(frame, f"Fingers: {fingers} | Hand: {hand_label}", (20, 245),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, "q=quit | l=land | a/d=yaw | m=speed",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow("Dronosaur Gesture AirSim Bridge V3", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key in [ord("a"), ord("d"), ord("m"), ord("l")]:
            send_keyboard_command(key)

finally:
    print("Cleaning up...")

    try:
        client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.2).join()
        client.hoverAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
    except Exception:
        pass

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print("Finished")
