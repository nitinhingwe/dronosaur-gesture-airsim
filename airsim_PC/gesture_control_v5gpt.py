import time
import collections
import cv2
import mediapipe as mp
import airsim


# =============================
# AirSim setup
# =============================
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()
time.sleep(1.0)

# Small initial lift
client.moveByVelocityBodyFrameAsync(0, 0, -0.8, 1.0).join()


# =============================
# MediaPipe setup
# =============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# =============================
# Camera setup
# =============================
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open USB camera. Try CAMERA_INDEX = 1")

cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Control", 1100, 750)


# =============================
# Control settings
# =============================
gesture_history = collections.deque(maxlen=7)
stable_gesture = "NONE"
current_command = "HOVER"
last_sent_display = "NONE"

last_send_time = 0.0
send_interval = 0.10   # smaller = smoother updates

# Speed modes
speed_mode = "PRECISION"

SPEED_PROFILES = {
    "PRECISION": {
        "vx_forward": 0.7,
        "vx_backward": -0.6,
        "vy_right": 0.6,
        "vy_left": -0.6,
        "vz_up": -0.6,
        "vz_down": 0.6,
        "yaw_rate": 15,
        "duration": 0.12
    },
    "FAST": {
        "vx_forward": 1.4,
        "vx_backward": -1.1,
        "vy_right": 1.0,
        "vy_left": -1.0,
        "vz_up": -1.0,
        "vz_down": 1.0,
        "yaw_rate": 28,
        "duration": 0.12
    }
}


# =============================
# Helper functions
# =============================
def fingers_state(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y

    if handedness_label == "Right":
        thumb_open = lm[4].x < lm[3].x
    else:
        thumb_open = lm[4].x > lm[3].x

    return [
        int(thumb_open),
        int(index_up),
        int(middle_up),
        int(ring_up),
        int(pinky_up)
    ]


def classify_gesture(fingers, hand_landmarks):
    thumb, index, middle, ring, pinky = fingers
    lm = hand_landmarks.landmark

    thumb_tip_above_base = lm[4].y < lm[2].y
    thumb_tip_left_of_index_mcp = lm[4].x < lm[5].x
    thumb_tip_right_of_index_mcp = lm[4].x > lm[5].x
    others_closed = (index == 0 and middle == 0 and ring == 0 and pinky == 0)

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

    # L Shape (thumb + index) -> Left
    if thumb == 1 and index == 1 and middle == 0 and ring == 0 and pinky == 0:
        if thumb_tip_left_of_index_mcp or thumb_tip_right_of_index_mcp:
            return "LEFT"

    # Thumbs Up -> Up
    if thumb == 1 and thumb_tip_above_base and others_closed:
        return "UP"

    return "UNKNOWN"


def get_stable_gesture(current_gesture):
    gesture_history.append(current_gesture)

    if len(gesture_history) < gesture_history.maxlen:
        return "NONE"

    most_common = collections.Counter(gesture_history).most_common(1)[0]

    if most_common[1] >= 5:
        return most_common[0]

    return "NONE"


def update_current_command(stable):
    global current_command
    if stable in ["HOVER", "FORWARD", "BACKWARD", "RIGHT", "LEFT", "UP"]:
        current_command = stable


def send_motion_command():
    global last_send_time, last_sent_display

    now = time.time()
    if now - last_send_time < send_interval:
        return

    last_send_time = now

    profile = SPEED_PROFILES[speed_mode]
    duration = profile["duration"]

    if current_command == "HOVER":
        client.hoverAsync()
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


def send_keyboard_command(key):
    global last_sent_display, current_command, speed_mode

    profile = SPEED_PROFILES[speed_mode]
    duration = profile["duration"]

    if key == ord("j"):
        client.moveByVelocityBodyFrameAsync(0, 0, profile["vz_down"], duration)
        last_sent_display = "DOWN"

    elif key == ord("a"):
        client.rotateByYawRateAsync(-profile["yaw_rate"], duration)
        last_sent_display = "YAW LEFT"

    elif key == ord("d"):
        client.rotateByYawRateAsync(profile["yaw_rate"], duration)
        last_sent_display = "YAW RIGHT"

    elif key == ord("m"):
        if speed_mode == "PRECISION":
            speed_mode = "FAST"
        else:
            speed_mode = "PRECISION"
        print("Speed mode:", speed_mode)

    elif key == ord("l"):
        print("Manual land triggered")
        client.landAsync().join()
        current_command = "HOVER"
        last_sent_display = "LAND"


# =============================
# Main loop
# =============================
try:
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

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_label = results.multi_handedness[0].classification[0].label

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            fingers = fingers_state(hand_landmarks, hand_label)
            raw_gesture = classify_gesture(fingers, hand_landmarks)
            stable_gesture = get_stable_gesture(raw_gesture)

            update_current_command(stable_gesture)

            cv2.putText(
                frame,
                f"Hand: {hand_label}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Fingers: {fingers}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )

        else:
            gesture_history.clear()
            stable_gesture = "NONE"
            current_command = "HOVER"

        # send gesture-driven command continuously
        send_motion_command()

        # overlay
        cv2.putText(
            frame,
            f"Raw Gesture: {raw_gesture}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Stable Gesture: {stable_gesture}",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 200, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Active Command: {current_command}",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 200, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Last Sent: {last_sent_display}",
            (20, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (180, 255, 180),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Speed Mode: {speed_mode}",
            (20, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 100, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "q=quit | l=land | j=down | a/d=yaw | m=speed",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 200, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Gesture Control", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key in [ord("j"), ord("a"), ord("d"), ord("m"), ord("l")]:
            send_keyboard_command(key)

finally:
    print("Cleaning up...")
    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    try:
        client.hoverAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
    except Exception:
        pass

    print("Finished")