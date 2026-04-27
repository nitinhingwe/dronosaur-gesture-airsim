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
CAMERA_INDEX = 0   # change to 1 if your USB camera is not on 0
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open USB camera. Try CAMERA_INDEX = 1")

cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Control", 1000, 700)


# =============================
# Control settings
# =============================
gesture_history = collections.deque(maxlen=7)
stable_gesture = "NONE"
current_command = "HOVER"

last_send_time = 0.0
send_interval = 0.12   # smaller = smoother but more command spam

# movement tuning
VX_FORWARD = 0.8
VX_BACKWARD = -0.7
VY_RIGHT = 0.7
VY_LEFT = -0.7
VZ_UP = -0.7

last_sent_display = "NONE"


# =============================
# Helper functions
# =============================
def fingers_state(hand_landmarks, handedness_label):
    """
    Returns finger states:
    [thumb, index, middle, ring, pinky]
    """
    lm = hand_landmarks.landmark

    # Standard finger-up logic
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y

    # Thumb logic depends on handedness
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
    """
    Final V4 gesture classifier
    """
    thumb, index, middle, ring, pinky = fingers
    lm = hand_landmarks.landmark

    # helpful thumb geometry
    thumb_tip_above_base = lm[4].y < lm[2].y
    thumb_tip_left_of_index_mcp = lm[4].x < lm[5].x
    thumb_tip_right_of_index_mcp = lm[4].x > lm[5].x

    # 1) Open Palm -> Hover
    if fingers == [1, 1, 1, 1, 1]:
        return "HOVER"

    # 2) Victory -> Forward
    if fingers == [0, 1, 1, 0, 0]:
        return "FORWARD"

    # 3) Closed Fist -> Backward
    if fingers == [0, 0, 0, 0, 0]:
        return "BACKWARD"

    # 4) Index Only -> Right
    if fingers == [0, 1, 0, 0, 0]:
        return "RIGHT"

    # 5) L Shape (thumb + index) -> Left
    if thumb == 1 and index == 1 and middle == 0 and ring == 0 and pinky == 0:
        # extra looseness: allow thumb to be visibly extended away from hand
        if thumb_tip_left_of_index_mcp or thumb_tip_right_of_index_mcp:
            return "LEFT"

    # 6) Thumbs Up -> Up
    others_closed = (index == 0 and middle == 0 and ring == 0 and pinky == 0)
    if thumb == 1 and thumb_tip_above_base and others_closed:
        return "UP"

    return "UNKNOWN"


def get_stable_gesture(current_gesture):
    gesture_history.append(current_gesture)

    if len(gesture_history) < gesture_history.maxlen:
        return "NONE"

    most_common = collections.Counter(gesture_history).most_common(1)[0]

    # Require strong majority to avoid flicker
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
    last_sent_display = current_command

    if current_command == "HOVER":
        client.hoverAsync()

    elif current_command == "FORWARD":
        client.moveByVelocityBodyFrameAsync(VX_FORWARD, 0, 0, 0.15)

    elif current_command == "BACKWARD":
        client.moveByVelocityBodyFrameAsync(VX_BACKWARD, 0, 0, 0.15)

    elif current_command == "RIGHT":
        client.moveByVelocityBodyFrameAsync(0, VY_RIGHT, 0, 0.15)

    elif current_command == "LEFT":
        client.moveByVelocityBodyFrameAsync(0, VY_LEFT, 0, 0.15)

    elif current_command == "UP":
        client.moveByVelocityBodyFrameAsync(0, 0, VZ_UP, 0.15)


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

        # send active command continuously
        send_motion_command()

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
            "q = quit | l = land",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Gesture Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("l"):
            print("Manual land triggered")
            client.landAsync().join()

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