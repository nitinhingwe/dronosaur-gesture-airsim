import time
import collections
import cv2
import mediapipe as mp
import airsim


# -----------------------------
# AirSim setup
# -----------------------------
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()
time.sleep(1)

# Small initial lift
client.moveByVelocityBodyFrameAsync(0, 0, -0.8, 1.0).join()


# -----------------------------
# MediaPipe setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# -----------------------------
# Camera setup
# -----------------------------
CAMERA_INDEX = 0   # change to 1 if needed
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open USB camera. Try CAMERA_INDEX = 1")

cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Control", 1000, 700)


# -----------------------------
# Gesture smoothing / control
# -----------------------------
gesture_history = collections.deque(maxlen=7)
stable_gesture = "NONE"

last_send_time = 0
send_interval = 0.12   # smaller interval = smoother updates

current_command = "HOVER"


def fingers_state(hand_landmarks, handedness_label):
    """
    Returns finger states:
    [thumb, index, middle, ring, pinky]
    """
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
    """
    Safer rule-based gesture classification.
    LAND is manual only via keyboard.
    """
    thumb, index, middle, ring, pinky = fingers
    lm = hand_landmarks.landmark

    thumb_tip_above_base = lm[4].y < lm[2].y
    thumb_tip_below_base = lm[4].y > lm[5].y
    others_closed = (index == 0 and middle == 0 and ring == 0 and pinky == 0)

    # Open palm
    if fingers == [1, 1, 1, 1, 1]:
        return "HOVER"

    # Victory
    if fingers == [0, 1, 1, 0, 0]:
        return "FORWARD"

    # Index only
    if fingers == [0, 1, 0, 0, 0]:
        return "RIGHT"

    # Thumbs up
    if thumb == 1 and thumb_tip_above_base and others_closed:
        return "UP"

    # Thumbs down
    if thumb == 1 and thumb_tip_below_base and others_closed:
        return "DOWN"

    # Fist = hover-safe
    if fingers == [0, 0, 0, 0, 0]:
        return "HOVER"

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

    if stable in ["HOVER", "UP", "DOWN", "FORWARD", "RIGHT"]:
        current_command = stable


def send_motion_command():
    global last_send_time, current_command

    now = time.time()
    if now - last_send_time < send_interval:
        return

    last_send_time = now

    # Smoother smaller repeated motions
    if current_command == "HOVER":
        client.hoverAsync()

    elif current_command == "UP":
        client.moveByVelocityBodyFrameAsync(0, 0, -0.6, 0.15)

    elif current_command == "DOWN":
        client.moveByVelocityBodyFrameAsync(0, 0, 0.6, 0.15)

    elif current_command == "FORWARD":
        client.moveByVelocityBodyFrameAsync(0.7, 0, 0, 0.15)

    elif current_command == "RIGHT":
        client.moveByVelocityBodyFrameAsync(0, 0.7, 0, 0.15)


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
        fingers = None
        hand_label = "None"

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
            stable = get_stable_gesture(raw_gesture)
            stable_gesture = stable

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

        # send current command continuously
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