import time
import collections
import cv2
import mediapipe as mp
import airsim


PC_IP = "YOUR_PC_IP"

CAMERA_INDEX = 0
FRAME_W = 480
FRAME_H = 360
TARGET_FPS = 30

gesture_history = collections.deque(maxlen=7)
stable_gesture = "NONE"
current_command = "HOVER"
last_sent_display = "NONE"

last_send_time = 0.0
send_interval = 0.10

last_valid_gesture_time = time.time()
NO_GESTURE_TIMEOUT = 0.20

# Real-drone-like safe speeds
VX_FORWARD = 0.65
VX_BACKWARD = -0.55
VY_RIGHT = 0.55
VY_LEFT = -0.55
VZ_UP = -0.45
VZ_DOWN = 0.45
YAW_RATE = 18
CMD_DURATION = 0.12


client = airsim.MultirotorClient(ip=PC_IP)
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()
time.sleep(1.0)
client.moveByVelocityBodyFrameAsync(0, 0, -0.5, 0.7).join()
client.hoverAsync().join()


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
)


cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

if not cap.isOpened():
    raise RuntimeError("Could not open USB camera")

cv2.namedWindow("Dronosaur Gesture AirSim Bridge V4", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Dronosaur Gesture AirSim Bridge V4", 720, 520)


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
        int(pinky_up),
    ]


def classify_gesture(fingers, hand_landmarks):
    thumb, index, middle, ring, pinky = fingers
    lm = hand_landmarks.landmark

    others_closed = index == 0 and middle == 0 and ring == 0 and pinky == 0

    thumb_tip_above = lm[4].y < lm[3].y and lm[4].y < lm[2].y
    thumb_tip_below = lm[4].y > lm[3].y and lm[4].y > lm[2].y

    # Thumb vertical commands first, because thumb_open can fail by angle
    if others_closed and thumb_tip_above:
        return "UP"

    if others_closed and thumb_tip_below:
        return "DOWN"

    # Open palm -> yaw right
    if fingers == [1, 1, 1, 1, 1]:
        return "YAW_RIGHT"

    # Victory -> forward
    if fingers == [0, 1, 1, 0, 0]:
        return "FORWARD"

    # Fist -> backward
    if fingers == [0, 0, 0, 0, 0]:
        return "BACKWARD"

    # Index only -> right
    if fingers == [0, 1, 0, 0, 0]:
        return "RIGHT"

    # Pinky only -> left
    if fingers == [0, 0, 0, 0, 1]:
        return "LEFT"

    # L shape -> yaw left
    if thumb == 1 and index == 1 and middle == 0 and ring == 0 and pinky == 0:
        return "YAW_LEFT"

    return "UNKNOWN"


def get_stable_gesture(current_gesture):
    gesture_history.append(current_gesture)

    if len(gesture_history) < gesture_history.maxlen:
        return "NONE"

    most_common = collections.Counter(gesture_history).most_common(1)[0]

    if most_common[1] >= 5 and most_common[0] != "UNKNOWN":
        return most_common[0]

    return "NONE"


def update_current_command(stable):
    global current_command, last_valid_gesture_time

    valid_commands = [
        "FORWARD",
        "BACKWARD",
        "RIGHT",
        "LEFT",
        "UP",
        "DOWN",
        "YAW_LEFT",
        "YAW_RIGHT",
    ]

    if stable in valid_commands:
        current_command = stable
        last_valid_gesture_time = time.time()


def force_hover_if_no_valid_gesture():
    global current_command
    if time.time() - last_valid_gesture_time > NO_GESTURE_TIMEOUT:
        current_command = "HOVER"


def send_motion_command():
    global last_send_time, last_sent_display

    now = time.time()
    if now - last_send_time < send_interval:
        return

    last_send_time = now

    if current_command == "HOVER":
        client.moveByVelocityBodyFrameAsync(0, 0, 0, CMD_DURATION)
        last_sent_display = "HOVER"

    elif current_command == "FORWARD":
        client.moveByVelocityBodyFrameAsync(VX_FORWARD, 0, 0, CMD_DURATION)
        last_sent_display = "FORWARD"

    elif current_command == "BACKWARD":
        client.moveByVelocityBodyFrameAsync(VX_BACKWARD, 0, 0, CMD_DURATION)
        last_sent_display = "BACKWARD"

    elif current_command == "RIGHT":
        client.moveByVelocityBodyFrameAsync(0, VY_RIGHT, 0, CMD_DURATION)
        last_sent_display = "RIGHT"

    elif current_command == "LEFT":
        client.moveByVelocityBodyFrameAsync(0, VY_LEFT, 0, CMD_DURATION)
        last_sent_display = "LEFT"

    elif current_command == "UP":
        client.moveByVelocityBodyFrameAsync(0, 0, VZ_UP, CMD_DURATION)
        last_sent_display = "UP"

    elif current_command == "DOWN":
        client.moveByVelocityBodyFrameAsync(0, 0, VZ_DOWN, CMD_DURATION)
        last_sent_display = "DOWN"

    elif current_command == "YAW_LEFT":
        client.rotateByYawRateAsync(-YAW_RATE, CMD_DURATION)
        last_sent_display = "YAW_LEFT"

    elif current_command == "YAW_RIGHT":
        client.rotateByYawRateAsync(YAW_RATE, CMD_DURATION)
        last_sent_display = "YAW_RIGHT"


prev_time = time.time()
fps = 0.0

try:
    print("Running V4 gesture AirSim control")
    print("Controls: l=land | q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            current_command = "HOVER"
            send_motion_command()
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        raw_gesture = "No Hand"
        hand_label = "None"
        fingers = None

        now = time.time()

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_label = results.multi_handedness[0].classification[0].label

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = fingers_state(hand_landmarks, hand_label)
            raw_gesture = classify_gesture(fingers, hand_landmarks)
            stable_gesture = get_stable_gesture(raw_gesture)

            update_current_command(stable_gesture)

            if stable_gesture == "NONE":
                force_hover_if_no_valid_gesture()

        else:
            gesture_history.clear()
            stable_gesture = "NONE"
            current_command = "HOVER"

        send_motion_command()

        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(frame, f"Raw Gesture: {raw_gesture}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Stable Gesture: {stable_gesture}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 200, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Active Command: {current_command}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 200, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Last Sent: {last_sent_display}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (180, 255, 180), 2, cv2.LINE_AA)

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

        if fingers is not None:
            cv2.putText(frame, f"Fingers: {fingers} | Hand: {hand_label}", (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, "l=land | q=quit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow("Dronosaur Gesture AirSim Bridge V4", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("l"):
            print("Emergency land triggered")
            client.landAsync().join()
            current_command = "HOVER"

        if key == ord("q"):
            break

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