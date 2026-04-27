# ====== V5 FINAL (Based on your V4) ======

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
send_interval = 0.08   # faster loop

last_valid_gesture_time = time.time()
NO_GESTURE_TIMEOUT = 0.15

# 🚀 SPEED SYSTEM
speed_multiplier = 1.0
MIN_SPEED = 0.6
MAX_SPEED = 2.0
SPEED_STEP = 0.2

last_command_time = time.time()
COMMAND_HOLD_TIME = 0.3

# ⚡ Faster base speeds
VX_FORWARD = 1.2
VX_BACKWARD = -1.2
VY_RIGHT = 1.0
VY_LEFT = -1.0
VZ_UP = -0.8
VZ_DOWN = 0.8
YAW_RATE = 30
CMD_DURATION = 0.25

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

cv2.namedWindow("Dronosaur Gesture AirSim Bridge V5", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Dronosaur Gesture AirSim Bridge V5", 720, 520)

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

    return [int(thumb_open), int(index_up), int(middle_up), int(ring_up), int(pinky_up)]

def classify_gesture(fingers, lm):
    thumb, index, middle, ring, pinky = fingers

    others_closed = index == 0 and middle == 0 and ring == 0 and pinky == 0

    thumb_tip_above = lm[4].y < lm[3].y
    thumb_tip_below = lm[4].y > lm[3].y

    if others_closed and thumb_tip_above:
        return "UP"

    if others_closed and thumb_tip_below:
        return "DOWN"

    if fingers == [1,1,1,1,1]:
        return "YAW_RIGHT"

    if fingers == [0,1,1,0,0]:
        return "FORWARD"

    if fingers == [0,0,0,0,0]:
        return "BACKWARD"

    if fingers == [0,1,0,0,0]:
        return "RIGHT"

    if fingers == [0,0,0,0,1]:
        return "LEFT"

    if thumb == 1 and index == 1 and middle == 0 and ring == 0 and pinky == 0:
        return "YAW_LEFT"

    # 🕷️ Spiderman = speed up
    if fingers == [1,1,0,0,1]:
        return "SPEED_UP"

    # 🤟 Three fingers = speed down
    if fingers == [0,1,1,0,1]:
        return "SPEED_DOWN"

    return "UNKNOWN"

def get_stable_gesture(g):
    gesture_history.append(g)
    if len(gesture_history) < gesture_history.maxlen:
        return "NONE"

    most_common = collections.Counter(gesture_history).most_common(1)[0]
    if most_common[1] >= 5 and most_common[0] != "UNKNOWN":
        return most_common[0]

    return "NONE"

def update_command(stable):
    global current_command, speed_multiplier, last_valid_gesture_time

    if stable == "SPEED_UP":
        speed_multiplier = min(MAX_SPEED, speed_multiplier + SPEED_STEP)
        print(f"Speed ↑ {speed_multiplier}")
        return

    if stable == "SPEED_DOWN":
        speed_multiplier = max(MIN_SPEED, speed_multiplier - SPEED_STEP)
        print(f"Speed ↓ {speed_multiplier}")
        return

    valid = ["FORWARD","BACKWARD","RIGHT","LEFT","UP","DOWN","YAW_LEFT","YAW_RIGHT"]

    if stable in valid:
        current_command = stable
        last_valid_gesture_time = time.time()

def send_motion():
    global last_send_time, last_command_time

    now = time.time()
    if now - last_send_time < send_interval:
        return

    last_send_time = now

    if current_command != "HOVER":
        last_command_time = now

    # HOLD behavior
    if current_command == "HOVER":
        if now - last_command_time < COMMAND_HOLD_TIME:
            return
        client.hoverAsync()
        return

    s = speed_multiplier

    if current_command == "FORWARD":
        client.moveByVelocityBodyFrameAsync(VX_FORWARD*s,0,0,CMD_DURATION)

    elif current_command == "BACKWARD":
        client.moveByVelocityBodyFrameAsync(VX_BACKWARD*s,0,0,CMD_DURATION)

    elif current_command == "RIGHT":
        client.moveByVelocityBodyFrameAsync(0,VY_RIGHT*s,0,CMD_DURATION)

    elif current_command == "LEFT":
        client.moveByVelocityBodyFrameAsync(0,VY_LEFT*s,0,CMD_DURATION)

    elif current_command == "UP":
        client.moveByVelocityBodyFrameAsync(0,0,VZ_UP*s,CMD_DURATION)

    elif current_command == "DOWN":
        client.moveByVelocityBodyFrameAsync(0,0,VZ_DOWN*s,CMD_DURATION)

    elif current_command == "YAW_LEFT":
        client.rotateByYawRateAsync(-YAW_RATE*s, CMD_DURATION)

    elif current_command == "YAW_RIGHT":
        client.rotateByYawRateAsync(YAW_RATE*s, CMD_DURATION)

# ===== LOOP =====

while True:
    ret, frame = cap.read()
    if not ret:
        current_command = "HOVER"
        send_motion()
        continue

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        lm = results.multi_hand_landmarks[0]
        label = results.multi_handedness[0].classification[0].label

        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        fingers = fingers_state(lm,label)
        raw = classify_gesture(fingers, lm.landmark)
        stable = get_stable_gesture(raw)

        update_command(stable)

    else:
        current_command = "HOVER"

    send_motion()

    cv2.putText(frame,f"Cmd: {current_command}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

    cv2.putText(frame,f"Speed: x{speed_multiplier:.1f}",(20,80),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

    cv2.imshow("V5",frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("l"):
        client.landAsync().join()
    if key == ord("q"):
        break