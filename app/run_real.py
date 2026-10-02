import sys
import time
from pathlib import Path

import cv2
import yaml
import mediapipe as mp

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from perception.gesture_detector import GestureDetector
from decision.gesture_to_command import gesture_to_command
from decision.filters import CommandFilter
from adapters.pixhawk_adapter import PixhawkAdapter


with open(BASE_DIR / "config" / "real.yaml", "r") as f:
    cfg = yaml.safe_load(f)

def main():
    
    if cfg["safety"]["allow_takeoff"] is True:
        raise RuntimeError("Safety stop: allow_takeoff must remain false")

    if cfg["safety"]["allow_arm"] is True:
        print("ARMING ENABLED: props must be removed")
    else:
        print("ARMING DISABLED: allow_arm=false")

    adapter = PixhawkAdapter(
        device=cfg["connection"]["device"],
        baud=cfg["connection"]["baud"],
    )

    adapter.connect()
    adapter.print_status()

    gesture_detector = GestureDetector(history_size=7, min_votes=5)
    command_filter = CommandFilter(cooldown=0.4, hold_time=0.15)

    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open USB camera")

    current_command = "HOVER"
    last_valid_time = time.time()
    no_gesture_timeout = 0.25

    last_command_sent = None
    last_real_send_time = 0.0

    window_name = "Dronosaur REAL Bench Control"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 720, 520)

    print("Running REAL bench control")
    print("PROPS REMOVED | RC ON | allow_arm=false")
    print("Throttle gestures disabled in adapter for bench safety")
    print("Keys: q=quit")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                current_command = "HOVER"
                adapter.send_command(current_command)

                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            raw_gesture, stable_gesture, hand_landmarks, fingers = gesture_detector.process(rgb)

            if hand_landmarks is not None:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                raw_command = gesture_to_command(stable_gesture)

                if raw_command is not None:
                    current_command = command_filter.apply(raw_command)
                    last_valid_time = time.time()
                elif time.time() - last_valid_time > no_gesture_timeout:
                    current_command = "HOVER"
            else:
                current_command = "HOVER"

            adapter.send_command(current_command)

            cv2.putText(frame, f"Raw: {raw_gesture}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.putText(frame, f"Stable: {stable_gesture}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

            cv2.putText(frame, f"Command: {current_command}", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

            cv2.putText(frame, f"Last Sent: {adapter.last_command}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 180), 2)

            cv2.putText(frame, "REAL BENCH MODE | q=quit",
                        (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("a"):
                if cfg["safety"]["allow_arm"] is True:
                    adapter.arm()
                
                else:
                    print("ARM blocked: allow_arm=false")
                    
            if key == ord("d"):
                adapter.disarm()
    
            if key == ord("q"):
                break

    finally:
        print("Cleaning up real bench control...")
        adapter.clear_rc_override()
        adapter.cleanup()
        cap.release()
        gesture_detector.hands.close()
        cv2.destroyAllWindows()
        print("Finished")


if __name__ == "__main__":
    main()
