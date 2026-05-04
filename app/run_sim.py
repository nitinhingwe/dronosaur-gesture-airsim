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
from adapters.airsim_adapter import AirSimAdapter


with open(BASE_DIR / "config" / "sim.yaml", "r") as f:
    cfg = yaml.safe_load(f)


def main():
    pc_ip = cfg["pc_ip"]

    camera_index = cfg["camera"]["index"]
    frame_w = cfg["camera"]["width"]
    frame_h = cfg["camera"]["height"]
    target_fps = cfg["camera"]["fps"]

    no_gesture_timeout = cfg["gesture"]["no_gesture_timeout"]

    current_command = "HOVER"
    last_valid_gesture_time = time.time()
    last_sent_display = "NONE"

    airsim_adapter = AirSimAdapter(
        ip=pc_ip,
        speed_cfg={
            "forward": cfg["speed"]["forward"],
            "backward": cfg["speed"]["backward"],
            "right": cfg["speed"]["right"],
            "left": cfg["speed"]["left"],
            "up": cfg["speed"]["up"],
            "down": cfg["speed"]["down"],
            "yaw_rate": cfg["speed"]["yaw_rate"],
            "command_duration": cfg["speed"]["command_duration"],
            "send_interval": cfg["speed"]["send_interval"],
        },
    )

    airsim_adapter.connect_and_takeoff()

    gesture_detector = GestureDetector(
        history_size=cfg["gesture"]["history_size"],
        min_votes=cfg["gesture"]["min_votes"],
    )

    command_filter = CommandFilter(
        cooldown=0.2,
        hold_time=0.15,
    )

    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    if not cap.isOpened():
        raise RuntimeError("Could not open USB camera")

    window_name = "Dronosaur Gesture AirSim Bridge V5"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 720, 520)

    prev_time = time.time()
    fps = 0.0
    
    try:
        print("Running modular V5 simulation")
        print("Controls: l=land | q=quit")

        while True:
            ret, frame = cap.read()

            if not ret:
                current_command = "HOVER"
                last_sent_display = airsim_adapter.send_command(current_command)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            raw_gesture, stable_gesture, hand_landmarks, fingers = gesture_detector.process(rgb)

            if hand_landmarks is not None:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                raw_command = gesture_to_command(stable_gesture)

                if raw_command is not None:
                    current_command = command_filter.apply(raw_command)
                    last_valid_gesture_time = time.time()
                else:
                    current_command = command_filter.fallback_hover(
                        no_gesture_timeout,
                        last_valid_gesture_time,
                    )

            else:
                current_command = "HOVER"

            last_sent_display = airsim_adapter.send_command(current_command)

            now = time.time()
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
                cv2.putText(frame, f"Fingers: {fingers}", (20, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, "l=land | q=quit",
                        (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("l"):
                print("Emergency land triggered")
                airsim_adapter.land()
                current_command = "HOVER"

            if key == ord("q"):
                break

    finally:
        print("Cleaning up...")

        airsim_adapter.cleanup()

        cap.release()
        gesture_detector.hands.close()
        cv2.destroyAllWindows()

        print("Finished")


if __name__ == "__main__":
    main()   
