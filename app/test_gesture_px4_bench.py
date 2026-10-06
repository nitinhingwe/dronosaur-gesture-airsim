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
from adapters.px4_offboard_adapter import PX4OffboardAdapter
from perception.camera import RaspberryPiCamera


with open(BASE_DIR / "config" / "real.yaml", "r") as f:
    cfg = yaml.safe_load(f)

def main():

    print("================================================")
    print(" DRONOSAUR GESTURE -> PX4 BENCH INTEGRATION")
    print(" PROPS OFF | DISARMED | NO OFFBOARD MODE CHANGE")
    print("================================================")

    # -------------------------------------------------
    # HARD SAFETY GATES
    # -------------------------------------------------

    if cfg["safety"].get("bench_read_only") is not True:
        raise RuntimeError(
            "SAFETY STOP: bench_read_only must be true."
        )

    if cfg["safety"]["allow_arm"] is not False:
        raise RuntimeError(
            "SAFETY STOP: allow_arm must be false."
        )

    if cfg["safety"]["allow_takeoff"] is not False:
        raise RuntimeError(
            "SAFETY STOP: allow_takeoff must be false."
        )

    # -------------------------------------------------
    # PX4
    # -------------------------------------------------

    adapter = PX4OffboardAdapter(
        device=cfg["connection"]["device"],
        baud=cfg["connection"]["baud"],
        send_rate_hz=10,
    )

    # -------------------------------------------------
    # PERCEPTION / DECISION
    # -------------------------------------------------

    gesture_detector = GestureDetector(
        history_size=cfg["gesture"]["history_size"],
        min_votes=cfg["gesture"]["min_votes"],
    )

    command_filter = CommandFilter(
        cooldown=cfg["gesture"]["command_cooldown"],
        hold_time=0.15,
    )

    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # -------------------------------------------------
    # RASPBERRY PI CAMERA MODULE 3 WIDE
    # -------------------------------------------------

    camera = RaspberryPiCamera(
        width=cfg["camera"]["width"],
        height=cfg["camera"]["height"],
        fps=cfg["camera"]["fps"],
    )

    # -------------------------------------------------
    # STATE
    # -------------------------------------------------

    current_command = "HOVER"

    last_valid_gesture_time = time.time()

    no_gesture_timeout = cfg["gesture"]["no_gesture_timeout"]

    last_display_command = None

    prev_time = time.time()
    fps = 0.0

    window_name = "Dronosaur Gesture -> PX4 BENCH"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window_name,
        720,
        520,
    )

    try:

        # ---------------------------------------------
        # START CAMERA
        # ---------------------------------------------

        print("Starting Raspberry Pi Camera Module 3 Wide...")
        camera.start()
        
        # ---------------------------------------------
        # CONNECT PIXHAWK
        # ---------------------------------------------       
        
        adapter.connect()

        print()
        print(f"PX4 armed: {adapter.is_armed()}")

        if adapter.is_armed():
            raise RuntimeError(
                "SAFETY STOP: Pixhawk is already armed."
            )

        print()
        print("Starting PX4 setpoint stream at 10 Hz.")
        print("NO mode change will be requested.")
        print("NO arm command exists in this test.")
        print("NO takeoff command exists in this test.")
        print("Press q to quit.")
        print()

        # Start from guaranteed zero
        adapter.set_zero()
        adapter.start_stream()

        # ---------------------------------------------
        # MAIN LOOP
        # ---------------------------------------------

        while True:

            ret, frame = camera.read()

            if not ret:

                current_command = "HOVER"

                adapter.set_command(
                    current_command,
                    cfg["velocity"],
                )

                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            (
                raw_gesture,
                stable_gesture,
                hand_landmarks,
                fingers,
            ) = gesture_detector.process(rgb)

            # -----------------------------------------
            # GESTURE -> COMMAND
            # -----------------------------------------

            if hand_landmarks is not None:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

                raw_command = gesture_to_command(
                    stable_gesture
                )

                if raw_command is not None:

                    current_command = command_filter.apply(
                        raw_command
                    )

                    last_valid_gesture_time = time.time()

                else:

                    current_command = (
                        command_filter.fallback_hover(
                            no_gesture_timeout,
                            last_valid_gesture_time,
                        )
                    )

            else:

                # Important Dronosaur behavior:
                # no hand = HOVER
                current_command = "HOVER"

            # -----------------------------------------
            # COMMAND -> PX4 SETPOINT
            # -----------------------------------------

            result = adapter.set_command(
                current_command,
                cfg["velocity"],
            )

            # Only print when command changes
            if current_command != last_display_command:

                print(
                    f"{current_command:10} -> "
                    f"vx={result['vx']:+.2f}  "
                    f"vy={result['vy']:+.2f}  "
                    f"vz={result['vz']:+.2f}  "
                    f"yaw={result['yaw_rate_rad_s']:+.3f}"
                )

                last_display_command = current_command

            # -----------------------------------------
            # FPS
            # -----------------------------------------

            now = time.time()

            dt = now - prev_time
            prev_time = now

            if dt > 0:
                fps = (
                    0.9 * fps
                    + 0.1 * (1.0 / dt)
                )

            # -----------------------------------------
            # DISPLAY
            # -----------------------------------------

            cv2.putText(
                frame,
                f"Raw: {raw_gesture}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Stable: {stable_gesture}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"PX4 Cmd: {current_command}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                (
                    f"Setpoint: "
                    f"{result['vx']:+.2f}, "
                    f"{result['vy']:+.2f}, "
                    f"{result['vz']:+.2f}"
                ),
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (180, 255, 180),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if fingers is not None:

                cv2.putText(
                    frame,
                    f"Fingers: {fingers}",
                    (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                "BENCH ONLY | DISARMED | q=quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                window_name,
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        print()
        print("Cleaning up gesture/PX4 bench test...")

        # Force zero before stopping transmission
        try:
            adapter.set_zero()
            time.sleep(0.3)
        except Exception:
            pass

        adapter.cleanup()

        try:
            camera.stop()
        except Exception:
            pass       

        gesture_detector.hands.close()

        cv2.destroyAllWindows()

        print("PX4 final command: ZERO")
        print("Finished.")


if __name__ == "__main__":
    main()
