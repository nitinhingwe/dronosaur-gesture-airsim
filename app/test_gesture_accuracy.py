import sys
from pathlib import Path

import cv2
import yaml
import mediapipe as mp


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))


from perception.camera import RaspberryPiCamera
from perception.gesture_detector import GestureDetector


with open(
    BASE_DIR / "config" / "real.yaml",
    "r"
) as f:

    cfg = yaml.safe_load(f)


def main():

    camera = RaspberryPiCamera(
        width=cfg["camera"]["width"],
        height=cfg["camera"]["height"],
        fps=cfg["camera"]["fps"],
    )

    detector = GestureDetector(
        history_size=cfg["gesture"]["history_size"],
        min_votes=cfg["gesture"]["min_votes"],
    )

    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    try:

        camera.start()

        print("==============================")
        print(" DRONOSAUR GESTURE TEST")
        print(" No Pixhawk commands")
        print("==============================")
        print("q = quit")

        while True:

            ok, frame = camera.read()

            if not ok:
                continue

            frame = cv2.flip(
                frame,
                1
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            (
                raw,
                stable,
                landmarks,
                fingers,
            ) = detector.process(rgb)

            if landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

            cv2.putText(
                frame,
                f"RAW: {raw}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"STABLE: {stable}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 255),
                2,
            )

            cv2.putText(
                frame,
                f"FINGERS: {fingers}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

            cv2.imshow(
                "Dronosaur Gesture Accuracy",
                frame,
            )

            if (
                cv2.waitKey(1) & 0xFF
                ==
                ord("q")
            ):
                break

    finally:

        camera.stop()

        detector.hands.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
