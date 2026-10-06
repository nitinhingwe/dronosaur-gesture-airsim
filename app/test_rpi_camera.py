import sys
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))


from perception.camera import RaspberryPiCamera


def main():

    print("================================")
    print(" DRONOSAUR RPI CAMERA TEST")
    print(" Camera Module 3 Wide")
    print("================================")

    camera = RaspberryPiCamera(
        width=480,
        height=360,
        fps=30,
    )

    try:

        camera.start()

        print("Press q to quit.")

        while True:

            ok, frame = camera.read()

            if not ok:
                continue

            cv2.putText(
                frame,
                "Dronosaur Camera Module 3 Wide",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "Dronosaur RPi Camera",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
