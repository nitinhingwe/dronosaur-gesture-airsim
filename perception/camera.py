import subprocess
import threading
import time

import cv2
import numpy as np


class RaspberryPiCamera:
    """
    Raspberry Pi Camera Module backend using rpicam-vid.

    Designed for:
    - Raspberry Pi 5
    - Camera Module 3 / 3 Wide
    - OpenCV / MediaPipe pipeline

    Does not require Picamera2 Python package.
    """

    def __init__(
        self,
        width=480,
        height=360,
        fps=30,
    ):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.process = None
        self.thread = None

        self.running = False

        self.latest_frame = None
        self.lock = threading.Lock()

    def start(self):

        if self.running:
            return

        command = [
            "rpicam-vid",
            "--width", str(self.width),
            "--height", str(self.height),
            "--framerate", str(self.fps),

            "--codec", "mjpeg",

            "--timeout", "0",

            "--nopreview",

            "--output", "-"
        ]

        print("Starting Raspberry Pi Camera Module...")

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

        self.running = True

        self.thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
        )

        self.thread.start()

        # Give camera exposure/AWB a moment
        time.sleep(1.0)

        print("Raspberry Pi Camera started.")

    def _reader_loop(self):

        buffer = bytearray()

        while self.running:

            if self.process is None:
                break

            chunk = self.process.stdout.read(4096)

            if not chunk:
                time.sleep(0.005)
                continue

            buffer.extend(chunk)

            while True:

                start = buffer.find(b"\xff\xd8")

                if start == -1:
                    break

                end = buffer.find(
                    b"\xff\xd9",
                    start + 2,
                )

                if end == -1:
                    break

                jpeg = bytes(
                    buffer[start:end + 2]
                )

                del buffer[:end + 2]

                image_array = np.frombuffer(
                    jpeg,
                    dtype=np.uint8,
                )

                frame = cv2.imdecode(
                    image_array,
                    cv2.IMREAD_COLOR,
                )

                if frame is None:
                    continue

                with self.lock:
                    self.latest_frame = frame

    def read(self):

        with self.lock:

            if self.latest_frame is None:
                return False, None

            return True, self.latest_frame.copy()

    def stop(self):

        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=1)

        if self.process is not None:

            try:
                self.process.terminate()
                self.process.wait(timeout=2)

            except Exception:

                try:
                    self.process.kill()
                except Exception:
                    pass

        self.process = None
        self.thread = None

        print("Raspberry Pi Camera stopped.")
