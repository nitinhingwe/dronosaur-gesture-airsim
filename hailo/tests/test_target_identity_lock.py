import cv2
import numpy as np

# ?? Set correct path manually
CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade from: {CASCADE_PATH}")

cap = cv2.VideoCapture(0)

target_face = None
tracking_enabled = False


def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face, (100, 100))
    return face, (x, y, w, h)


def compare_faces(f1, f2):
    if f1 is None or f2 is None:
        return False

    diff = np.mean((f1 - f2) ** 2)

    return diff < 2000  # tune later


while True:
    ret, frame = cap.read()
    if not ret:
        break

    face, box = extract_face(frame)

    if face is not None and box is not None:
        x, y, w, h = box

        if tracking_enabled and target_face is not None:
            if compare_faces(face, target_face):
                color = (0, 255, 0)
                label = "TARGET LOCKED"
            else:
                color = (0, 0, 255)
                label = "NOT TARGET"
        else:
            color = (255, 255, 0)
            label = "FACE DETECTED"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, "E = Enroll | T = Track | R = Reset",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Target Identity Lock", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('e') and face is not None:
        target_face = face.copy()
        print("? Target face enrolled")

    elif key == ord('t'):
        tracking_enabled = True
        print("?? Tracking enabled")

    elif key == ord('r'):
        target_face = None
        tracking_enabled = False
        print("?? Reset")

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
