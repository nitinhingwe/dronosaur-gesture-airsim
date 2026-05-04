import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opened. Try index 1.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        break

    cv2.imshow("USB Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
