import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opened. Try camera index 1 or check USB connection.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("USB Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
