import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

print("Webcam opened successfully.")
print("Press Q to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read frame.")
        break

    cv2.imshow("ISL Project - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
