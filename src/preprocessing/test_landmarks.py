import cv2
import mediapipe as mp
import numpy as np

from hand_landmarks import extract_hand_landmarks


mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:

        success, frame = cap.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)
        
        if results.multi_handedness:
            labels = [
                hand.classification[0].label
                for hand in results.multi_handedness
            ]

            print("MediaPipe detected:", labels)

        landmarks = extract_hand_landmarks(results)

        left_hand = landmarks[:63]
        right_hand = landmarks[63:]

        print(
            "Shape:", landmarks.shape,
            "| Left non-zero:", np.count_nonzero(left_hand),
            "| Right non-zero:", np.count_nonzero(right_hand)
        )

        cv2.imshow(
            "ISL - Landmark Extraction Test",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()