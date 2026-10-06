import os
import cv2
import numpy as np
import mediapipe as mp

from landmark_extractor import extract_landmarks


# -----------------------------
# CONFIGURATION
# -----------------------------

ACTIONS = [
    "hello",
    "thank_you",
    "yes",
    "no",
    "help",
    "none"
]

SEQUENCES_PER_ACTION = 15
SEQUENCE_LENGTH = 30

DATA_PATH = os.path.join(
    "data",
    "landmarks"
)


# -----------------------------
# CREATE DATASET FOLDERS
# -----------------------------

for action in ACTIONS:

    action_path = os.path.join(
        DATA_PATH,
        action
    )

    os.makedirs(
        action_path,
        exist_ok=True
    )


# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------

mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    for action in ACTIONS:

        print(
            f"\nCollecting data for: {action}"
        )

        for sequence in range(
            SEQUENCES_PER_ACTION
        ):

            sequence_data = []

            print(
                f"Sequence "
                f"{sequence + 1}/"
                f"{SEQUENCES_PER_ACTION}"
            )

            # Short ready screen
            while True:

                success, frame = cap.read()

                if not success:
                    break

                cv2.putText(
                    frame,
                    f"Action: {action}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Press SPACE to record",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "ISL Data Collection",
                    frame
                )

                key = cv2.waitKey(1)

                if key == 32:  # SPACE
                    break

                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    raise SystemExit

            # Record 30 frames
            for frame_num in range(
                SEQUENCE_LENGTH
            ):

                success, frame = cap.read()

                if not success:
                    break

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                results = holistic.process(
                    rgb_frame
                )

                landmarks = extract_landmarks(
                    results
                )

                sequence_data.append(
                    landmarks
                )

                cv2.putText(
                    frame,
                    f"{action} "
                    f"{frame_num + 1}/"
                    f"{SEQUENCE_LENGTH}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "ISL Data Collection",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    raise SystemExit

            sequence_data = np.array(
                sequence_data
            )

            save_path = os.path.join(
                DATA_PATH,
                action,
                f"{sequence}.npy"
            )

            np.save(
                save_path,
                sequence_data
            )


cap.release()
cv2.destroyAllWindows()

print("\nData collection finished.")