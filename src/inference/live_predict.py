import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf

import sys
import os


sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "preprocessing"
    )
)

from landmark_extractor import extract_landmarks


# -----------------------------
# LOAD MODEL
# -----------------------------

model = tf.keras.models.load_model(
    os.path.join(
        "models",
        "isl_v1.keras"
    )
)

with open(
    os.path.join(
        "models",
        "labels.json"
    )
) as f:

    ACTIONS = json.load(f)


SEQUENCE_LENGTH = 30

sequence = []

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:

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

        sequence.append(
            landmarks
        )

        sequence = sequence[
            -SEQUENCE_LENGTH:
        ]


        prediction_text = "Collecting..."

        confidence = 0.0


        if len(sequence) == SEQUENCE_LENGTH:

            input_data = np.expand_dims(
                sequence,
                axis=0
            )

            prediction = model.predict(
                input_data,
                verbose=0
            )[0]

            index = np.argmax(
                prediction
            )

            confidence = prediction[
                index
            ]

            if confidence > 0.75:

                prediction_text = ACTIONS[
                    index
                ]

            else:

                prediction_text = "uncertain"


        cv2.putText(
            frame,
            f"Prediction: {prediction_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )


        cv2.imshow(
            "ISL Recognition V1",
            frame
        )


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()