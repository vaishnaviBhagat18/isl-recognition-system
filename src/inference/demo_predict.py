import cv2
import json
import os
import sys
import numpy as np
import mediapipe as mp
import tensorflow as tf


# --------------------------------------------------
# IMPORT LANDMARK EXTRACTOR
# --------------------------------------------------

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "preprocessing"
    )
)

from landmark_extractor import extract_landmarks


# --------------------------------------------------
# LOAD MODEL + LABELS
# --------------------------------------------------

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
    ),
    "r"
) as f:
    ACTIONS = json.load(f)


# --------------------------------------------------
# TRANSLATIONS
# --------------------------------------------------

TRANSLATIONS = {

    "hello": {
        "english": "Hello",
        "marathi": "Namaskar"
    },

    "thank_you": {
        "english": "Thank You",
        "marathi": "Dhanyavaad"
    },

    "yes": {
        "english": "Yes",
        "marathi": "Ho"
    },

    "no": {
        "english": "No",
        "marathi": "Nahi"
    },

    "help": {
        "english": "Help",
        "marathi": "Madat"
    },

    "none": {
        "english": "",
        "marathi": ""
    }
}


SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.55


# --------------------------------------------------
# MEDIAPIPE
# --------------------------------------------------

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


prediction_label = "none"
confidence = 0.0


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:

        success, frame = cap.read()

        if not success:
            break


        # ------------------------------------------
        # DISPLAY CURRENT RESULT
        # ------------------------------------------

        cv2.putText(
            frame,
            "Press SPACE to recognize sign",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        if prediction_label != "none":

            english = TRANSLATIONS[
                prediction_label
            ]["english"]

            marathi = TRANSLATIONS[
                prediction_label
            ]["marathi"]

            cv2.putText(
                frame,
                f"Prediction: {english}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Marathi: {marathi}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )


        cv2.imshow(
            "ISL Recognition V1",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        # ------------------------------------------
        # QUIT
        # ------------------------------------------

        if key == ord("q"):
            break


        # ------------------------------------------
        # RECORD ONE SIGN
        # ------------------------------------------

        if key == 32:  # SPACE

            sequence = []

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


                sequence.append(
                    landmarks
                )


                # Draw landmarks
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


                cv2.putText(
                    frame,
                    f"Recording sign... "
                    f"{frame_num + 1}/{SEQUENCE_LENGTH}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )


                cv2.imshow(
                    "ISL Recognition V1",
                    frame
                )

                cv2.waitKey(1)


            # --------------------------------------
            # PREDICT
            # --------------------------------------

            if len(sequence) == SEQUENCE_LENGTH:

                input_data = np.expand_dims(
                    np.array(sequence),
                    axis=0
                )


                prediction = model.predict(
                    input_data,
                    verbose=0
                )[0]


                index = np.argmax(
                    prediction
                )


                confidence = float(
                    prediction[index]
                )


                predicted_action = ACTIONS[
                    index
                ]


                if confidence >= CONFIDENCE_THRESHOLD:

                    prediction_label = predicted_action

                else:

                    prediction_label = "none"


                print(
                    "\nPrediction:",
                    predicted_action,
                    "| Confidence:",
                    round(confidence, 3)
                )


cap.release()   
cv2.destroyAllWindows()