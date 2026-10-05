import numpy as np


def extract_hand_landmarks(results):
    """
    Convert MediaPipe hand results into a fixed-length NumPy array.

    Output:
        126 values
        = 63 values for left hand
        + 63 values for right hand
    """

    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            # Convert 21 (x, y, z) landmarks into 63 numbers
            landmarks = np.array([
                [landmark.x, landmark.y, landmark.z]
                for landmark in hand_landmarks.landmark
            ]).flatten()

            hand_label = handedness.classification[0].label

            if hand_label == "Left":
                left_hand = landmarks

            elif hand_label == "Right":
                right_hand = landmarks

    return np.concatenate([left_hand, right_hand])