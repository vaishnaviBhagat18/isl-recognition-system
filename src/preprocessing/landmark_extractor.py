import numpy as np


def extract_landmarks(results):
    """
    Convert MediaPipe Holistic results into a fixed-size feature vector.

    Left hand:   21 × 3 = 63
    Right hand:  21 × 3 = 63
    Pose:        33 × 4 = 132

    Total: 258 features per frame
    """

    # Left hand
    if results.left_hand_landmarks:
        left_hand = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.left_hand_landmarks.landmark
        ]).flatten()
    else:
        left_hand = np.zeros(63)

    # Right hand
    if results.right_hand_landmarks:
        right_hand = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.right_hand_landmarks.landmark
        ]).flatten()
    else:
        right_hand = np.zeros(63)

    # Pose
    if results.pose_landmarks:
        pose = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ]).flatten()
    else:
        pose = np.zeros(132)

    return np.concatenate([
        left_hand,
        right_hand,
        pose
    ])