import os
import numpy as np


DATA_PATH = os.path.join(
    "data",
    "landmarks"
)

ACTIONS = [
    "hello",
    "thank_you",
    "yes",
    "no",
    "help",
    "none"
]


for action in ACTIONS:

    folder = os.path.join(
        DATA_PATH,
        action
    )

    files = os.listdir(folder)

    print(
        f"\n{action}: {len(files)} samples"
    )

    if files:

        sample = np.load(
            os.path.join(
                folder,
                files[0]
            )
        )

        print(
            "Sample shape:",
            sample.shape
        )