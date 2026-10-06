import os
import json
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# -----------------------------
# CONFIG
# -----------------------------

DATA_PATH = os.path.join(
    "data",
    "landmarks"
)

MODEL_PATH = os.path.join(
    "models",
    "isl_v1.keras"
)

LABEL_PATH = os.path.join(
    "models",
    "labels.json"
)

ACTIONS = [
    "hello",
    "thank_you",
    "yes",
    "no",
    "help",
    "none"
]


# -----------------------------
# LOAD DATA
# -----------------------------

X = []
y = []

for label_index, action in enumerate(ACTIONS):

    action_folder = os.path.join(
        DATA_PATH,
        action
    )

    for file_name in os.listdir(
        action_folder
    ):

        if not file_name.endswith(".npy"):
            continue

        sequence = np.load(
            os.path.join(
                action_folder,
                file_name
            )
        )

        if sequence.shape == (30, 258):

            X.append(sequence)
            y.append(label_index)


X = np.array(X)
y = np.array(y)


print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------
# MODEL
# -----------------------------

model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(30, 258)
    ),

    Dropout(0.2),

    LSTM(
        64
    ),

    Dropout(0.2),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        len(ACTIONS),
        activation="softmax"
    )
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# -----------------------------
# TRAIN
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=40,
    validation_split=0.20,
    batch_size=8
)


# -----------------------------
# TEST
# -----------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(
    f"\nTest accuracy: {accuracy:.2%}"
)


# -----------------------------
# SAVE
# -----------------------------

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    MODEL_PATH
)

with open(
    LABEL_PATH,
    "w"
) as f:

    json.dump(
        ACTIONS,
        f
    )


print(
    "\nModel saved:",
    MODEL_PATH
)