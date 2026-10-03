import os
import numpy as np
import yaml
import pandas as pd
import tensorflow as tf

def train():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    dense_units   = params["train"]["dense_units"]
    dropout_rate  = params["train"]["dropout_rate"]
    learning_rate = params["train"]["learning_rate"]
    epochs        = params["train"]["epochs"]
    batch_size    = params["train"]["batch_size"]

    print("Loading processed data...")
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_val   = np.load("data/processed/X_val.npy")
    y_val   = np.load("data/processed/y_val.npy")

    print("Building ANN model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    print(f"Training for {epochs} epochs, batch_size={batch_size}...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )

    os.makedirs("models", exist_ok=True)

    model.save("models/model.h5")
    print("Model saved to models/model.h5")

    pd.DataFrame(history.history).to_csv("models/history.csv", index=False)
    print("Training history saved to models/history.csv")

if __name__ == "__main__":
    train()
