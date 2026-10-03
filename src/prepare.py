import os
import numpy as np
import tensorflow as tf

def prepare():
    print("Loading Fashion-MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    os.makedirs("data/raw", exist_ok=True)

    np.save("data/raw/X_train.npy", X_train)
    np.save("data/raw/y_train.npy", y_train)
    np.save("data/raw/X_test.npy", X_test)
    np.save("data/raw/y_test.npy", y_test)

    print(f"Train images : {X_train.shape}")
    print(f"Train labels : {y_train.shape}")
    print(f"Test images  : {X_test.shape}")
    print(f"Test labels  : {y_test.shape}")
    print("Raw data saved to data/raw/")

if __name__ == "__main__":
    prepare()
