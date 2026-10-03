import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

def preprocess():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    test_size = params["preprocess"]["test_size"]
    seed = params["preprocess"]["seed"]

    print("Loading raw data...")
    X_train = np.load("data/raw/X_train.npy")
    y_train = np.load("data/raw/y_train.npy")
    X_test = np.load("data/raw/X_test.npy")
    y_test = np.load("data/raw/y_test.npy")

    print("Normalizing using min-max scaling...")
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

    print(f"Splitting validation set (test_size={test_size}, seed={seed})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=seed
    )

    os.makedirs("data/processed", exist_ok=True)

    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/X_val.npy", X_val)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_test.npy", y_test)

    print(f"Train   : {X_train.shape}")
    print(f"Val     : {X_val.shape}")
    print(f"Test    : {X_test.shape}")
    print("Processed data saved to data/processed/")

if __name__ == "__main__":
    preprocess()
