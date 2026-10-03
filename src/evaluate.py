import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate():
    print("Loading model...")
    model = tf.keras.models.load_model("models/model.h5")

    print("Loading test data...")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"Test Loss     : {loss:.4f}")
    print(f"Test Accuracy : {accuracy:.4f}")

    metrics = {"test_loss": round(loss, 4), "test_accuracy": round(accuracy, 4)}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to metrics.json")

    print("Generating confusion matrix...")
    y_pred = np.argmax(model.predict(X_test), axis=1)

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix — Fashion-MNIST ANN")
    plt.tight_layout()

    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/confusion_matrix.png")
    print("Confusion matrix saved to reports/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
