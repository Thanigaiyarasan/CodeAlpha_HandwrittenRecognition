import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Load MNIST handwritten digit dataset
print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

# 2. Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Build Neural Network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

# 4. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Train the model
print("\nTraining the model...")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# 6. Evaluate the model
print("\nEvaluating the model...")

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# 7. Save trained model
model.save("handwritten_digit_model.keras")
print("Model saved successfully!")

# 8. Plot training accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.close()

# 9. Predict test images
predictions = model.predict(x_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

# 10. Save confusion matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)
)

display.plot(cmap="Blues", values_format="d")
plt.title("Handwritten Digit Recognition - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# 11. Display classification report
report = classification_report(y_test, y_pred)

print("\nClassification Report:")
print(report)

with open("classification_report.txt", "w") as file:
    file.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n")
    file.write(report)

# 12. Show sample predictions
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    plt.title(f"Predicted: {y_pred[i]}\nActual: {y_test[i]}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.show()

print("\nProject completed successfully!")
print("Check the generated graphs and result files.")