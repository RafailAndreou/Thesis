from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Define the model architecture
model = keras.Sequential([
    layers.Dense(512, activation="relu", input_shape=(28 * 28,)),  # Add input shape for the first layer
    layers.Dense(10, activation="softmax")  # Output layer with 10 classes
])

# Compile the model
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Preprocess the data
train_images = train_images.reshape((60000, 28 * 28))  # Flatten the training images
train_images = train_images.astype("float32") / 255  # Normalize pixel values to [0, 1]
test_images = test_images.reshape((10000, 28 * 28))  # Flatten the test images
test_images = test_images.astype("float32") / 255  # Normalize pixel values to [0, 1]

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Make predictions on the test images
predictions = model.predict(test_images)

# Find the first 3 incorrect predictions
incorrect_indices = []
for i in range(len(test_images)):
    if predictions[i].argmax() != test_labels[i]:
        incorrect_indices.append(i)
    if len(incorrect_indices) == 3:  # Stop after finding 3 incorrect predictions
        break

# Plot the first 3 incorrect predictions
for idx, incorrect_index in enumerate(incorrect_indices):
    plt.figure(figsize=(4, 4))
    digit = test_images[incorrect_index].reshape(28, 28)  # Reshape the flattened image back to 28x28
    plt.imshow(digit, cmap=plt.cm.binary)  # Display the image in binary (black and white) color map
    plt.title(f"True: {test_labels[incorrect_index]}, Predicted: {predictions[incorrect_index].argmax()}")
    plt.axis("off")
    plt.show()