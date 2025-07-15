from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

# === CONFIGURATION ===
BASE_DIR = Path(r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\nku")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
OUTPUT_MODEL = BASE_DIR / "resnet_transfer_model.h5"

# === LOAD DATASETS ===
def load_rgb_datasets(base_dir, image_size, batch_size):
    # Train
    train_raw = tf.keras.utils.image_dataset_from_directory(
        base_dir / "train",
        label_mode="int",
        color_mode="grayscale",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )
    class_names = train_raw.class_names
    train_ds = train_raw.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))

    # Val
    val_raw = tf.keras.utils.image_dataset_from_directory(
        base_dir / "val",
        label_mode="int",
        color_mode="grayscale",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )
    val_ds = val_raw.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))

    # Test
    test_raw = tf.keras.utils.image_dataset_from_directory(
        base_dir / "test",
        label_mode="int",
        color_mode="grayscale",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )
    test_ds = test_raw.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))

    return train_ds, val_ds, test_ds, class_names

# === BUILD MODEL ===
def build_transfer_model(input_shape, num_classes):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False  # Freeze pretrained weights

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# === TRAINING PIPELINE ===
train_ds, val_ds, test_ds, class_names = load_rgb_datasets(BASE_DIR, IMAGE_SIZE, BATCH_SIZE)
num_classes = len(class_names)

model = build_transfer_model(IMAGE_SIZE + (3,), num_classes)
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# === SAVE MODEL ===
model.save(OUTPUT_MODEL)
print(f"🧠 Model saved to: {OUTPUT_MODEL}")

# === EVALUATE ON TEST SET ===
test_loss, test_acc = model.evaluate(test_ds)
print(f"✅ Final Test Accuracy: {test_acc:.4f}")
