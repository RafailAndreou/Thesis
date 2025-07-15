import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import TopKCategoricalAccuracy

# === CONFIG ===
input_shape = (75, 159, 3)
num_classes = 51
batch_size = 32
epochs = 50
learning_rate = 1e-4
dataset_dir = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\train_dataset"

# === DATA AUGMENTATION ===
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

# === DATA LOADERS ===
train_data = train_gen.flow_from_directory(
    os.path.join(dataset_dir, "train"),
    target_size=(75, 159),
    batch_size=batch_size,
    class_mode='categorical'
)
val_data = val_gen.flow_from_directory(
    os.path.join(dataset_dir, "val"),
    target_size=(75, 159),
    batch_size=batch_size,
    class_mode='categorical'
)
test_data = test_gen.flow_from_directory(
    os.path.join(dataset_dir, "test"),
    target_size=(75, 159),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# === BUILD MODEL ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
for layer in base_model.layers[-30:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# === COMPILE MODEL ===
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ]
)

# === CALLBACKS ===
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint("mobilenetv2_best.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
]

# === TRAIN ===
model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data,
    callbacks=callbacks
)

# === SAVE FINAL MODEL ===
model.save("mobilenetv2_fft_final.h5")

# === EVALUATE ===
loss, acc, top5 = model.evaluate(test_data)
print(f"\n✅ Final Test Accuracy: {acc:.4f}")
print(f"✅ Final Test Top-5 Accuracy: {top5:.4f}")
