import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ==== CONFIG ====
dataset_root = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\split_dataset2"
image_size = (224, 224)
batch_size = 16
num_classes = 60
epochs = 20
learning_rate = 1e-5
output_model_path = "resnet50_fft_unfrozen_from_start2.h5"

# ==== DATA GENERATORS ====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_root, "train"),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(dataset_root, "val"),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(dataset_root, "test"),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ==== MODEL SETUP ====
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(image_size[0], image_size[1], 3)
)
base_model.trainable = True  # ✅ Unfreeze from the start

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ==== TRAIN FULL MODEL ====
print("\n🚀 Training full model from the start (ResNet + classifier)...")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# ==== EVALUATE ====
loss, acc = model.evaluate(test_generator)
print(f"\n✅ Final Test Accuracy: {acc*100:.2f}%")

# ==== SAVE MODEL ====
model.save(output_model_path)
print(f"\n💾 Model saved to: {output_model_path}")