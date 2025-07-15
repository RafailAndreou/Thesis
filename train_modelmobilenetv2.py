import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# === CONFIG ===
input_shape = (75, 159, 3)
num_classes = 51
batch_size = 32
epochs = 20
learning_rate = 1e-4  # Higher than ResNet, since MobileNetV2 is lighter

dataset_dir = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\train_dataset"

# === DATA LOADERS ===
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

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

# Unfreeze top layers if you want to fine-tune
for layer in base_model.layers[-30:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === TRAIN ===
model.fit(train_data,
          epochs=epochs,
          validation_data=val_data)

# === SAVE ===
model.save("mobilenetv2_fft_finetuned.h5")

# === EVALUATE ===
loss, acc = model.evaluate(test_data)
print(f"\n✅ MobileNetV2 Test Accuracy: {acc:.4f}")
