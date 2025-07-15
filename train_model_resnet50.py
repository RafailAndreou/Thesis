import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# === CONFIG ===
input_shape = (75, 159, 3)
num_classes = 51
batch_size = 32
epochs = 20
learning_rate = 1e-5

# ✅ NEW DATASET PATH
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
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

# UNFREEZE the top N layers
for layer in base_model.layers[-50:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
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
model.save("resnet50_fft_finetuned.h5")

# === EVALUATE ON TEST SET ===
loss, acc = model.evaluate(test_data)
print(f"\n✅ Fine-tuned Test accuracy: {acc:.4f}")
