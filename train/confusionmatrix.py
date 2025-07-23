import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# ==== CONFIG ====
val_dir = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\split_dataset\val"
model_path = "resnet50_fft_unfrozen_from_start.h5"
image_size = (224, 224)
batch_size = 16

# ==== LOAD MODEL ====
print("Loading model...")
model = load_model(model_path)

# ==== DATA GENERATOR ====
datagen = ImageDataGenerator(rescale=1.0/255.0)
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# ==== PREDICT ====
print("Predicting on validation set...")
predictions = model.predict(val_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# ==== CONFUSION MATRIX ====
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

# ==== PLOT CONFUSION MATRIX ====
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=False, fmt='g', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Validation Set')
plt.tight_layout()
plt.show()
