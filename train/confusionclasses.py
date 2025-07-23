import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ==== CONFIG ====
model_path = r"resnet50_fft_unfrozen_from_start.h5"  # <-- change to your saved model
val_dir = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\Rafail_dataset\ntu\split_dataset\val"
image_size = (224, 224)
batch_size = 16

# ==== LOAD MODEL ====
print(f"Loading model from {model_path} ...")
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

# ==== PREDICTIONS ====
print("Predicting on validation set...")
predictions = model.predict(val_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# ==== CLASSIFICATION REPORT ====
print("\nClassification Report:")
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

# ==== CONFUSION MATRIX ====
cm = confusion_matrix(y_true, y_pred)

# ---- Top 5 Confusions ----
cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)
top_confusions = []
for _ in range(5):
    i, j = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
    top_confusions.append((class_labels[i], class_labels[j], cm_copy[i, j]))
    cm_copy[i, j] = 0

print("\nTop 5 most confused class pairs:")
for true_class, pred_class, count in top_confusions:
    print(f"True: {true_class} → Predicted: {pred_class} ({count} samples)")

# ---- Plot Normalized Confusion Matrix ----
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(20, 20))
sns.heatmap(cm_norm, annot=False, cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Normalized Confusion Matrix - Validation Set')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png")
print("\nConfusion matrix saved to confusion_matrix_normalized.png")
