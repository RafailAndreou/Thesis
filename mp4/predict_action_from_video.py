# predict_action_from_video.py

from ntu_skeleton_extractor import extract_ntu_skeleton_from_video
from fft_transform import csv_to_fft_image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from pathlib import Path
from tkinter import filedialog, Tk

# CONFIG
MODEL_PATH = "resnet50_fft_unfrozen_from_start.h5"
IMAGE_SIZE = (224, 224)

CLASS_MAPPING = {i: f"Action_{i+1:03d}" for i in range(60)}



def predict_action(fft_image_path, model_path=MODEL_PATH, image_size=IMAGE_SIZE):
    """
    Loads an FFT image and predicts the action class using a pre-trained ResNet50 model.
    """
    model = load_model(model_path)
    img = load_img(fft_image_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array = img_array / 255.0

    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = preds[0][predicted_class]

    return predicted_class, confidence, CLASS_MAPPING.get(predicted_class, f"Class_{predicted_class}")


def full_pipeline_predict(video_path):
    """
    Complete pipeline: video → CSV → FFT image → prediction
    """
    print("🎥 Extracting skeleton...")
    csv_path = extract_ntu_skeleton_from_video(video_path, verbose=True)

    print("⚡ Applying FFT transform...")
    fft_image_path = csv_to_fft_image(csv_path, verbose=True)

    print("🤖 Predicting action...")
    class_idx, confidence, class_name = predict_action(fft_image_path)

    print(f"\n✅ Predicted: {class_name} (class {class_idx}) with {confidence*100:.2f}% confidence")
    return class_idx, class_name, confidence


# === Standalone usage ===
if __name__ == "__main__":
    Tk().withdraw()
    selected_video = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.gif")]
    )

    if selected_video:
        full_pipeline_predict(selected_video)
    else:
        print("❌ No file selected.")
