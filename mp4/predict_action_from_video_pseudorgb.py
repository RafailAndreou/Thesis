import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from ntu_skeleton_extractor import extract_ntu_skeleton_from_video

# ==== CONFIG ====
MODEL_PATH = "resnet50_pseudo_rgb_unfrozen.h5"  # trained pseudo-RGB model
IMAGE_SIZE = (224, 224)
CLASS_MAPPING = {i: f"Action_{i+1:03d}" for i in range(60)}

# ==== Load model once ====
print("📦 Loading model...")
MODEL = load_model(MODEL_PATH)

# ==========================================================
# Translation + Scale Normalization
# ==========================================================
def normalize_scale_translation(ntu_data):
    frames = ntu_data.reshape(ntu_data.shape[0], 25, 3)
    normalized = []
    for frame in frames:
        # Center around hip
        hip_center = (frame[12] + frame[16]) / 2
        frame = frame - hip_center

        # Scale normalization using torso length (SpineShoulder - SpineBase)
        torso_length = np.linalg.norm(frame[20] - frame[0])  # joint 20 = SpineShoulder, joint 0 = SpineBase
        if torso_length > 0:
            frame /= torso_length

        normalized.append(frame)
    return np.array(normalized).reshape(ntu_data.shape[0], -1)

# ==========================================================
# Pseudo-RGB conversion
# ==========================================================
def skeleton_to_pseudo_rgb_image(normalized_skeleton, target_shape=(224, 224)):
    x_coords = normalized_skeleton[:, 0::3]
    y_coords = normalized_skeleton[:, 1::3]
    z_coords = normalized_skeleton[:, 2::3]

    def norm(data):
        scaler = MinMaxScaler()
        return (scaler.fit_transform(data) * 255).astype(np.uint8)

    R, G, B = norm(x_coords), norm(y_coords), norm(z_coords)
    pseudo_rgb = np.stack([R, G, B], axis=-1)
    return Image.fromarray(pseudo_rgb).resize(target_shape, Image.BILINEAR)

# ==========================================================
# Prediction (Top-5)
# ==========================================================
def predict_action_pseudo_rgb(img, top_k=5):
    img_array = img_to_array(img.resize(IMAGE_SIZE))
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    preds = MODEL.predict(img_array)[0]
    top_indices = preds.argsort()[-top_k:][::-1]
    return [(CLASS_MAPPING.get(i, f"Class_{i}"), float(preds[i])) for i in top_indices]

# ==========================================================
# Full pipeline
# ==========================================================
def full_pipeline_predict(video_path):
    print("🎥 Extracting NTU skeleton...")
    csv_path = extract_ntu_skeleton_from_video(video_path, verbose=True)

    print("📏 Loading and normalizing (scale + translation)...")
    df = pd.read_csv(csv_path, header=None)
    data = df.to_numpy()  # direct NTU 25 joints
    normalized = normalize_scale_translation(data)

    print("🎨 Creating pseudo-RGB image...")
    img = skeleton_to_pseudo_rgb_image(normalized)

    print("🤖 Predicting action (Top 5)...")
    predictions = predict_action_pseudo_rgb(img)

    print("\n=== Top 5 Predictions ===")
    for i, (cls, conf) in enumerate(predictions, start=1):
        print(f"{i}. {cls} - {conf*100:.2f}%")
    return predictions

# === Standalone ===
if __name__ == "__main__":
    from tkinter import filedialog, Tk
    Tk().withdraw()
    selected_video = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.gif")]
    )
    if selected_video:
        full_pipeline_predict(selected_video)
    else:
        print("❌ No file selected.")
