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

# ==========================================================
# Map MediaPipe (33 joints) -> NTU (25 joints)
# ==========================================================
def mediapipe_to_ntu_25(frame):
    joints = frame.reshape(-1, 3)
    def mid(a, b): return (joints[a] + joints[b]) / 2
    ntu = np.zeros((25, 3))

    ntu[0]  = mid(23, 24)    # SpineBase
    ntu[20] = mid(11, 12)    # SpineShoulder
    ntu[1]  = mid(ntu[0], ntu[20])  # SpineMid
    ntu[2]  = ntu[20]        # Neck
    ntu[3]  = joints[0]      # Head (nose)
    ntu[4]  = joints[11]     # LeftShoulder
    ntu[5]  = joints[13]     # LeftElbow
    ntu[6]  = joints[15]     # LeftWrist
    ntu[7]  = joints[17]     # LeftHand
    ntu[8]  = joints[12]     # RightShoulder
    ntu[9]  = joints[14]     # RightElbow
    ntu[10] = joints[16]     # RightWrist
    ntu[11] = joints[18]     # RightHand
    ntu[12] = joints[23]     # LeftHip
    ntu[13] = joints[25]     # LeftKnee
    ntu[14] = joints[27]     # LeftAnkle
    ntu[15] = joints[31]     # LeftFoot
    ntu[16] = joints[24]     # RightHip
    ntu[17] = joints[26]     # RightKnee
    ntu[18] = joints[28]     # RightAnkle
    ntu[19] = joints[32]     # RightFoot
    ntu[21] = joints[19]     # LeftHandTip
    ntu[22] = joints[21]     # LeftThumb
    ntu[23] = joints[20]     # RightHandTip
    ntu[24] = joints[22]     # RightThumb

    return ntu.reshape(-1)

def convert_all_frames_to_ntu_25(data):
    return np.array([mediapipe_to_ntu_25(frame) for frame in data.reshape(data.shape[0], -1)])

# ==========================================================
# Translation-only normalization
# ==========================================================
def normalize_translation_only(ntu_data):
    frames = ntu_data.reshape(ntu_data.shape[0], 25, 3)
    normalized = []
    for frame in frames:
        hip_center = (frame[12] + frame[16]) / 2  # LeftHip + RightHip
        frame = frame - hip_center
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
# Prediction
# ==========================================================
def predict_action_pseudo_rgb(img, model_path=MODEL_PATH, top_k=5):
    model = load_model(model_path)
    img_array = img_to_array(img.resize(IMAGE_SIZE))
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    preds = model.predict(img_array)[0]
    top_indices = preds.argsort()[-top_k:][::-1]
    return [(CLASS_MAPPING.get(i, f"Class_{i}"), float(preds[i])) for i in top_indices]

# ==========================================================
# Full pipeline
# ==========================================================
def full_pipeline_predict(video_path):
    print("🎥 Extracting skeleton...")
    csv_path = extract_ntu_skeleton_from_video(video_path, verbose=True)

    print("🔄 Converting MediaPipe → NTU 25...")
    df = pd.read_csv(csv_path, header=None)
    data = df.iloc[:, 1:].to_numpy()   # <-- FIXED: skip frame index
    ntu_data = convert_all_frames_to_ntu_25(data)

    print("📏 Translation-only normalization...")
    normalized = normalize_translation_only(ntu_data)

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
