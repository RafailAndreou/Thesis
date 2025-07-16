# 🎥 Skeleton Extraction & Visualization Pipeline

This mini pipeline provides tools to:

1. Extract 3D skeleton data from a video using **MediaPipe** and save it to CSV.
2. Visualize the extracted skeleton (alone or side-by-side with the original video).
3. Run the entire pipeline from video to visualization in one go.

---

## 📦 Scripts Overview

| Script             | Purpose                                                                 |
| ------------------ | ----------------------------------------------------------------------- |
| `video_to_csv.py`  | Extracts pose landmarks from video using MediaPipe and saves to CSV.    |
| `visualize_csv.py` | Loads a `.skeleton.csv` and visualizes the skeleton with animation.     |
| `full_pipeline.py` | Full pipeline: extract skeleton CSV + visualize alongside video frames. |

---

## 🔧 Requirements

Install dependencies using:

```bash
pip install opencv-python mediapipe numpy matplotlib
```

---

## 📂 Output Format

### Skeleton CSV (e.g. `myvideo.skeleton.csv`):

Each row corresponds to a single frame with 3D pose landmarks:

```
frame_idx, x0, y0, z0, x1, y1, z1, ..., x32, y32, z32
```

- 33 joints per frame (MediaPipe Pose format).
- Stored in `.csv` file with no header.

---

## ▶️ Usage

### 1. Run Full Pipeline

```bash
python full_pipeline.py
```

- Opens a file dialog to select a video.
- Saves skeleton CSV.
- Visualizes both video and skeleton side-by-side.

### 2. Run Individually

#### Extract skeleton only:

```bash
python video_to_csv.py
```

Then select a video file.

#### Visualize from CSV:

```bash
python visualize_csv.py
```

Then select a `.skeleton.csv` file.

---

## 🖼️ Visualization Details

- Uses **Matplotlib** animation.
- Joints are color-coded:
  - 🟦 Left side: blue/green
  - 🟥 Right side: red/orange
  - 🟫 Face: gray
- Displays joints connected by MediaPipe's standard skeleton structure.

---

## 📁 Example Output

- `myvideo.mp4` → `myvideo.skeleton.csv`
- Visualization displays:
  - Original frame (left)
  - Centered skeleton (right)

---

## 🧠 Use Case

This tool is ideal for:

- Previewing skeleton extraction quality.
- Creating datasets for human pose-based tasks.
- Validating MediaPipe pose tracking before transformation (e.g., FFT, DCT).

---

## ✨ Integration in Thesis

You can connect the output `.skeleton.csv` files from this tool to your existing FFT image generation and training pipeline found in:

```
image_transformation_code/FFT/
```

---

## 🧪 Next Steps

- Transform CSVs into images using FFT: `fft_image.py`
- Use TVT split: `fft_crossview_TVT_directories.py` or `fft_crossubject_TVT_directories.py`
- Train with CNN: `train/fourie_images/`

---

## 📝 License

This code is built on top of [MediaPipe](https://github.com/google/mediapipe) and is intended for academic and educational use.
