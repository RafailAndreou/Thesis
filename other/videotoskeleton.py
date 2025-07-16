import cv2
import csv
import mediapipe as mp
from pathlib import Path
from tkinter import Tk, filedialog

def extract_skeleton_from_video(video_path: Path) -> list[list[float]]:
    """
    Extracts 33-joint skeleton frames from a video using MediaPipe.
    Returns a list of frames, each containing 99 float values (x,y,z for 33 joints).
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

    cap = cv2.VideoCapture(str(video_path))
    skeleton_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            joints = results.pose_landmarks.landmark
            frame_data = []
            for joint in joints:  # 33 joints
                frame_data.extend([joint.x, joint.y, joint.z])
            skeleton_frames.append(frame_data)

    cap.release()
    pose.close()
    return skeleton_frames

def save_skeleton_to_csv(skeleton_data: list[list[float]], output_csv_path: Path) -> None:
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(skeleton_data)

# === RUN THIS FILE DIRECTLY ===
if __name__ == "__main__":
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.gif")]
    )
    if not file_path:
        print("❌ No file selected.")
        exit()

    VIDEO_PATH = Path(file_path)
    OUTPUT_CSV = VIDEO_PATH.with_suffix(".skeleton.csv")

    print(f"📥 Processing: {VIDEO_PATH.name}")
    skeleton = extract_skeleton_from_video(VIDEO_PATH)
    save_skeleton_to_csv(skeleton, OUTPUT_CSV)
    print(f"✅ Saved MediaPipe skeleton to: {OUTPUT_CSV}")
