# full_pipeline.py
from video_to_csv import extract_skeleton_from_video
from visualize_csv import load_skeleton_from_csv, visualize_skeleton
import cv2
from tkinter import Tk, filedialog

def extract_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

if __name__ == "__main__":
    Tk().withdraw()
    video_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.gif")]
    )

    if video_path:
        csv_path = extract_skeleton_from_video(video_path)
        skeleton_frames, frame_indices = load_skeleton_from_csv(csv_path)
        video_frames = extract_video_frames(video_path)

        visualize_skeleton(skeleton_frames, frame_indices, video_frames)
    else:
        print("❌ No file selected.")
