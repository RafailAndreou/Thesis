# full_pipeline.py
from video_to_csv import extract_skeleton_from_video
from visualize_csv import visualize_csv
from tkinter import Tk, filedialog

if __name__ == "__main__":
    Tk().withdraw()
    video_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.gif")]
    )
    if not video_path:
        print("❌ No file selected.")
    else:
        csv_path = extract_skeleton_from_video(video_path)
        visualize_csv(csv_path)
