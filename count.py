import os

folder_path = r"C:\Users\rafai\Desktop\Programs\Python\Ptyxiaki\πτυχιακή\action_recognition_code+dataset\action_recognition_code+dataset\datasets\ut\SkeletonData"

file_count = sum([len(files) for _, _, files in os.walk(folder_path)])

print(f"Total files in folder M: {file_count}")
