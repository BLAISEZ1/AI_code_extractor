import os
import cv2
import pytesseract
from PIL import Image
from transformers import pipeline

# Initialize AI classifier
print("Loading AI classifier for code detection...")
classifier = pipeline("text-classification", model="microsoft/codebert-base")

def is_code(text):
    """Use AI model to check if text looks like code."""
    try:
        result = classifier(text[:512])  # Avoid long input
        return result[0]['label'] == 'LABEL_1'
    except:
        return False

def extract_frames(video_path, output_dir, interval=2):
    """Extracts frames every `interval` seconds from the video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    saved_frames = []

    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % int(fps * interval) == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            filename = f"{output_dir}/frame_{int(timestamp)}.jpg"
            cv2.imwrite(filename, frame)
            saved_frames.append((filename, timestamp))
        frame_count += 1

    cap.release()
    return saved_frames

def extract_code_from_frames(frames):
    """Extract and AI-filter code from frames."""
    code_data = []
    for file_path, timestamp in frames:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)

        if text.strip() and is_code(text.strip()):
            code_data.append((timestamp, text.strip()))
    return code_data

def save_code_to_file(code_data, output_file):
    """Save extracted code snippets to file with timestamps."""
    with open(output_file, "w", encoding="utf-8") as f:
        for timestamp, code in code_data:
            f.write(f"\n# Code at {timestamp:.2f} seconds\n")
            f.write(code + "\n")

def process_video(video_path, output_dir="frames", output_file="extracted_code.txt"):
    print("Extracting frames...")
    frames = extract_frames(video_path, output_dir)

    print("Extracting and filtering code...")
    code_data = extract_code_from_frames(frames)

    print("Saving code to file...")
    save_code_to_file(code_data, output_file)

    print(f"\n✅ Done! Extracted code saved to: {output_file}")

if __name__ == "__main__":
    # Set your video path here
    video_path = r"C:\Users\PK.SNOW\Desktop\extract\video\lazarus\1.mp4"
    process_video(video_path)
