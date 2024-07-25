import os
import cv2
import yt_dlp
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

# Step 1: Download video using yt_dlp
def download_video(url, output_path='video.mp4'):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Step 2: Extract frames using OpenCV
def extract_frames(video_file, frame_rate=1):
    cap = cv2.VideoCapture(video_file)
    frame_list = []
    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (fps // frame_rate) == 0:
            frame_list.append(frame)

        frame_count += 1

    cap.release()
    return frame_list

# Step 3: Preprocess each frame image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert OpenCV image (BGR) to PIL image (RGB)
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# Step 4: Analyze frames (example with preprocessing)
def analyze_frames(frames):
    preprocessed_frames = [preprocess_image(frame) for frame in frames]
    num_frames = len(preprocessed_frames)
    print(f"Total frames: {num_frames}")
    print(f"Preprocessed frames shape: {preprocessed_frames[0].shape if num_frames > 0 else 'No frames'}")

def save_frames_as_images(frames, output_dir='frames'):
    """프레임을 이미지 파일로 저장합니다."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, frame in enumerate(frames):
        image_filename = os.path.join(output_dir, f'frame_{i:04d}.png')
        cv2.imwrite(image_filename, frame)
        print(f'Saved {image_filename}')

# Main function
def main():
    video_url = 'https://www.youtube.com/watch?v=hs_uw_o0fT4'
    download_video(video_url)
    
    frames = extract_frames('video.mp4')
    save_frames_as_images(frames)  # Save frames as images
    analyze_frames(frames)

if __name__ == "__main__":
    main()
