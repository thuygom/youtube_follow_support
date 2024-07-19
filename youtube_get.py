import yt_dlp
import cv2

def download_video(url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'outtmpl': 'video.mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def extract_frames(video_file):
    frames = []
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def analyze_frames(frames):

    num_frames = len(frames)
    print(f"Total frames: {num_frames}")

def main():
    video_url = 'https://youtube.com/watch?v=l9c4JeN5NE8'
    download_video(video_url)
    frames = extract_frames('video.mp4')
    analyze_frames(frames)

if __name__ == "__main__":
    main()
