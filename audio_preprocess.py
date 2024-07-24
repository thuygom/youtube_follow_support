from pydub import AudioSegment
from transformers import pipeline

# Step 1: Extract audio from video
def extract_audio(video_path, audio_output_path='audio.wav'):
    audio = AudioSegment.from_file(video_path)
    audio.export(audio_output_path, format='wav')
    return audio_output_path

# Step 2: Analyze audio using a pre-trained model
def analyze_audio(audio_path):
    # Load a pre-trained automatic speech recognition model for Korean
    # Example model name: "kresnik/wav2vec2-large-xlsr-korean"
    audio_classifier = pipeline("automatic-speech-recognition", model="kresnik/wav2vec2-large-xlsr-korean")

    # Process the audio file
    result = audio_classifier(audio_path)
    return result['text']

# Main function
def main():
    video_path = 'video.mp4'  # Ensure this path matches the video file path from the video processing script
    audio_path = 'audio.wav'
    
    extract_audio(video_path, audio_output_path=audio_path)
    audio_text = analyze_audio(audio_path)
    
    print("Audio Analysis Results:", audio_text)

if __name__ == "__main__":
    main()
