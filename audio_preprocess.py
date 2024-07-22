from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

# Step 1: Extract audio from video
def extract_audio(video_path, audio_output_path='audio.wav'):
    audio = AudioSegment.from_file(video_path)
    audio.export(audio_output_path, format='wav')
    return audio_output_path

# Step 2: Analyze audio using a pre-trained model
def analyze_audio(audio_path):
    # Load the processor and model for multilingual support (including Korean)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.squeeze()

    # Resample if the sample rate is not 16000 Hz
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Tokenize the audio
    input_values = processor(waveform, sampling_rate=16000, return_tensors="pt", padding="longest").input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the predicted tokens into text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Main function
def main():
    video_path = 'video.mp4'  # Ensure this path matches the video file path from the video processing script
    audio_path = 'audio.wav'
    
    extract_audio(video_path, audio_output_path=audio_path)
    audio_text = analyze_audio(audio_path)
    
    print("Audio Analysis Results:", audio_text)

if __name__ == "__main__":
    main()
