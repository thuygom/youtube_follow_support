from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment
import io
import os
import wave

def get_sample_rate(file_path):
    """WAV 파일의 샘플 레이트를 확인합니다."""
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()
    return sample_rate

def convert_to_mono(file_path):
    """오디오 파일을 모노로 변환합니다."""
    audio = AudioSegment.from_file(file_path)
    if audio.channels != 1:
        audio = audio.set_channels(1)
        mono_file = f"mono_{os.path.basename(file_path)}"
        audio.export(mono_file, format="wav")
        return mono_file
    return file_path

def resample_audio(file_path, target_sample_rate=48000):
    """오디오 파일을 리샘플링합니다."""
    audio = AudioSegment.from_file(file_path)
    if audio.frame_rate != target_sample_rate:
        audio = audio.set_frame_rate(target_sample_rate)
        resampled_file = f"resampled_{os.path.basename(file_path)}"
        audio.export(resampled_file, format="wav")
        return resampled_file
    return file_path

def convert_to_16bit(file_path):
    """WAV 파일을 16비트 샘플로 변환합니다."""
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_sample_width(2)  # 16비트 샘플
    bit16_file = f"bit16_{os.path.basename(file_path)}"
    audio.export(bit16_file, format="wav")
    return bit16_file

def split_audio(file_path, chunk_length_ms=10000):
    """오디오 파일을 주어진 길이(밀리초)로 자릅니다."""
    audio = AudioSegment.from_file(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_file = f"chunk_{i // chunk_length_ms}.wav"
        chunk.export(chunk_file, format="wav")
        chunks.append(chunk_file)
    return chunks

def transcribe_audio_file(file_path):
    """Google Cloud Speech-to-Text API를 사용하여 음성을 텍스트로 변환합니다."""
    client = speech.SpeechClient.from_service_account_file('myKey.json')

    # 오디오 파일의 샘플 레이트 확인
    sample_rate = get_sample_rate(file_path)

    # 오디오 파일을 모노로 변환
    mono_file_path = convert_to_mono(file_path)

    # 오디오 파일 리샘플링
    resampled_file_path = resample_audio(mono_file_path, target_sample_rate=sample_rate)

    # 오디오 파일 비트 깊이 변환
    bit16_file_path = convert_to_16bit(resampled_file_path)

    with io.open(bit16_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,  # 리샘플링한 샘플 레이트 사용
        language_code="ko-KR",
    )

    # 파일이 길 경우, long_running_recognize 사용
    operation = client.long_running_recognize(config=config, audio=audio)
    print('Waiting for operation to complete...')
    response = operation.result(timeout=90)

    transcripts = []
    for result in response.results:
        transcripts.append(result.alternatives[0].transcript)

    return transcripts

def transcribe_chunks(chunk_files):
    """여러 오디오 조각을 텍스트로 변환합니다."""
    all_transcripts = []
    for chunk_file in chunk_files:
        print(f"Transcribing {chunk_file}...")
        transcripts = transcribe_audio_file(chunk_file)
        print(transcripts)
        all_transcripts.extend(transcripts)
        os.remove(chunk_file)  # 임시 조각 파일 삭제
    return all_transcripts

# 사용 예제
audio_file_path = 'audio.wav'
chunk_files = split_audio(audio_file_path, chunk_length_ms=10000)  # 10초 길이의 조각으로 나누기
transcripts = transcribe_chunks(chunk_files)

# 전체 텍스트 출력
print("\n".join(transcripts))
