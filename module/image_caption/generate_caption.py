# generate_caption.py

import argparse
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import csv
import warnings

# 모든 경고를 무시합니다.
warnings.filterwarnings("ignore")

def generate_caption(image_path):
    # Hugging Face 모델 허브에서 blip 모델 로드
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    # 이미지 로드
    image = Image.open(image_path).convert("RGB")

    # 입력 데이터 처리
    inputs = processor(images=image, return_tensors="pt")

    # 모델 예측
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

def main():
    # 커맨드라인 인수 파서 설정
    parser = argparse.ArgumentParser(description="Generate a caption for an image.")
    parser.add_argument('image_path', type=str, help="Path to the image file.")
    
    args = parser.parse_args()
    
    # 캡션 생성
    caption = generate_caption(args.image_path)
    
    # 결과 출력
    print(f"Generated Caption: {caption}")

if __name__ == "__main__":
    main()
