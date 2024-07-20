import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import matplotlib.pyplot as plt

# Hugging Face 모델 허브에서 blip 모델 로드
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# 이미지 로드
image_path = "../images/default.jpg"
image = Image.open(image_path).convert("RGB")

# 입력 데이터 처리
inputs = processor(images=image, return_tensors="pt")

# 모델 예측
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print(f"Generated Caption: {caption}")


def add_keywords_to_caption(caption, keywords):
    # 간단한 예시로, 키워드를 캡션 끝에 추가합니다.
    new_caption = caption + " " + " ".join(keywords)
    return new_caption

keywords = ["new", "keywords"]
modified_caption = add_keywords_to_caption(caption, keywords)
print(f"Modified Caption: {modified_caption}")


def extract_image_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 여기서는 간단히 SIFT를 사용하여 주요 특징점을 추출합니다.
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def print_keypoints_and_descriptors(keypoints, descriptors):
    print(f"Extracted {len(keypoints)} keypoints and descriptors.")
    
    for i, kp in enumerate(keypoints):
        print(f"Keypoint {i}:")
        print(f" - pt: {kp.pt}")
        print(f" - size: {kp.size}")
        print(f" - angle: {kp.angle}")
        print(f" - response: {kp.response}")
        print(f" - octave: {kp.octave}")
        print(f" - class_id: {kp.class_id}")
        print(f"Descriptor {i}: {descriptors[i]}")


keypoints, descriptors = extract_image_features(image_path)
print_keypoints_and_descriptors(keypoints, descriptors)

