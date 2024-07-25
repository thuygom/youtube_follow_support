import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import matplotlib.pyplot as plt
import csv

# Hugging Face 모델 허브에서 blip 모델 로드
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# 이미지 로드
image_path = "../../images/default.jpg"
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


def save_caption_to_csv(caption, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Generated Caption"])
        writer.writerow([caption])

def save_keypoints_and_descriptors_to_csv(keypoints, descriptors, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Keypoints 정보를 저장
        writer.writerow(["Index", "pt_x", "pt_y", "size", "angle", "response", "octave", "class_id"])
        for i, kp in enumerate(keypoints):
            writer.writerow([i, kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id])
        
        # Descriptors 정보를 저장
        writer.writerow([])  # 빈 줄 추가
        writer.writerow(["Index"] + [f"descriptor_{i}" for i in range(descriptors.shape[1])])
        for i, desc in enumerate(descriptors):
            writer.writerow([i] + desc.tolist())

caption_file_path = "caption.csv"
save_caption_to_csv(modified_caption, caption_file_path)

keypoints_file_path = "keypoints_and_descriptors.csv"
save_keypoints_and_descriptors_to_csv(keypoints, descriptors, keypoints_file_path)

print(f"Caption saved to {caption_file_path}")
print(f"Keypoints and descriptors saved to {keypoints_file_path}")
