import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Hugging Face 모델 허브에서 파인튜닝된 CLIP 모델 로드
model_name = "openai/clip-vit-base-patch32"  # 여기서 다른 파인튜닝된 모델 이름을 사용할 수 있습니다.
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 이미지 로드
image = Image.open("../images/default.jpg")

# 텍스트 후보군 설정
texts = ["a photo of a cat", "a photo of a dog", "a photo of a landscape", "a photo of a person", "a photo of a building"]

# 입력 데이터 처리
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 모델 예측
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # 이미지와 텍스트 간의 유사도 점수
probs = logits_per_image.softmax(dim=1)  # 소프트맥스 함수를 사용하여 확률 계산

# 가장 높은 확률의 텍스트 선택
best_caption = texts[probs.argmax()]

print(f"Generated Caption: {best_caption}")
