import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

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
