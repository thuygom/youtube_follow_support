import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd

# Hugging Face 모델 허브에서 blip 모델 로드
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# 이미지 로드
image_path = "../images/default.jpg"
image = Image.open(image_path).convert("RGB")

# 입력 데이터 처리
inputs = processor(images=image, return_tensors="pt")

# 모델의 중간 출력 가져오기
with torch.no_grad():
    encoder_outputs = model.vision_model(**inputs)

# 중간 출력의 피처 추출
last_hidden_states = encoder_outputs.last_hidden_state.squeeze().cpu().numpy()

# 특성 벡터를 DataFrame으로 변환
df = pd.DataFrame(last_hidden_states)

# 각 패치의 인덱스를 추가
df.index.name = 'Patch Index'
df.reset_index(inplace=True)

# DataFrame 확인
print(df.head())

# 모델 예측
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print(f"Generated Caption: {caption}")

# 저장하려면 CSV 파일로 저장
df.to_csv("patch_features.csv", index=False)
