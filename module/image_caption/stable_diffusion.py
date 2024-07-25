import torch
from PIL import Image
import numpy as np
import pandas as pd
from diffusers import StableDiffusionPipeline

# 이미지 로드
image_path = "../../images/a.jpg"
image = Image.open(image_path).convert("RGB")

# caption.csv 파일에서 새로운 캡션 읽어오기
caption_file_path = "caption.csv"
captions_df = pd.read_csv(caption_file_path)
# new_caption = captions_df.iloc[0, 0]  # 첫 번째 캡션을 읽어옵니다.
new_caption = "a girl introduce for japanese language class"
print(f"New Caption from CSV: {new_caption}")

# Stable Diffusion 파이프라인 로드 (GPU로 설정)
stable_diffusion_model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(stable_diffusion_model_id)

# CUDA 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = pipeline.to(device)  # GPU 또는 CPU로 설정

# GPU 사용 여부 출력
if device == "cuda":
    print("Using GPU for inference")
else:
    print("Using CPU for inference")

# Stable Diffusion을 사용하여 이미지 생성
def generate_image(prompt, init_image, num_inference_steps=50, guidance_scale=7.5):
    init_image = init_image.resize((512, 512))
    init_image = np.array(init_image) / 255.0
    init_image = torch.tensor(init_image).permute(2, 0, 1).unsqueeze(0).to(device)  # GPU 또는 CPU로 설정

    output = pipeline(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
    generated_image = output.images[0]
    
    return generated_image

# 원본 이미지와 새로운 캡션을 기반으로 이미지 생성
generated_image = generate_image(new_caption, image)
generated_image.save("generated_image.png")

print("Generated image saved as 'generated_image.png'")
