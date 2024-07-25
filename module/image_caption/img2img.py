import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline

# 이미지 로드
init_image_path = "../../images/default.jpg"
init_image = Image.open(init_image_path).convert("RGB")

# 텍스트 프롬프트
prompt = "A futuristic cityscape with flying cars and neon lights"

# 모델 로드
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 변형 함수
def generate_image(prompt, init_image, num_inference_steps=50, strength=0.75, guidance_scale=7.5):
    # 이미지 크기 조정
    init_image = init_image.resize((512, 512))

    # 이미지를 numpy 배열로 변환한 후, torch tensor로 변환
    init_image_array = np.array(init_image).astype(np.float32) / 255.0
    init_image_tensor = torch.from_numpy(init_image_array).permute(2, 0, 1).unsqueeze(0).to(pipe.device)

    # `diffusers` 라이브러리는 이미지 텐서를 `[B, C, H, W]` 형식으로 처리합니다.
    init_image_tensor = init_image_tensor.float()

    # 이미지 생성
    output = pipe(prompt=prompt, init_image=init_image_tensor, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
    generated_image = output.images[0]
    return generated_image

# 이미지 변형
generated_image = generate_image(prompt, init_image)
generated_image.save("generated_image.png")

print("Generated image saved as 'generated_image.png'")
