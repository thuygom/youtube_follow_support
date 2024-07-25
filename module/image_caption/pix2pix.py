import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

# Pix2Pix 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('junyanz/pytorch-CycleGAN-and-pix2pix', 'pix2pix', 'edges2shoes', pretrained=True)
model.to(device).eval()

# 이미지 변환 함수
def transform_image(image):
    transform_list = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
    transform = transforms.Compose(transform_list)
    return transform(image).unsqueeze(0).to(device)

# 결과 이미지 변환 함수
def transform_output(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor + 1) / 2  # De-normalize
    return transforms.ToPILImage()(tensor)

# 샘플 이미지 로드 (여기서는 URL에서 다운로드하는 방식 사용)
url = "https://i.imgur.com/f9z5iA4.png"
response = requests.get(url)
input_image = Image.open(BytesIO(response.content)).convert("RGB")

# 이미지 변환
input_tensor = transform_image(input_image)
output_tensor = model(input_tensor)

# 결과 이미지
output_image = transform_output(output_tensor)

# 결과 이미지 저장
output_image.save("output_image.png")
print("Generated image saved as 'output_image.png'")
