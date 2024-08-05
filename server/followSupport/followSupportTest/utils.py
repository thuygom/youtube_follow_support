# myapp/utils.py
import subprocess

def create_caption(image_url):
    #conda 환경에서 스크립트 실행
    conda_env_name = "img2img-turbo"
    script_path = "C:/Windows/System32/generate_caption.py"  # 스크립트 경로를 설정
    command = f"conda run -n {conda_env_name} python {script_path} {image_url}"

    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def variation_image(image_url, model, prompt):
    conda_env_name = "img2img-turbo"
    script_path = "C:/Users/repli/Desktop/갠프/pythonDjango/youtube_follow_support/module/image_caption/img2img_turbo/img2img-turbo/src/inference_paired.py"  # 스크립트 경로를 설정
    output_path = "C:/Windows/System32/followSupport/media/uploads"
    # 인용부호를 이스케이프 처리하여 커맨드 문자열을 작성합니다.
    if model == "edge_to_image":
        command = (
            f"conda run -n {conda_env_name} python {script_path} "
            f"--model_name \"{model}\" "
            f"--input_image \"{image_url}\" "
            f"--prompt \"{prompt}\" "
            f"--output_dir \"{output_path}\""
        )
    elif model == "sketch_to_image_stochastic":
        command = (
            f"conda run -n {conda_env_name} python {script_path} "
            f"--model_name \"{model}\" "
            f"--input_image \"{image_url}\" "
            f"--gamma 0.4 "
            f"--prompt \"{prompt}\" "
            f"--output_dir \"{output_path}\""
        )
    print(f"Prompt received: {command}")  # 로그 출력
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

# 예시로 사용
# print(variation_image("path/to/your/image.png"))