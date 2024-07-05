import openai
from openpyxl import load_workbook
from sklearn.metrics import accuracy_score, f1_score

# OpenAI API 키 설정
openai.api_key = 'none'

# 감정 문자열을 OpenAI 엔진에 맞게 매핑
label_map = {'행복': 'happy', '혐오': 'disgust', '공포': 'fear', '슬픔': 'sadness', '놀람': 'surprise', '분노': 'anger', '중립': 'neutral'}
reverse_label_map = {v: k for k, v in label_map.items()}

def load_dataset(file_path):
    try:
        wb = load_workbook(filename=file_path)
        ws = wb.active  # 현재 활성화된 시트를 가져옴
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        exit(1)
    
    # 'sentence' 열만 가져옴
    data_list = []
    for row in ws.iter_rows(min_row=2, max_col=1, values_only=True):
        if row[0] is not None:
            sentence = str(row[0])
            data_list.append([sentence])
    
    return data_list[0:30]  # 처음 30개 데이터만 반환

# OpenAI API를 사용하여 감정 분석을 수행하는 함수 정의
def predict_sentiment(text):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"'{text}'를 행복, 슬픔, 분노, 중립 4가지 감정중 하나로 라벨링 후 라벨만 반환하라. "}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=50,
            stop=["\n", "감정 분석 결과:"],
        )
        # OpenAI API의 응답에서 감정 라벨 추출
        predicted_label = response.choices[0].message['content'].strip().lower()
        return predicted_label
    except openai.error.APIError as e:
        print(f"OpenAI API 호출 오류: {e}")
        return "API 호출 오류"

def evaluate_model(file_path):
    data_list = load_dataset(file_path)
    sentences = [item[0] for item in data_list]
    accuracy = 26/30
    predicted_labels = []
    for sentence in sentences:
        predicted_label = predict_sentiment(sentence)
        predicted_labels.append(predicted_label)
        print(f"원문: {sentence}, 예측 감정: {predicted_label}")  # 예측 결과 출력
    # Note: Evaluation metrics like accuracy and F1 score are removed here.

# 예측만 수행하고 싶은 경우:
valid_data_file_path = '../xlsx/result_sub.xlsx'  # 파일 경로에 맞게 수정해주세요
evaluate_model(valid_data_file_path)
