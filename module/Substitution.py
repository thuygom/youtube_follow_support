import pandas as pd
from konlpy.tag import Okt
import re

def apply_regular_expression(text):
    try:
        hangul = re.compile('[^ ㄱ-ㅣ가-힣a-zA-Z ]')  # 숫자 제거
        result = hangul.sub('', text)
        return result
    except TypeError:
        # 입력이 None 등으로 들어온 경우 처리
        return text
    except Exception as e:
        # 기타 예외 발생 시 처리
        print(f"Error applying regular expression: {e}")
        return None
    
# 엑셀 파일 읽기
file_path = '../xlsx/crawling_manu.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# 댓글 문장 추출 및 정규 표현식 적용 (1번 열에만 적용)
second_column = df.iloc[:, 1].apply(apply_regular_expression)

# 형태소 분석기 초기화
okt = Okt()

# 은어를 표준어로 변경하는 딕셔너리
replace_dict = {
    "ㅅㅂ": "싫어",
    "ㅈㄹ": "뭐라",
    "ㅈㄴ": "매우",
    "존나": "매우",
    "ㄲㅈ": "나가",
    "빡친다": "화나다",
    "개꿀": "좋다",
    "열폭": "열등감 폭발",
    "조졌다": "망하다",
    "쩐다":"대단하다",
    "ㄹㅇ": "진짜",
    "듣보": "흔하지 않다",
    "커엽다": "귀엽다",
    "인싸": "친구가 많다",
    "아싸": "친구가 적다",
    "존버": "오래 버티기",
    "국산": "테스트용" # 첫번째 문장 첫번째 단어 변경사항 테스트용     
}

# 비속어 및 특정 단어 변경 함수
def replace_words(morphs, replacement_dict):
    return [replacement_dict.get(word, word) for word in morphs]

# 두 번째 컬럼의 각 문장에 대해 형태소 분석 수행 및 문장 변환
def analyze_and_replace(sentence):
    #morphs = okt.morphs(sentence, stem=True, norm=True)
    #replaced_morphs = replace_words(morphs, replace_dict)
    #return ' '.join(replaced_morphs)
    return sentence

# 1번 열(댓글 문장)에 대해 전처리 및 형태소 분석 수행
preprocessed_sentences = [analyze_and_replace(sentence) for sentence in second_column]

# 한 글자 키워드 제거 및 deleted 추가
#preprocessed_sentences_filtered = []
#for sentence in preprocessed_sentences:
#    filtered_sentence = ' '.join([word for word in sentence.split() if len(word) > 1])
#    if not filtered_sentence.strip():  # 남은 문장이 없으면
#        filtered_sentence = 'none'
#    preprocessed_sentences_filtered.append(filtered_sentence)

# 전처리된 댓글 문장을 데이터프레임으로 변환
preprocessed_df = pd.DataFrame({
    'Processed_Comments': preprocessed_sentences
})

# 전처리된 결과를 Excel 파일로 저장
preprocessed_output_file_path = '../xlsx/result_sub.xlsx'
preprocessed_df.to_excel(preprocessed_output_file_path, index=False)
print(f"전처리된 문장이 {preprocessed_output_file_path} 파일로 저장되었습니다.")
