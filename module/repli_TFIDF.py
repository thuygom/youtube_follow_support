import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 전처리된 결과를 담은 엑셀 파일 읽기
preprocessed_file_path = '../xlsx/result_sub.xlsx'
preprocessed_df = pd.read_excel(preprocessed_file_path)

# 전처리된 댓글 문장 추출
preprocessed_sentences = preprocessed_df['Processed_Comments'].tolist()

# TF-IDF 모델 생성 및 데이터프레임 저장
def save_tfidf_to_excel(corpus, output_file_path, threshold=0.2):
    # TF-IDF 모델 생성
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    # TF-IDF 값 데이터프레임 생성
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # 임계값을 넘는 열 확인
    above_threshold_cols = tfidf_df.columns[(tfidf_df > threshold).any()]

    # 임계값을 넘는 열만 포함하여 저장
    tfidf_df_filtered = tfidf_df[above_threshold_cols]

    # 결과를 엑셀 파일로 저장
    tfidf_df_filtered.to_excel(output_file_path, index=False)
    print(f"TF-IDF 결과가 {output_file_path} 파일로 저장되었습니다.")

# TF-IDF 결과 저장
output_file_path = '../xlsx/result_tfidf.xlsx'
save_tfidf_to_excel(preprocessed_sentences, output_file_path)
