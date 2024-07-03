import pandas as pd
import json

# 엑셀 파일을 읽어옵니다 (openpyxl 엔진을 사용하여 읽기).
excel_file = 'sampled_train_data.xlsx'
df = pd.read_excel(excel_file, engine='openpyxl')

# 변환된 데이터를 저장할 .jsonl 파일 경로를 설정합니다.
jsonl_file = 'sentiment_data.jsonl'

# DataFrame의 각 행을 JSON 객체로 변환하고, 이를 .jsonl 파일로 저장합니다.
with open(jsonl_file, 'w') as file:
    for index, row in df.iterrows():
        # 행을 딕셔너리로 변환한 후 JSON 형식으로 인코딩합니다.
        json_record = row.to_dict()
        json_str = json.dumps(json_record)
        file.write(json_str + '\n')
