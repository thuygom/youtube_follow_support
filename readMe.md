**유튜브 인플루언서 스탯 분석 및 댓글 감정 분석을 통한 충성도 검출 추가적인 멀티모달리티 유틸리티 기능 제공**

**1. Project 설명**

- 데이터 마이닝 강의에서 배운 데이터 마이닝 기법을 통해 소셜 미디어(유튜브)에서 데이터를 추출하고(**구글 youtubve API version3를 사용한 웹 크롤링**) **영상 댓글을 평가하여 우호적인 댓글과 비우호적 댓글을 점수를 매기는 알고리즘을 AI 모델링**(KoBert Model 파인 튜닝)을 통해 개발한다. 이는 직접 AI를 모델링하거나 만들어진 AI 모델을 가져와(openAi API 등) 우리가 사용할 목적에 맞게 **AI 파인튜닝하고** 적절히 사용할 수 있다. 
- 댓글 감정 분석 모델 및 댓글의 주체(Object)를 판별해 영상에 대해 Negative한지, Positive한지 분류하고, 유튜버에 공감하는지 등을 분류하고 **Follow Support(충성도)**를 도출해내고, 해당 유튜버의 모든 영상에 대해  다양한 Stat들을 도출해 해당 유튜버의 팔로워 지지율까지 도출해낸다. 
  - Stat
    - 구독자 수
    - 영상 평균 조회수
    - 댓글 충성도
    - 유튜브 영상 썸네일
    - 영상 조회수
    - 영상 좋아요 수
    - 댓글 개수
    - 영상 주기
- **목표 1. 도출해 낸 다양한 Stat을 기반으로 유튜버 인플루언서들의 Ranking List를 만들고, 유튜버의 Stat을 보여줄 수 있는 웹서비스를 운영한다.(1~3주차 진행)**
- **목표2. 웹서비스에 멀티 모달리티를 활용한 자동 썸네일 제작 혹은 추가적인 기능을 제공하는 AI Model을 제작한다.(4~7주차 진행)**

[프로그램 흐름도]
![흐름도](https://github.com/user-attachments/assets/708cc30f-6adc-4376-a1d2-f72e84674b11)

**2. 활동 수행 결과물** 

- 파이썬 언어 기반 웹크롤링 모듈

  ```python
      buttons = driver.find_elements(By.CSS_SELECTOR, "ytd-button-renderer#more-replies > yt-button-shape > button")
      print(buttons[1])
      for i in range(len(buttons)):
          buttons[i].send_keys(Keys.ENTER)
      print("Preprocessing complete")
      #상호작용할 수 있는 버튼을 가져와 답글 더보기 버튼에 Enter키를 보내는 로직
      
      id_list = soup.select("div#header-author > h3 > #author-text > span")
      comment_list = soup.select("div#content > yt-attributed-string#content-text > span")
      like_list = soup.select("span#vote-count-middle")
      #CSS Selector로 원하는 정보를 선택해 가져오는 로직 코드
  ```
![image](https://github.com/thuygom/youtube_follow_support/assets/138266353/627c9f1b-670a-4614-82fa-82746c2e34ff)
![image](https://github.com/thuygom/youtube_follow_support/assets/138266353/c424a008-e3a4-46ab-b588-c33065427727)

Selenium과 BeautifulSoup4 라이브러리를 사용해 웹페이지의 Object와 상호작용할 수 있으며 전체 html코드를 가져온 후, CSS Selector를 통해 필요한 데이터들을 수집할 수 있는 모듈.

```python
# Function to get video statistics
def get_video_info(api_obj, video_id):
    response = api_obj.videos().list(
        part='snippet,statistics,topicDetails',
        id=video_id
    ).execute()
    video_info = response['items'][0]
    snippet = video_info['snippet']
    statistics = video_info['statistics']
    topic_details = video_info.get('topicDetails', {})  # Handle cases where topicDetails might not exist
    return snippet, statistics, topic_details
#google API를 통해 필요한 stat정보를 담고있는 객체를 불러오는 핵심 로직
```

google Youtuve Api Version3를 사용해 비디오 Hash값부터 다양한 stat들 또한 수집할 수 있다.

| Main Developer | 김정훈 |
| :------------: | :----: |
| Sub Developer  | 노태원 |

- 인터넷 은어 및 욕설 치환 모듈(Python)

  ```python
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
      morphs = okt.morphs(sentence, stem=True, norm=True)
      replaced_morphs = replace_words(morphs, replace_dict)
      return ' '.join(replaced_morphs)
  ```

  konlpy의 Okt라이브러리를 통해 문장을 형태소 단위로 분해하고, 단어들을 원형복구를 진행하며, 인터넷 은어들을 치환시켜 AI가 이해하기 쉽게 전처리 해준다.

  | Main Developer | 노태원 |
  | :------------: | :----: |

- TF-IDF방식의 단어 벡터화 모듈

  형태소 단위로 분해되어있으며, 원형복구가 되어있는 동사 및 형용사들을 기반으, 하나의 댓글을 하나의 Document로 인식하고 TF-IDF방식으로 단어 벡터화를 진행했다. 문장에서 각 단어가 지니는 중요도를 구별하는데 유용하게 사용되는 데이터이다. 사이킷 런의 TfidfVectorizer를 통해서 단어 벡터화를 진행했다.

  ```python
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
  ```
![image](https://github.com/thuygom/youtube_follow_support/assets/138266353/e151f6dc-80e5-4e8f-948e-11051adba2b6)

| Main Developer | 이연준 |
| :------------: | :----: |

- 감정 분석 및 감정 대상 분석 AI Model

  Youtube영상에서 Context(문맥)을 추출하고, 문맥을 기반으로 댓글이 담은 감정을 파악하며, 댓글의 감정이 향하는 대상(Object)또한 구별할 수 있다.

  openAi gpt3.5Turbo 모델과 KoBert모델을 사용한 AI 모델이며, pyTorch방식으로 AI 모델을 파인튜닝하였다. 

  [openAi Model]

  ```python
  # OpenAI API를 사용하여 감정 분석을 수행하는 함수 정의
  def predict_sentiment(text):
      try:
          messages = [
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": f"'{context}' 문맥을 기반으로 '{text}'를 행복, 웃김, 슬픔, 분노, 중립 5가지 감정중 하나로 라벨링 후 감정 라벨을 반환하라. "}
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
  
  # OpenAI API를 사용하여 감정 분석을 수행하는 함수 정의
  def predict_object(text, num_predictions=5):
      try:
          messages = [
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": f"문맥: {context}. 유튜버: {youtuber}."},
              {"role": "user", "content": "다음 문장이 얘기하는 대상이 context, youtuber, other중 어떤 것인지 선택하세요. 예: '안드레진 미친놈이네' -> context"},
              {"role": "user", "content": f"문장: '{text}'"}
          ]
          results = []
          for _ in range(num_predictions):
              response = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages=messages,
                  max_tokens=50,
              )
              predicted_label = response.choices[0].message['content'].strip().lower()
              results.append(predicted_label)
          most_common_label = Counter(results).most_common(1)[0][0]
          return most_common_label
      except openai.error.APIError as e:
          print(f"OpenAI API 호출 오류: {e}")
          return "API 호출 오류"
  ```

  openAi의 경우 Prompt를 생성하여, openAi 사이트에서 jsonl형식의 DataSet을 기반으로 파인튜닝 했으며, Accuracy 87퍼센트 정도를 보여준다. 응답 품질을 향상시키기 위해서 데이터 셋에 걸맞게 프롬프트 튜닝을 하였고, 추가적으로 프롬프트 앙상블링(여러개의 prompt로 모델의 응답을 생성한 후, 이를 종합하여 최종응답을 도출하는 방법)으로 정답의 품질을 향상시켰다. 밑에 예시 사진으로는 오킹이라는 유튜버에 사과영상의 댓글을 분석한 것인데, 댓글을 기반으로 누구에 대해 감정을 느끼고, 그대상이 누구인지에 대해서 분석한 결과를 반환한다.

![image](https://github.com/thuygom/youtube_follow_support/assets/138266353/81e94100-55d8-47aa-bc2f-ca53b27a52c8)

![image](https://github.com/thuygom/youtube_follow_support/assets/138266353/dbac2972-bb05-4d4e-9c8e-3db8f7d9c126)

  [koBert model]

  ```python
  # 데이터셋 클래스 정의
  class CommentDataset(Dataset):
      def __init__(self, data_list, tokenizer, max_length):
          self.data_list = data_list
          self.tokenizer = tokenizer
          self.max_length = max_length
          
      def __len__(self):
          return len(self.data_list)
      
      def __getitem__(self, idx):
          comment, label = self.data_list[idx]
          
          encoding = self.tokenizer(comment, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
          inputs = {
              'input_ids': encoding['input_ids'].flatten(),
              'attention_mask': encoding['attention_mask'].flatten(),
              'labels': torch.tensor(label, dtype=torch.long)
          }
          return inputs
  
  # 학습 데이터셋 로딩
  data_list = load_dataset(train_file_path)
  print(f"전체 데이터셋 크기: {len(data_list)}")
  
  # 데이터셋을 학습용과 검증용으로 분리 (60% 학습, 40% 검증)
  train_data_list, eval_data_list = train_test_split(data_list, train_size=0.6, test_size=0.4, random_state=42)
  
  train_dataset = CommentDataset(train_data_list, tokenizer, max_length=128)
  eval_dataset = CommentDataset(eval_data_list, tokenizer, max_length=128)
  
  training_args = TrainingArguments(
      per_device_train_batch_size=128,
      per_device_eval_batch_size=32,
      num_train_epochs=5,
      logging_dir='./logs',
      logging_steps=100,
      evaluation_strategy="epoch",  # 에포크마다 검증
      save_steps=500,  # 500 steps 마다 모델 저장
      output_dir='./results2',  # 모델 저장 디렉토리
      overwrite_output_dir=True,
  )
  
  
  # 정확도와 F1 score를 계산하는 함수 정의
  def compute_metrics(pred):
      labels = pred.label_ids
      preds = pred.predictions.argmax(-1)
      acc = accuracy_score(labels, preds)
      f1 = f1_score(labels, preds, average='weighted')  # weighted average F1 score
      return {
          'accuracy': acc,
          'f1_score': f1,
      }
  
  
  # Trainer 초기화 및 Fine-tuning
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      compute_metrics=compute_metrics,
  )
  
  # 모델 가중치 일부 확인
  print("모델 가중치 확인:")
  print(model.bert.encoder.layer[0].attention.self.query.weight[:5])  # 일부 가중치 출력
  
  # Fine-tuning 진행
  trainer.train()
  
  # 학습된 모델 및 토크나이저 저장
  trainer.save_model('./results2')
  tokenizer.save_pretrained('./results2')
  print("모델과 토크나이저가 ./results2 디렉토리에 저장되었습니다.")
  ```

  koBert모델의 경우 koBert모델을 가져온 후, 하이퍼 파라미터를 설정하고, 학습을 진행한후, 학습이 진행된 모델을 저장 후, 다른 코드에서 불러와서 학습된 모델을 기반으로 평가를 진행할 수 있다.

  ![epoch5](https://github.com/thuygom/youtube_follow_support/assets/138266353/2e7d7016-3f51-462d-9e63-b325c014a1cf)

  ![koBert](https://github.com/thuygom/youtube_follow_support/assets/138266353/8e20dcd9-bdde-4f93-aa60-85263ba92f39)

  ![openAI](https://github.com/thuygom/youtube_follow_support/assets/138266353/87f59c4e-396e-461d-8647-fcf33841eb60)

  | Main Developer | 김정훈 |
  | :------------: | :----: |

- SQL DB연동

  파이썬에서 MySql DB로 유튜버의 스탯과 댓글들을 연동시켰다.

  ```python
  # 데이터프레임의 각 행을 Youtuber 테이블에 삽입
  for index, row in df.iterrows():
      insert_query = """
      INSERT INTO VideoStat (video_id, upload_date, date, view_count, like_count, comment_count, subscriber_count, channel_id, channel_title, channel_description, topic_categories, title, description, tags, thumbnails)
      VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
      """
      data = (
          row['video_id'] if pd.notnull(row['video_id']) else None,
          row['upload_date'] if pd.notnull(row['upload_date']) else None,
          row['date'] if pd.notnull(row['date']) else None,
          row['view_count'] if pd.notnull(row['view_count']) else None,
          row['like_count'] if pd.notnull(row['like_count']) else None,
          row['comment_count'] if pd.notnull(row['comment_count']) else None,
          row['subscriber_count'] if pd.notnull(row['subscriber_count']) else None,
          row['channel_id'] if pd.notnull(row['channel_id']) else None,
          row['channel_title'] if pd.notnull(row['channel_title']) else None,
          row['channel_description'] if pd.notnull(row['channel_description']) else None,
          row['topic_categories'] if pd.notnull(row['topic_categories']) else None,
          row['title'] if pd.notnull(row['title']) else None,
          row['description'] if pd.notnull(row['description']) else None,
          row['tags'] if pd.notnull(row['tags']) else None,
          row['thumbnails'] if pd.notnull(row['thumbnails']) else None
      )
      cursor.execute(insert_query, data)
  
  for index, row in df2.iterrows():
          insert_query = """
          INSERT INTO Comments (comment, author, date, num_likes, video_id)
          VALUES (%s, %s, %s, %s, %s)
          """
          data = (
              row['comment'],
              row['author'],
              row['date'],
              row['num_likes'],
              row['video_id']
          )
          cursor.execute(insert_query, data)
  ```

  이렇게 python에서 DB쪽으로 업로드 해주면 다음 사진과 같이 DB에 올라간다.

  ![youtuber_stat](https://github.com/thuygom/youtube_follow_support/assets/138266353/b0c7e030-9341-41ab-a952-f55da65cf816)

  ![comment_db](https://github.com/thuygom/youtube_follow_support/assets/138266353/0876b3b7-620e-4f5b-bef7-2ffff023c614)

  | Main Developer | 김정훈 |
  | :------------: | :----: |
  | Sub Developer  | 이연준 |

  

- 자동 로그 수집 모듈

  원하는 유튜버들의 영상에 대해 일자별로 로그를 수집할 수 있도록 기존 웹크롤링을 모듈화 시켜 자동로그 수집 되도록 만들었다.

  ```python
  VIDEO_Oking = ['oakQvwCbvr8', '-HZeqsIgGHo', 'nAy-7zuCVQs', 'QojVuirFx58', 'ibI5OOZXSj8']
  VIDEO_STYLE = ['g5KDoSqT24Q', 'mnn1_yu0aDQ', 'pp_C0MGj9ZM', '_Otk-iMD_X0', '_HZ63R-8z4E']
  VIDEO_GAME = ['yLlKOd3I8CA', 'BYWO-z-4tfo', 'uNq7RMRwIHs', 'ZLXz98YW_U0', 'qkXc1M3d7g4']
  VIDEO_MUSIC = ['-rHqqCxjhM4', 'FGjrtBFgTXY', 'TOSrdHy5KRc', 'wdUu3XQK8cE', 'LamRCcz4zqg']
  VIDEO_ISSUE = ['ahcPfSLbT-M', '8l4GZ4datyM', '7I790Er-zkc', '8SJs1Cg7hpU', 'VWmWScovllY']
  
  # File paths
  STATS_FILE_PATH = '../xlsx/stats0709.xlsx'
  COMMENTS_FILE_PATH = '../xlsx/crawling_auto0709.xlsx'
  
  extract(VIDEO_Oking,STATS_FILE_PATH, COMMENTS_FILE_PATH)
  
  extract(VIDEO_STYLE,STATS_FILE_PATH, COMMENTS_FILE_PATH)
  
  extract(VIDEO_GAME,STATS_FILE_PATH, COMMENTS_FILE_PATH)
  
  extract(VIDEO_MUSIC,STATS_FILE_PATH, COMMENTS_FILE_PATH)
  
  extract(VIDEO_ISSUE,STATS_FILE_PATH, COMMENTS_FILE_PATH)
  ```

  | Main Developer | 김정훈 |
  | :------------: | :----: |


- Mysql 과 Django Server 연동 및 웹퍼블리싱

  ```python
  # followSupportTest/models.py
  
  from django.db import models
  
  class VideoStat(models.Model):
      video_num = models.IntegerField(primary_key=True)
      video_id = models.CharField(max_length=50)
      upload_date = models.DateTimeField()
      date = models.DateField()
      view_count = models.IntegerField()
      like_count = models.IntegerField()
      comment_count = models.IntegerField()
      subscriber_count = models.IntegerField()
      channel_id = models.CharField(max_length=50)
      channel_title = models.CharField(max_length=100)
      channel_description = models.TextField(null=True)
      topic_categories = models.TextField(null=True)
      title = models.CharField(max_length=200, null=True)
      description = models.TextField(null=True)
      tags = models.TextField(null=True)
      thumbnails = models.CharField(max_length=255)
  
      class Meta:
          managed = False
          db_table = 'VideoStat'
  
      def __str__(self):
          return self.video_id
  
  class Comment(models.Model):
      comment_id = models.IntegerField(primary_key=True)
      comment = models.TextField()
      author = models.CharField(max_length=100)
      date = models.DateTimeField()
      num_likes = models.IntegerField()
      video_id = models.CharField(max_length=100)
      emotion = models.CharField(max_length=100)
      object = models.CharField(max_length=100)
  
      class Meta:
          managed = False
          db_table = 'Comments'
  
      def __str__(self):
          return self.comment
  ```

  파이썬 Django Server에서 DB스키마를 생성하고, Managed를 false로 설정하므로서, 새로이 마이그레이션 하지않고 기존 Mysql DB 에서 데이터를 가져오도록 설정되었다. 이제 서버를 키고 localhost로 접속해보면

  ![runserver](https://github.com/user-attachments/assets/e06d9fec-8da9-48e3-9491-81676109446a)

  ![prototype_web](https://github.com/user-attachments/assets/663f88ac-a430-433c-a67e-78b2e7ca3ea2)

  위처럼 유튜버들의 스탯과 댓글들을 확인할 수 있다.

  | Main Developer | 김정훈 |
  | :------------: | :----: |

  

- google trend API를 활용한 일자별 관심도 시각화 및 연관 검색어 추출

  [google_trend_api.py]

  ```python
  import time
  from pytrends.request import TrendReq
  import matplotlib.pyplot as plt
  from matplotlib import font_manager, rc
  import pandas as pd
  
  # 한글 폰트 설정 (Windows에서 기본 폰트 사용)
  font_path = 'C:/Windows/Fonts/malgun.ttf'  # Windows 기본 폰트 경로 설정
  
  # 폰트 설정
  font_name = font_manager.FontProperties(fname=font_path).get_name()
  rc('font', family=font_name)
  
  # pytrends 세션 시작
  pytrends = TrendReq(hl='ko', tz=360)  # 'hl'을 'ko'로 설정하여 한글 결과 받기
  
  # 유튜버 키워드 리스트
  keywords = ["오킹", "한동숙", "뻑가", "깡 스타일리스트", "때잉 플레이리스트"]
  
  # 3개월 전 데이터를 가져오기 위해 시간 범위 설정
  timeframe = 'today 3-m'
  
  # pytrends로 데이터 요청
  pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
  time.sleep(60)  # 요청 후 대기
  
  # 관심도 데이터 가져오기
  interest_over_time_df = pytrends.interest_over_time()
  
  # 데이터프레임을 Excel 파일로 저장
  interest_over_time_df.to_excel('trend_interest.xlsx', index=True)
  
  # 데이터 시각화
  plt.figure(figsize=(14, 8))
  
  for keyword in keywords:
      plt.plot(interest_over_time_df.index, interest_over_time_df[keyword], label=keyword)
  
  plt.title('유튜버 관심도 (3개월 전 데이터)', fontsize=15)
  plt.xlabel('날짜', fontsize=12)
  plt.ylabel('관심도', fontsize=12)
  plt.legend(title='유튜버', fontsize=10)
  plt.xticks(rotation=45)
  plt.grid(True)
  plt.show()
  ```

  ![interst](https://github.com/user-attachments/assets/c6810ae7-0bbf-4e46-91dc-3846e56637d6)

  ![interest_xlsx](https://github.com/user-attachments/assets/6e5006d8-dd67-47d3-b94e-a012b4a7aa8b)

  이처럼 구글 트렌드 api를 통해 일자별 유튜버에 대한 관심도를 얻을 수 있었다. 추가적으로 이 데이터들을 웹 페이지에 띄워주어서 사용자들에게 제공하려고 한다.

   

  | Main Developer | 김정훈 |
  | :------------: | :----: |
  
  [ Google_related.py ]
  
  ```python
  import time
  from pytrends.request import TrendReq
  import pandas as pd
  
  # pytrends 세션 시작
  pytrends = TrendReq(hl='ko', tz=360)
  
  # 유튜버 키워드 리스트
  keywords = ["오킹", "한동숙", "뻑가", "깡 스타일리스트", "때잉 플레이리스트"]
  
  # 연관 검색어를 저장할 데이터프레임 초기화
  related_queries_df = pd.DataFrame(columns=['유튜버', '연관 검색어'])
  
  # 각 유튜버 키워드에 대해 연관 검색어 가져오기
  for keyword in keywords:
      time.sleep(600)  # 요청 간 대기
      pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='', gprop='')
      related_queries = pytrends.related_queries()[keyword]['top']
      
      if related_queries is not None:
          # 각 연관 검색어를 데이터프레임에 추가
          for _, row in related_queries.iterrows():
              new_row = pd.DataFrame({'유튜버': [keyword], '연관 검색어': [row['query']]})
              related_queries_df = pd.concat([related_queries_df, new_row], ignore_index=True)
      print(related_queries)
  
  # 데이터프레임을 Excel 파일로 저장
  related_queries_df.to_excel('related_queries.xlsx', index=False)
  
  print("연관 검색어가 'related_queries.xlsx'로 저장되었습니다.")
  
  ```
  
  ![related_keyword](https://github.com/user-attachments/assets/0c1b8eda-bf28-4d81-849a-9be44d17c80b)
  
  이처럼 연관검색어를 얻고 이를 웹페이지에 부가적인 정보로 알려주며, 추가적인 연관검색어 분석을 통해 positive와 negative를 얻을 수 있다.
  
  | Main Developer | 김정훈 |
  | :------------: | :----: |
  
  [image_caption.py]
  
  ```python
  import torch
  from PIL import Image
  from transformers import BlipProcessor, BlipForConditionalGeneration
  import cv2
  import matplotlib.pyplot as plt
  
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
  
  
  def add_keywords_to_caption(caption, keywords):
      # 간단한 예시로, 키워드를 캡션 끝에 추가합니다.
      new_caption = caption + " " + " ".join(keywords)
      return new_caption
  
  keywords = ["new", "keywords"]
  modified_caption = add_keywords_to_caption(caption, keywords)
  print(f"Modified Caption: {modified_caption}")
  
  
  def extract_image_features(image_path):
      image = cv2.imread(image_path)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # 여기서는 간단히 SIFT를 사용하여 주요 특징점을 추출합니다.
      sift = cv2.SIFT_create()
      keypoints, descriptors = sift.detectAndCompute(gray, None)
      return keypoints, descriptors
  
  def print_keypoints_and_descriptors(keypoints, descriptors):
      print(f"Extracted {len(keypoints)} keypoints and descriptors.")
      
      for i, kp in enumerate(keypoints):
          print(f"Keypoint {i}:")
          print(f" - pt: {kp.pt}")
          print(f" - size: {kp.size}")
          print(f" - angle: {kp.angle}")
          print(f" - response: {kp.response}")
          print(f" - octave: {kp.octave}")
          print(f" - class_id: {kp.class_id}")
          print(f"Descriptor {i}: {descriptors[i]}")
  
  
  keypoints, descriptors = extract_image_features(image_path)
  print_keypoints_and_descriptors(keypoints, descriptors)
  
  
  ```
  
  이미지를 Blip모델이 분석하고 캡션을 생성하며, openCV 라이브러리의 SIFT알고리즘을 통해 원본이미지의 패치별 키값와 설명을 저장하고, 생성된 캡션을 추가적인 작업을 통해 수정하고 수정된 캡션과 원본이미지의 패치별 특징을 기반으로 새로운 이미지를 생성하기 위한 전처리를 진행했다. 이후 다음 파일에서는 이러한 값들을 기반으로 이미지를 생성할 것이다.

![image](https://github.com/user-attachments/assets/481f740c-adf0-415c-8058-fffb0a362093)

| Main Developer | 김정훈 |
| :------------: | :----: |

[speech to text python module]

```python
from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment
import io
import os
import wave

def get_sample_rate(file_path):
    """WAV 파일의 샘플 레이트를 확인합니다."""
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()
    return sample_rate

def convert_to_mono(file_path):
    """오디오 파일을 모노로 변환합니다."""
    audio = AudioSegment.from_file(file_path)
    if audio.channels != 1:
        audio = audio.set_channels(1)
        mono_file = f"mono_{os.path.basename(file_path)}"
        audio.export(mono_file, format="wav")
        return mono_file
    return file_path

def resample_audio(file_path, target_sample_rate=48000):
    """오디오 파일을 리샘플링합니다."""
    audio = AudioSegment.from_file(file_path)
    if audio.frame_rate != target_sample_rate:
        audio = audio.set_frame_rate(target_sample_rate)
        resampled_file = f"resampled_{os.path.basename(file_path)}"
        audio.export(resampled_file, format="wav")
        return resampled_file
    return file_path

def convert_to_16bit(file_path):
    """WAV 파일을 16비트 샘플로 변환합니다."""
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_sample_width(2)  # 16비트 샘플
    bit16_file = f"bit16_{os.path.basename(file_path)}"
    audio.export(bit16_file, format="wav")
    return bit16_file

def split_audio(file_path, chunk_length_ms=10000):
    """오디오 파일을 주어진 길이(밀리초)로 자릅니다."""
    audio = AudioSegment.from_file(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_file = f"chunk_{i // chunk_length_ms}.wav"
        chunk.export(chunk_file, format="wav")
        chunks.append(chunk_file)
    return chunks

def transcribe_audio_file(file_path):
    """Google Cloud Speech-to-Text API를 사용하여 음성을 텍스트로 변환합니다."""
    client = speech.SpeechClient.from_service_account_file('myKey.json')

    # 오디오 파일의 샘플 레이트 확인
    sample_rate = get_sample_rate(file_path)

    # 오디오 파일을 모노로 변환
    mono_file_path = convert_to_mono(file_path)

    # 오디오 파일 리샘플링
    resampled_file_path = resample_audio(mono_file_path, target_sample_rate=sample_rate)

    # 오디오 파일 비트 깊이 변환
    bit16_file_path = convert_to_16bit(resampled_file_path)

    with io.open(bit16_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,  # 리샘플링한 샘플 레이트 사용
        language_code="ko-KR",
    )

    # 파일이 길 경우, long_running_recognize 사용
    operation = client.long_running_recognize(config=config, audio=audio)
    print('Waiting for operation to complete...')
    response = operation.result(timeout=90)

    transcripts = []
    for result in response.results:
        transcripts.append(result.alternatives[0].transcript)

    return transcripts

def transcribe_chunks(chunk_files):
    """여러 오디오 조각을 텍스트로 변환합니다."""
    all_transcripts = []
    for chunk_file in chunk_files:
        print(f"Transcribing {chunk_file}...")
        transcripts = transcribe_audio_file(chunk_file)
        print(transcripts)
        all_transcripts.extend(transcripts)
        os.remove(chunk_file)  # 임시 조각 파일 삭제
    return all_transcripts

# 사용 예제
audio_file_path = 'audio.wav'
chunk_files = split_audio(audio_file_path, chunk_length_ms=10000)  # 10초 길이의 조각으로 나누기
transcripts = transcribe_chunks(chunk_files)

# 전체 텍스트 출력
print("\n".join(transcripts))

```

![speech2text](https://github.com/user-attachments/assets/4ed0f1fb-8177-42bd-a0e3-71e31f263d89)

| Main Developer | 김정훈 |
| :------------: | :----: |
| Sub Developer  | 노태원 |

[google OCR Module]

```python
from google.cloud import vision_v1
import io

# 이미지에서 텍스트 추출
def detect_text_in_image(image_path, client):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision_v1.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        print(f"Detected text in {image_path}:")
        for text in texts:
            print(text.description)
    else:
        print(f"No text detected in {image_path}.")

# Google Cloud Vision 클라이언트 설정
def setup_vision_client():
    return vision_v1.ImageAnnotatorClient.from_service_account_file('myKey.json')

# 메인 함수
def main():
    image_path = './a.png'  # 텍스트를 추출할 이미지 파일의 경로

    client = setup_vision_client()
    detect_text_in_image(image_path, client)

if __name__ == '__main__':
    main()
```
![ocr](https://github.com/user-attachments/assets/054f71cd-baf9-4ff8-8357-2da39469ada3)

| Main Developer | 김정훈 |
| :------------: | :----: |
| Sub Developer  | 노태원 |



[image generate module]

```python
import os
import requests
import sys
import copy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd


class TwinConv(torch.nn.Module):
    def __init__(self, convin_pretrained, convin_curr):
        super(TwinConv, self).__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None

    def forward(self, x):
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        return x1 * (1 - self.r) + x2 * (self.r)


class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        if pretrained_name == "edge_to_image":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "edge_to_image_loras.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name == "sketch_to_image_stochastic":
            # download from url
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/sketch_to_image_stochastic_lora.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "sketch_to_image_stochastic_lora.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            convin_pretrained = copy.deepcopy(unet.conv_in)
            unet.conv_in = TwinConv(convin_pretrained, unet.conv_in)
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, c_t, prompt=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]
        if deterministic:
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc,).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            # scale the lora weights based on the r value
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            # combine the input and noise
            unet_input = encoded_control * r + noise_map * (1 - r)
            self.unet.conv_in.r = r
            unet_output = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc,).sample
            self.unet.conv_in.r = None
            x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)

```

| Main Developer | 김정훈 |
| :------------: | :----: |



[완성본 웹페이지 결과물]
![webMain](https://github.com/user-attachments/assets/bbcce266-67ff-4427-b76e-4abe4ddf78bd)

![adRank](https://github.com/user-attachments/assets/912926c8-7bc6-4646-82a4-d317c58f456d)

![NarakRank](https://github.com/user-attachments/assets/42995566-41ec-4d0b-8d9f-b803da091624)

![statDashBoard](https://github.com/user-attachments/assets/f9428ea2-e912-4fc3-9ac3-0406141b1e4c)

![statChart](https://github.com/user-attachments/assets/13e38238-8359-4e84-983e-a067f9e35956)

![googleTrendsSearchCount](https://github.com/user-attachments/assets/ceea6b51-d1a5-43e8-bfb5-b55c1f76f8cc)

![relative](https://github.com/user-attachments/assets/7e8a5943-121d-4c5d-8d28-84c3e621092b)

![captionGenerate](https://github.com/user-attachments/assets/d6475b77-bd27-4760-b695-ba0a80627a85)

![imageVariation](https://github.com/user-attachments/assets/b09cfc9f-fdc0-4cd0-9c0b-bab0943ca63f)

**3. 개발 레퍼런스**

https://htrend-4d67e.web.app/

https://kr.noxinfluencer.com/

**4. 시현 영상**
인플루언서 스탯분석에 관한 발표 영상
https://www.youtube.com/watch?v=e9kM0qYI3K4&t

이미지 캡션 모듈 연동 추가후 추가 발표 영상

**팀 기여도**

| 김정훈 |  7   |
| :----: | :--: |
| 노태원 |  2   |
| 이연준 |  1   |

