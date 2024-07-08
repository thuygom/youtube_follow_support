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
- **목표2. 웹서비스에 멀티 모달리티를 활용한 자동 썸네일 제작 혹은 추가적인 기능을 제공하는 AI Model을 제작한다.(4~7주차 진행 예정)**

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

  | Main Developer | 김정훈 |
  | :------------: | :----: |

  

**3. 개발 레퍼런스**

https://htrend-4d67e.web.app/

https://kr.noxinfluencer.com/



**팀 기여도**

| 김정훈 |  6   |
| :----: | :--: |
| 노태원 |  2   |
| 이연준 |  2   |

