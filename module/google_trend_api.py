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
