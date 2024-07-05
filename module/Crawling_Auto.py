import pandas as pd
from googleapiclient.discovery import build
import warnings  # 경고창 무시
from datetime import datetime

warnings.filterwarnings('ignore')

myKey = 'AIzaSyDW3zH1Dl-NcrvNrdQn7CsCDcBE41o24QM'
myVideo = '2FH0VaqQc84'

# API 객체 초기화
api_obj = build('youtube', 'v3', developerKey=myKey)

# 동영상 통계 정보를 가져오는 함수
def get_video_statistics(video_id):
    response = api_obj.videos().list(
        part='statistics',
        id=video_id
    ).execute()
    stats = response['items'][0]['statistics']
    return stats

# 동영상 통계 정보 가져오기
statistics = get_video_statistics(myVideo)
current_date = datetime.now().strftime('%Y-%m-%d')

# 모든 통계 정보를 DataFrame으로 저장
stats_data = {
    'date': [current_date],
    'view_count': [statistics.get('viewCount')],
    'like_count': [statistics.get('likeCount')],
    'dislike_count': [statistics.get('dislikeCount')],
    'comment_count': [statistics.get('commentCount')]
}

stats_df = pd.DataFrame(stats_data)

# 기존 통계 데이터 로드 (파일이 있을 경우)
try:
    existing_stats_df = pd.read_excel('../xlsx/stats.xlsx')
    stats_df = pd.concat([existing_stats_df, stats_df], ignore_index=True)
except FileNotFoundError:
    pass

# 업데이트된 통계 데이터 저장
stats_df.to_excel('../xlsx/stats.xlsx', index=False)
print("통계 정보 추출 완료")

# 댓글 추출
#comments = list()  # 댓글 리스트

#response = api_obj.commentThreads().list(
#    part='snippet,replies',
#    videoId=myVideo,
#    maxResults=5
#).execute()

#while response:
#    for item in response['items']:
#        comment = item['snippet']['topLevelComment']['snippet']
#        comments.append([comment['textDisplay'], comment['authorDisplayName'], comment['publishedAt'], comment['likeCount']])
#
#        if item['snippet']['totalReplyCount'] > 0:
#            for reply_item in item['replies']['comments']:
#                reply = reply_item['snippet']
#                comments.append([reply['textDisplay'], reply['authorDisplayName'], reply['publishedAt'], reply['likeCount']])
#
#    if 'nextPageToken' in response:
#        response = api_obj.commentThreads().list(part='snippet,replies', videoId=myVideo, pageToken=response['nextPageToken'], maxResults=100).execute()
#    else:
#        break
#
#df = pd.DataFrame(comments)
#df.to_excel('../xlsx/crawling_auto.xlsx', header=['comment', 'author', 'date', 'num_likes'], index=None)
#print("댓글 추출 완료")
