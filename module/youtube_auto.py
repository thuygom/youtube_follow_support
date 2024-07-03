import pandas
from googleapiclient.discovery import build

import warnings # 경고창 무시
warnings.filterwarnings('ignore')

myKey = 'AIzaSyDW3zH1Dl-NcrvNrdQn7CsCDcBE41o24QM'
myVideo = '2FH0VaqQc84'

comments = list() #댓글 list
api_obj = build('youtube', 'v3', developerKey = myKey) #google API객체 생성
response = api_obj.commentThreads().list(
    part='snippet,replies',
    videoId=myVideo,
    maxResults=5
).execute()
#해당 youtube page에서 F12를 누른 후 element(요소)에 들어가 videoId를 ctrl+F로 찾은 후 입력

while response:
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([comment['textDisplay'], comment['authorDisplayName'], comment['publishedAt'], comment['likeCount']])
 
        if item['snippet']['totalReplyCount'] > 0:
            for reply_item in item['replies']['comments']:
                reply = reply_item['snippet']
                comments.append([reply['textDisplay'], reply['authorDisplayName'], reply['publishedAt'], reply['likeCount']])
 
    if 'nextPageToken' in response:
        response = api_obj.commentThreads().list(part='snippet,replies', videoId=myVideo, pageToken=response['nextPageToken'], maxResults=100).execute()
    else:
        break

df = pandas.DataFrame(comments)
df.to_excel('../xlsx/crawling_auto.xlsx', header=['comment', 'author', 'date', 'num_likes'], index=None)
print("comment extract complete")
