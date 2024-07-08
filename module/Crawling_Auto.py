import pandas as pd
from googleapiclient.discovery import build
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# API key and video ID
API_KEY = 'AIzaSyDW3zH1Dl-NcrvNrdQn7CsCDcBE41o24QM'
VIDEO_ID = '6jjrOl2EmeQ'

# File paths
STATS_FILE_PATH = '../xlsx/stats.xlsx'
COMMENTS_FILE_PATH = '../xlsx/crawling_auto.xlsx'

# Initialize YouTube API object
def initialize_api(api_key):
    return build('youtube', 'v3', developerKey=api_key)

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

# Function to save statistics to Excel
def save_statistics_to_excel(stats_df, file_path):
    try:
        existing_stats_df = pd.read_excel(file_path)
        stats_df = pd.concat([existing_stats_df, stats_df], ignore_index=True)
    except FileNotFoundError:
        pass
    stats_df.to_excel(file_path, index=False)
    print(f"통계 정보를 {file_path}에 저장했습니다.")

# Function to extract comments
def extract_video_comments(api_obj, video_id):
    comments = []
    response = api_obj.commentThreads().list(
        part='snippet,replies',
        videoId=video_id,
        maxResults=100  # Increased maxResults for more comments
    ).execute()
    
    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([comment['textDisplay'], comment['authorDisplayName'], comment['publishedAt'], comment['likeCount']])
    
            if item['snippet']['totalReplyCount'] > 0:
                for reply_item in item['replies']['comments']:
                    reply = reply_item['snippet']
                    comments.append([reply['textDisplay'], reply['authorDisplayName'], reply['publishedAt'], reply['likeCount']])
    
        if 'nextPageToken' in response:
            response = api_obj.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=response['nextPageToken'],
                maxResults=100
            ).execute()
        else:
            break
    
    return pd.DataFrame(comments, columns=['comment', 'author', 'date', 'num_likes'])

# Function to save comments to Excel
def save_comments_to_excel(comments_df, file_path):
    comments_df.to_excel(file_path, header=['comment', 'author', 'date', 'num_likes'], index=None)
    print(f"댓글을 {file_path}에 저장했습니다.")

# Function to get channel information
def get_channel_info(api_obj, channel_id):
    response = api_obj.channels().list(
        part='snippet,statistics',
        id=channel_id
    ).execute()
    channel_info = response['items'][0]
    snippet = channel_info['snippet']
    statistics = channel_info['statistics']
    return snippet, statistics

def main():
    # Initialize API
    api_obj = initialize_api(API_KEY)
    
    # Get video info
    snippet, statistics, topic_details = get_video_info(api_obj, VIDEO_ID)
    
    # Extract relevant information
    current_date = datetime.now().strftime('%Y-%m-%d')
    upload_date = snippet['publishedAt']
    channel_id = snippet['channelId']  # Get channel ID
    
    # Get channel info for channel title and description
    channel_snippet, channel_statistics = get_channel_info(api_obj, channel_id)
    channel_title = channel_snippet['title']
    channel_description = channel_snippet['description']
    subscriber_count = channel_statistics.get('subscriberCount', 0)
    
    # Create DataFrame for statistics
    stats_data = {
        'video_id': VIDEO_ID,
        'upload_date': [upload_date],
        'date': [current_date],
        'view_count': [statistics.get('viewCount')],
        'like_count': [statistics.get('likeCount')],
        'comment_count': [statistics.get('commentCount')],
        'subscriber_count': [subscriber_count],
        'channel_id': [channel_id],
        'channel_title': [channel_title],
        'channel_description': [channel_description],
        'topic_categories': [', '.join(topic_details.get('topicCategories', []))],
        'title': [snippet['title']],
        'description': [snippet['description']],
        'tags': [', '.join(snippet.get('tags', []))],
        'thumbnails': [snippet['thumbnails']['default']['url']]  # Adjust the key as per your requirement
    }
    stats_df = pd.DataFrame(stats_data)
    
    # Save statistics to Excel
    save_statistics_to_excel(stats_df, STATS_FILE_PATH)
    
    # Extract comments
    comments_df = extract_video_comments(api_obj, VIDEO_ID)
    
    # Save comments to Excel
    save_comments_to_excel(comments_df, COMMENTS_FILE_PATH)

if __name__ == "__main__":
    main()
