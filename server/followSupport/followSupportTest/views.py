from django.shortcuts import render
from followSupportTest.models import VideoStat, Comment

def home(request):
    # VIDEO_Oking, VIDEO_STYLE 등 각 카테고리의 video_id 리스트 정의
    VIDEO_Oking = ['oakQvwCbvr8', '-HZeqsIgGHo', 'nAy-7zuCVQs', 'QojVuirFx58', 'ibI5OOZXSj8']
    VIDEO_STYLE = ['g5KDoSqT24Q', 'mnn1_yu0aDQ', 'pp_C0MGj9ZM', '_Otk-iMD_X0', '_HZ63R-8z4E']
    VIDEO_GAME = ['yLlKOd3I8CA', 'BYWO-z-4tfo', 'uNq7RMRwIHs', 'ZLXz98YW_U0', 'qkXc1M3d7g4']
    VIDEO_MUSIC = ['-rHqqCxjhM4', 'FGjrtBFgTXY', 'TOSrdHy5KRc', 'wdUu3XQK8cE', 'LamRCcz4zqg']
    VIDEO_ISSUE = ['ahcPfSLbT-M', '8l4GZ4datyM', '7I790Er-zkc', '8SJs1Cg7hpU', 'VWmWScovllY']

    # 각 카테고리에 해당하는 VideoStat 객체들 필터링
    videos_oking = VideoStat.objects.filter(video_id__in=VIDEO_Oking)
    videos_style = VideoStat.objects.filter(video_id__in=VIDEO_STYLE)
    videos_game = VideoStat.objects.filter(video_id__in=VIDEO_GAME)
    videos_music = VideoStat.objects.filter(video_id__in=VIDEO_MUSIC)
    videos_issue = VideoStat.objects.filter(video_id__in=VIDEO_ISSUE)

    VideoStat_data = VideoStat.objects.all()
    Comment_data = Comment.objects.all()

    # 사용자가 선택한 카테고리
    selected_category = request.GET.get('category', None)
    selected_video = request.GET.get('video_id',None)

    # 선택된 카테고리에 따라 VideoStat 필터링
    if selected_category == 'VIDEO_Oking':
        videos_selected = VideoStat.objects.filter(video_id__in=VIDEO_Oking)
    elif selected_category == 'VIDEO_STYLE':
        videos_selected = VideoStat.objects.filter(video_id__in=VIDEO_STYLE)
    elif selected_category == 'VIDEO_GAME':
        videos_selected = VideoStat.objects.filter(video_id__in=VIDEO_GAME)
    elif selected_category == 'VIDEO_MUSIC':
        videos_selected = VideoStat.objects.filter(video_id__in=VIDEO_MUSIC)
    elif selected_category == 'VIDEO_ISSUE':
        videos_selected = VideoStat.objects.filter(video_id__in=VIDEO_ISSUE)
    else:
        videos_selected = VideoStat.objects.none()  # 기본적으로 아무 것도 선택되지 않았을 때

    # 템플릿에 전달할 context 설정
    context = {
        'VideoStat_data': VideoStat_data,
        'Comment_data': Comment_data,
        'videos_selected': videos_selected,
        'selected_category': selected_category,
        'selected_video':selected_video,
    }

    return render(request, 'home.html',context)
