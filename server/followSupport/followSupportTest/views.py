from django.shortcuts import render, redirect
from followSupportTest.models import VideoStat, YoutuberStat, GoogleTrend, RelatedKeyword
import json
from followSupportTest.forms import ImageUploadForm
from followSupportTest.models import ImageUpload

from .utils import create_caption, variation_image

def home(request):
    # 템플릿에 전달할 context 설정
    context = {
    }
    return render(request, 'home.html', context)

def about(request):
    # 쿼리 매개변수 받기
    youtuber_id = request.GET.get('youtuber_id', 'Unknown')

    # image 필터링
    if youtuber_id == 'VIDEO_Oking':
        image_selected = "button1.jpg"
    elif youtuber_id == 'VIDEO_ISSUE':
        image_selected = "button2.jpg"
    elif youtuber_id == 'VIDEO_GAME':
        image_selected = "button3.jpg"
    elif youtuber_id == 'VIDEO_STYLE':
        image_selected = "button4.jpg"
    elif youtuber_id == 'VIDEO_MUSIC':
        image_selected = "button5.jpg"
    else:
        videos_selected = VideoStat.objects.none()  # 기본적으로 아무 것도 선택되지 않았을 때

    #채널이름 필터링
    if youtuber_id == 'VIDEO_Oking':
        id = "오킹TV"
    elif youtuber_id == 'VIDEO_ISSUE':
        id = "뻑가 PPKKa"
    elif youtuber_id == 'VIDEO_GAME':
        id = "동수칸TV"
    elif youtuber_id == 'VIDEO_STYLE':
        id = "깡스타일리스트"
    elif youtuber_id == 'VIDEO_MUSIC':
        id = "때잉"
    else:
        videos_selected = VideoStat.objects.none()  # 기본적으로 아무 것도 선택되지 않았을 때

    filtered_stat = YoutuberStat.objects.filter(channel_title=id)

    filtered_keyword = RelatedKeyword.objects.filter(channel_title=id)

    idReplace = id.replace(' ', '')

    google_trend_data = GoogleTrend.objects.filter(date__year=2024).values('date', idReplace)

    dates = [entry['date'].strftime('%Y-%m-%d') for entry in google_trend_data]
    values = [entry[idReplace] for entry in google_trend_data]

        # 데이터 변환
    stat_dates = [stat.date.strftime('%Y-%m-%d') for stat in filtered_stat]
    follow_support = [stat.followSupport for stat in filtered_stat]
    follow_score = [stat.followScore for stat in filtered_stat]
    emotion = [stat.emotion for stat in filtered_stat]
    avg_view_count = [stat.avg_view_count for stat in filtered_stat]
    avg_comment_count = [stat.avg_comment_count for stat in filtered_stat]
    avg_subscriber_count = [stat.avg_subscriber_count for stat in filtered_stat]


    # 템플릿에 전달할 context 설정
    context = {
        'youtuber_stat_data': filtered_stat,
        'google_trend_data': google_trend_data,
        'related_keyword_data': filtered_keyword,
        'image_selected': image_selected,
        'channel_id':id,

        'dates': json.dumps(dates),
        'values': json.dumps(values),

        'stat_dates': json.dumps(stat_dates),
        'follow_support': json.dumps(follow_support),
        'follow_score': json.dumps(follow_score),
        'emotion':json.dumps(emotion),
        'avg_view_count': json.dumps(avg_view_count),
        'avg_comment_count': json.dumps(avg_comment_count),
        'avg_subscriber_count': json.dumps(avg_subscriber_count),
    }

    return render(request, 'about.html', context)


def contact(request):
    # 초기값 설정
    uploaded_image_url = request.session.get('uploaded_image_url', None)
    caption = request.session.get('caption', None)

    # 폼을 항상 정의
    form = ImageUploadForm()

    if request.method == 'POST':
        if 'upload' in request.POST:
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                image_instance = form.save()
                # 업로드된 이미지 URL을 세션에 저장
                request.session['uploaded_image_url'] = image_instance.image.url
                # 페이지 새로고침 없이 계속 이미지를 보여주기 위해 redirect
                return redirect('contact')
        elif 'caption' in request.POST:
            if uploaded_image_url:
                # 이미지 URL을 create_caption 함수에 전달
                caption = create_caption("C:/Windows/System32/followSupport" + uploaded_image_url)
                # 생성된 캡션을 세션에 저장
                request.session['caption'] = caption
        elif 'variation' in request.POST:
            prompt = request.POST.get('prompt')
            print(f"Prompt received: {prompt}")  # 로그 출력
            if uploaded_image_url:
                variation_image("C:/Windows/System32/followSupport" + uploaded_image_url, "edge_to_image",prompt)
        elif 'generate' in request.POST:
            prompt = request.POST.get('prompt')
            print(f"Prompt received: {prompt}")  # 로그 출력
            if uploaded_image_url:
                variation_image("C:/Windows/System32/followSupport" + uploaded_image_url, "sketch_to_image_stochastic",prompt)

    return render(request, 'contact.html', {
        'form': form,
        'uploaded_image_url': uploaded_image_url,
        'caption': caption,
    })