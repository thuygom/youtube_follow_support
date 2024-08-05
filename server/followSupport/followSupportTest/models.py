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

class YoutuberStat(models.Model):
    youtuber_num = models.AutoField(primary_key=True)  # AutoField를 사용하여 자동 증가를 처리합니다.
    channel_title = models.CharField(max_length=100)
    date = models.DateField()
    followSupport = models.FloatField()
    followScore = models.FloatField()
    emotion = models.CharField(max_length=100, null=True, blank=True)
    avg_view_count = models.IntegerField(null=True, blank=True)
    avg_comment_count = models.IntegerField(null=True, blank=True)
    avg_subscriber_count = models.IntegerField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = 'YoutuberStat'

    def __str__(self):
        return f"{self.channel_title} - {self.date}"
    
class GoogleTrend(models.Model):
    trend_num = models.AutoField(primary_key=True)  # AutoField를 사용하여 자동 증가를 처리합니다.
    date = models.DateField()
    오킹TV = models.IntegerField(null=True, blank=True)
    동수칸TV = models.IntegerField(null=True, blank=True)
    뻑가PPKKa = models.IntegerField(null=True, blank=True)
    깡스타일리스트 = models.IntegerField(null=True, blank=True)
    때잉 = models.IntegerField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = 'googleTrend'

    def __str__(self):
        return str(self.date)
    
class RelatedKeyword(models.Model):
    keyword_num = models.AutoField(primary_key=True)  # AutoField를 사용하여 자동 증가를 처리합니다.
    channel_title = models.CharField(max_length=100)
    keyword = models.CharField(max_length=100)
    indicator = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'relatedKeyword'

    def __str__(self):
        return f"{self.channel_title} - {self.keyword}"
    
class ImageUpload(models.Model):
    image = models.ImageField(upload_to='uploads/')

    def __str__(self):
        return self.image.name