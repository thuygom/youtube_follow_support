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
