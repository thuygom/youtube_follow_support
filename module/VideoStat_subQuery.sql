use followSupport;

ALTER TABLE VideoStat
ADD COLUMN avg_view_count FLOAT,
ADD COLUMN avg_comment_count FLOAT,
ADD COLUMN avg_subscriber_count FLOAT,
ADD COLUMN comments_per_subscriber FLOAT,
ADD COLUMN likes_per_subscriber FLOAT,
ADD COLUMN views_per_subscriber FLOAT;

-- 평균 조회수, 평균 댓글 수, 평균 구독자 수 계산
UPDATE VideoStat v
JOIN (
    SELECT
        video_id,
        AVG(view_count) AS avg_view_count,
        AVG(comment_count) AS avg_comment_count,
        AVG(subscriber_count) AS avg_subscriber_count
    FROM VideoStat
    GROUP BY video_id
) AS avg_stats
ON v.video_id = avg_stats.video_id
SET 
    v.avg_view_count = avg_stats.avg_view_count,
    v.avg_comment_count = avg_stats.avg_comment_count,
    v.avg_subscriber_count = avg_stats.avg_subscriber_count;

-- 구독자 대비 댓글 수, 구독자 대비 좋아요 수, 구독자 대비 조회수 계산
UPDATE VideoStat
SET 
    comments_per_subscriber = IF(subscriber_count > 0, comment_count / subscriber_count, 0),
    likes_per_subscriber = IF(subscriber_count > 0, like_count / subscriber_count, 0),
    views_per_subscriber = IF(subscriber_count > 0, view_count / subscriber_count, 0);

select * from VideoStat;
