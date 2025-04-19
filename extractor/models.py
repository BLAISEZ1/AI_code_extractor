from django.db import models

class UploadedVideo(models.Model):
    title = models.CharField(max_length=255)
    video_file = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class CodeSegment(models.Model):
    video = models.ForeignKey(UploadedVideo, on_delete=models.CASCADE)
    code = models.TextField()
    start_time = models.FloatField()
    end_time = models.FloatField()

    def __str__(self):
        return f"{self.video.title} [{self.start_time}s - {self.end_time}s]"
