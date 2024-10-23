from django.db import models

class AudioFile(models.Model):
    audio_file = models.FileField(upload_to='audio/')  # 存储上传的WAV文件
    uploaded_at = models.DateTimeField(auto_now_add=True)
