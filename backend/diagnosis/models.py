from django.db import models
from django.contrib.auth.models import User

class DiagnosisResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='diagnoses')
    timestamp = models.DateTimeField(auto_now_add=True)
    label = models.CharField(max_length=20)  # 'healthy' or 'strep'
    probability = models.FloatField()
    image = models.ImageField(upload_to='throat_images/', null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.label} ({self.probability:.2f}) on {self.timestamp}"

class EducationalResource(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


    def __str__(self):
        return self.title