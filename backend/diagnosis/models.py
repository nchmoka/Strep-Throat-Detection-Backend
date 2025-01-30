from django.db import models
from django.contrib.auth.models import User

# Model to store the results of a diagnosis for a user
class DiagnosisResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='diagnoses')  # Links the diagnosis to a user
    timestamp = models.DateTimeField(auto_now_add=True)  # Automatically stores the time of diagnosis creation
    label = models.CharField(max_length=20)  # Stores diagnosis label, e.g., 'healthy' or 'strep'
    probability = models.FloatField()  # Probability score of the diagnosis
    image = models.ImageField(upload_to='throat_images/', null=True, blank=True)  # Optional image upload for diagnosis

    # String representation of the diagnosis result
    def __str__(self):
        return f"{self.user.username} - {self.label} ({self.probability:.2f}) on {self.timestamp}"