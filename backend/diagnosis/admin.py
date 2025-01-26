from django.contrib import admin
from .models import DiagnosisResult

@admin.register(DiagnosisResult)
class DiagnosisResultAdmin(admin.ModelAdmin):
    list_display = ('user', 'label', 'probability', 'timestamp')


