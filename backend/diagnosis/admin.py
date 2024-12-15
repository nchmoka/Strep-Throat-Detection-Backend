from django.contrib import admin
from .models import DiagnosisResult, EducationalResource

@admin.register(DiagnosisResult)
class DiagnosisResultAdmin(admin.ModelAdmin):
    list_display = ('user', 'label', 'probability', 'timestamp')

@admin.register(EducationalResource)
class EducationalResourceAdmin(admin.ModelAdmin):
    list_display = ('title', 'created_at', 'updated_at')
