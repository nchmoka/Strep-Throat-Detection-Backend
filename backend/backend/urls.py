from django.contrib import admin
from django.urls import path
from diagnosis.views import register_user, login_user, logout_user, analyze_image, list_resources, resource_detail
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('register/', register_user, name='register'),
    path('login/', login_user, name='login'),
    path('logout/', logout_user, name='logout'),
    path('analyze/', analyze_image, name='analyze'),
    path('resources/', list_resources, name='list_resources'),
    path('resources/<int:resource_id>/', resource_detail, name='resource_detail'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
