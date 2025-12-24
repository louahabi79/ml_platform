from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),  # Don't forget admin!
    path("", include("apps.api.ui_urls")),
    path("api/v1/", include("apps.api.v1.urls")),
]

# This is what serves the images/CSVs during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)