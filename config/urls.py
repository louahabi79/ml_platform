# from django.conf import settings
# from django.conf.urls.static import static
# from django.contrib import admin
# from django.urls import include, path

# urlpatterns = [
#     path("admin/", admin.site.urls),  # Don't forget admin!
#     path("", include("apps.api.ui_urls")),
#     path("api/v1/", include("apps.api.v1.urls")),
# ]

# # This is what serves the images/CSVs during development
# if settings.DEBUG:
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),

    # Django auth (login/logout/password reset, etc.)
    path("accounts/", include("django.contrib.auth.urls")),

    # Signup
    path("accounts/", include("apps.users.urls")),

    # SSR + API (from apps/api/urls.py)
    path("", include("apps.api.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
