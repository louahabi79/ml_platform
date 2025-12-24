from django.urls import include, path

urlpatterns = [
    # SSR UI
    path("", include("apps.api.ui_urls")),

    # REST API
    path("api/v1/", include("apps.api.v1.urls")),
]
