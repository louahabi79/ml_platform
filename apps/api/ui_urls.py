from django.urls import path
from apps.api import ui_views

urlpatterns = [
    path("", ui_views.dashboard, name="dashboard"),

    path("projects/<int:project_id>/", ui_views.project_detail, name="project_detail"),

    path("datasets/upload/", ui_views.dataset_upload, name="dataset_upload"),
    path("datasets/<int:dataset_id>/", ui_views.dataset_detail, name="dataset_detail"),
    path("dataset-versions/<int:version_id>/", ui_views.dataset_version_detail, name="dataset_version_detail"),

    path("pipelines/wizard/<int:version_id>/", ui_views.pipeline_wizard, name="pipeline_wizard"),
    path("pipelines/<int:pipeline_id>/", ui_views.pipeline_detail, name="pipeline_detail"),

    path("experiments/create/<int:pipeline_id>/", ui_views.experiment_create, name="experiment_create"),
    path("experiments/<int:experiment_id>/", ui_views.experiment_results, name="experiment_results"),

    path("artifacts/model/<int:experiment_id>/download/", ui_views.download_model, name="download_model"),

    path("experiments/<int:experiment_id>/predict/", ui_views.predict_view, name="predict_view"),
    path("experiments/<int:experiment_id>/report.pdf", ui_views.download_report_pdf, name="download_report_pdf"),
]
