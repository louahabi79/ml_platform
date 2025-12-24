from rest_framework.routers import DefaultRouter
from django.urls import include, path

from .viewsets import (
    AlgorithmViewSet,
    ProjectViewSet,
    DatasetViewSet,
    DatasetVersionViewSet,
    PipelineViewSet,
    PipelineStepViewSet,
    ExperimentViewSet,
)

router = DefaultRouter()
router.register(r"algorithms", AlgorithmViewSet, basename="algorithms")
router.register(r"projects", ProjectViewSet, basename="projects")
router.register(r"datasets", DatasetViewSet, basename="datasets")
router.register(r"dataset-versions", DatasetVersionViewSet, basename="dataset_versions")
router.register(r"pipelines", PipelineViewSet, basename="pipelines")
router.register(r"pipeline-steps", PipelineStepViewSet, basename="pipeline_steps")
router.register(r"experiments", ExperimentViewSet, basename="experiments")

urlpatterns = [
    path("", include(router.urls)),
]
