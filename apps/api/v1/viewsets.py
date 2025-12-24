from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from apps.api.v1.permissions import IsOwner, ReadOnly
from apps.api.v1.serializers import (
    AlgorithmSerializer,
    ProjectSerializer,
    DatasetSerializer,
    DatasetVersionSerializer,
    PipelineSerializer,
    PipelineStepSerializer,
    ExperimentSerializer,
)
from apps.datasets.models import Project, Dataset, DatasetVersion
from apps.ml_models.models import Algorithm, Pipeline, PipelineStep, Experiment
from apps.ml_models.services.experiment_service import run_experiment_sync, run_experiment_threaded


class OwnerQuerySetMixin:
    """
    Ensures all list queries are restricted to request.user.
    Assumes model has owner field.
    """

    def get_queryset(self):
        qs = super().get_queryset()
        return qs.filter(owner=self.request.user)


class AlgorithmViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Algorithm.objects.all()
    serializer_class = AlgorithmSerializer
    permission_classes = [IsAuthenticated, ReadOnly]


class ProjectViewSet(OwnerQuerySetMixin, viewsets.ModelViewSet):
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer
    permission_classes = [IsAuthenticated, IsOwner]

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)


class DatasetViewSet(OwnerQuerySetMixin, viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    permission_classes = [IsAuthenticated, IsOwner]

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)

    @action(detail=True, methods=["post"], url_path="upload-csv")
    def upload_csv(self, request, pk=None):
        """
        Uploads CSV and creates a new DatasetVersion.
        Phase 5 will improve parsing rules and profiling.
        """
        dataset = self.get_object()
        file = request.FILES.get("file")
        if not file:
            return Response({"file": "CSV file is required."}, status=status.HTTP_400_BAD_REQUEST)

        # Create dataset version (service can be introduced later; kept simple here)
        content = file.read()
        sha256 = DatasetVersion.compute_sha256(content)

        latest = DatasetVersion.objects.filter(dataset=dataset).order_by("-version_number").first()
        next_version = (latest.version_number + 1) if latest else 1

        # Rewind file for Django FileField storage
        file.seek(0)

        dv = DatasetVersion.objects.create(
            dataset=dataset,
            created_by=request.user,
            version_number=next_version,
            file=file,
            sha256=sha256,
            row_count=0,
            column_count=0,
            schema_json={},
            profile_json={},
        )
        return Response(DatasetVersionSerializer(dv).data, status=status.HTTP_201_CREATED)


class DatasetVersionViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Versions are immutable snapshots; keep read-only in API.
    """
    queryset = DatasetVersion.objects.select_related("dataset", "dataset__owner")
    serializer_class = DatasetVersionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return self.queryset.filter(dataset__owner=self.request.user)

    @action(detail=True, methods=["post"], url_path="profile")
    def profile(self, request, pk=None):
        """
        Phase 5 will implement full profiling + DatasetColumn creation.
        For now, keep it as a placeholder? Not allowed.
        So we do real basic profiling now: row/col counts + schema_json.
        """
        dv = self.get_object()
        import pandas as pd

        df = pd.read_csv(dv.file.path)
        dv.row_count = int(df.shape[0])
        dv.column_count = int(df.shape[1])
        dv.schema_json = {"columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]}
        dv.profile_json = {
            "missing_by_column": {c: int(df[c].isna().sum()) for c in df.columns},
            "unique_by_column": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
        }
        dv.save(update_fields=["row_count", "column_count", "schema_json", "profile_json"])
        return Response(DatasetVersionSerializer(dv).data, status=status.HTTP_200_OK)


class PipelineViewSet(OwnerQuerySetMixin, viewsets.ModelViewSet):
    queryset = Pipeline.objects.all()
    serializer_class = PipelineSerializer
    permission_classes = [IsAuthenticated, IsOwner]

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)


class PipelineStepViewSet(viewsets.ModelViewSet):
    queryset = PipelineStep.objects.select_related("pipeline", "pipeline__owner")
    serializer_class = PipelineStepSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return self.queryset.filter(pipeline__owner=self.request.user)

    def perform_create(self, serializer):
        pipeline = serializer.validated_data["pipeline"]
        if pipeline.owner_id != self.request.user.id:
            raise PermissionError("Forbidden.")
        serializer.save()


class ExperimentViewSet(OwnerQuerySetMixin, viewsets.ModelViewSet):
    queryset = Experiment.objects.select_related("pipeline", "dataset_version", "algorithm")
    serializer_class = ExperimentSerializer
    permission_classes = [IsAuthenticated, IsOwner]

    def perform_create(self, serializer):
        exp = serializer.save(owner=self.request.user)

        # Execution Strategy: synchronous by default; optionally threaded.
        # Client can pass ?mode=threaded to avoid blocking UI.
        mode = self.request.query_params.get("mode", "sync").lower()
        if mode == "threaded":
            run_experiment_threaded(exp.id)
        else:
            run_experiment_sync(exp.id)
