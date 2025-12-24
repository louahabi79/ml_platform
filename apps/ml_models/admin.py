from django.contrib import admin
from .models import (
    Algorithm,
    AlgorithmTutorialSection,
    Pipeline,
    PipelineStep,
    ManualFeature,
    Experiment,
    ModelArtifact,
    EvaluationResult,
    CVFoldResult,
    PredictionBatch,
    ExportArtifact,
)

@admin.register(Algorithm)
class AlgorithmAdmin(admin.ModelAdmin):
    list_display = ("display_name", "family", "key", "sklearn_class_path")
    search_fields = ("display_name", "key", "sklearn_class_path")
    list_filter = ("family",)

@admin.register(AlgorithmTutorialSection)
class AlgorithmTutorialSectionAdmin(admin.ModelAdmin):
    list_display = ("algorithm", "order", "title")
    list_filter = ("algorithm",)
    ordering = ("algorithm", "order")

admin.site.register(Pipeline)
admin.site.register(PipelineStep)
admin.site.register(ManualFeature)
admin.site.register(Experiment)
admin.site.register(ModelArtifact)
admin.site.register(EvaluationResult)
admin.site.register(CVFoldResult)
admin.site.register(PredictionBatch)
admin.site.register(ExportArtifact)
