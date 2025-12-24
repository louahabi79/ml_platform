from django.contrib import admin
from .models import Project, Dataset, DatasetDraft, DatasetVersion, DatasetColumn

admin.site.register(Project)
admin.site.register(Dataset)
admin.site.register(DatasetDraft)
admin.site.register(DatasetVersion)
admin.site.register(DatasetColumn)
