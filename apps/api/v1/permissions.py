from rest_framework.permissions import BasePermission, SAFE_METHODS


class IsOwner(BasePermission):
    """
    Object-level permission: only owners can access.
    Assumes model has `owner` field.
    """

    def has_object_permission(self, request, view, obj) -> bool:
        return getattr(obj, "owner_id", None) == getattr(request.user, "id", None)


class ReadOnly(BasePermission):
    def has_permission(self, request, view) -> bool:
        return request.method in SAFE_METHODS
