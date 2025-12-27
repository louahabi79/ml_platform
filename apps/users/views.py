from __future__ import annotations

from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.urls import reverse


def signup(request):
    """
    SSR signup using Django's built-in UserCreationForm.
    On success: log the user in and redirect to dashboard.
    """
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("dashboard")
        return render(request, "users/signup.html", {"form": form})

    form = UserCreationForm()
    return render(request, "users/signup.html", {"form": form})
