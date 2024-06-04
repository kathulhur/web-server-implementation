from django.urls import path
from . import views

urlpatterns = [
    path('inferencing/', views.InferencingView.as_view())
]