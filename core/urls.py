from django.urls import path
from . import views

urlpatterns = [
    path('inference/', views.InferenceView.as_view()),
    path('info/', views.InformationView.as_view()),
]