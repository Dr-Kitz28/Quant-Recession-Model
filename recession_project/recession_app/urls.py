# recession_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # The main page that shows the image instantly
    path('', views.home, name='home'),
    
    # Go to /update-heatmap/ to run the scripts and generate a new image
    path('update-heatmap/', views.update_heatmap, name='update_heatmap'),
]