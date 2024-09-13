from django.urls import path
from inference.views import *

urlpatterns = [
    path('', generate, name = 'generate'),
]