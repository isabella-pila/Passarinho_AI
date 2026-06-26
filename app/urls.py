from django.urls import path, re_path
from . import views
from django.contrib.staticfiles.views import serve

urlpatterns = [
    path('', views.index, name='index'),
    re_path(r'^sw\.js$', serve, {'path': 'sw.js'}),
]
