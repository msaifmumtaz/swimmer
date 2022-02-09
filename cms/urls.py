from django.contrib import admin
from django.urls import path
from cms import views
urlpatterns = [
    path('', views.index, name='home'),
    path('login', views.loginUser, name='login'),
    path('logout', views.logoutUser, name='logoutUser'),
    path('dashboard', views.dashboard, name='dashboard'),
    path('stream', views.streamer, name='stream'),
    path('textresponse', views.textresponse, name='textresponse'),
    path('process', views.process, name='process')
]