from django.shortcuts import render,redirect
from django.http import HttpResponse, StreamingHttpResponse, request,HttpResponseServerError
from .trackingobject import Tracker
from django.core.files.storage import default_storage
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
import os
from django.conf import settings
from django.contrib import messages

def index(request):
    return render(request, "index.html")

def loginUser(request):

    if request.user.is_authenticated:
        return redirect('/dashboard')
    else:
        if request.method == 'POST':
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)

            if user is not None:
                login(request, user)
                return redirect('/dashboard')
            else:
                messages.warning(request, 'Username OR password is incorrect')
    data = {'title': 'User', 'register_url': '/register'}
    return render(request, 'signin.html', data)

def logoutUser(request):
    logout(request)
    return redirect('/')
    
@login_required(login_url='/login')
def dashboard(request):
    if request.method=='POST':
        file=request.FILES.get('video')
        STORAGE_PATH = os.path.join(settings.BASE_DIR, 'media')
        if os.path.exists(STORAGE_PATH+"/video_file.mp4"):
            os.remove(STORAGE_PATH+"/video_file.mp4")
        file_name = default_storage.save('video_file.mp4', file)
        messages.success(
                    request, 'Video Uploaded Successfuly Now go to processing page.')
        return render(request, 'processmessage.html')
    else:
        return render(request,'dashboard.html')

@login_required(login_url='/login')
def streamer(request):
    t=Tracker()
    return StreamingHttpResponse(t.get_frame(), content_type="multipart/x-mixed-replace;boundary=frame")

@login_required(login_url='/login')
def textresponse(request):
    STORAGE_PATH = os.path.join(settings.BASE_DIR, 'media')

    f = open(STORAGE_PATH+'/response.txt', 'r')
    file_content = f.read()
    f.close()
    return HttpResponse(file_content, content_type="text/plain")

@login_required(login_url='/login')
def process(request):
    return render(request,'process.html')