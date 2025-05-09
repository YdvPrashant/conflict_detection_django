from django.shortcuts import render
from django.http import StreamingHttpResponse
from .camera import gen_frames

def index(request):
    return render(request, 'index.html')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
