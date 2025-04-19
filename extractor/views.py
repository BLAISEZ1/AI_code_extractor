from django.shortcuts import render, redirect
from .forms import VideoUploadForm
from .models import CodeSegment, UploadedVideo
from .utils.video_processor import VideoProcessor

def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
            processor = VideoProcessor(video)
            processor.process()
            return redirect('view_results', video_id=video.id)
    else:
        form = VideoUploadForm()
    return render(request, 'upload.html', {'form': form})

def view_results(request, video_id):
    video = UploadedVideo.objects.get(pk=video_id)
    segments = CodeSegment.objects.filter(video=video)
    return render(request, 'results.html', {'video': video, 'segments': segments})
