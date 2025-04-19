import os
import whisper
from .file_writer import FileWriter
from extractor.models import CodeSegment

class VideoProcessor:
    def __init__(self, video_instance):
        self.video = video_instance
        self.model = whisper.load_model("base")

    def is_probable_code(self, text):
        keywords = ['def', 'class', 'import', 'return', '=', 'for', 'while', '{', '}', '(', ')']
        return any(keyword in text for keyword in keywords)

    def process(self):
        result = self.model.transcribe(self.video.video_file.path)
        segments = result['segments']

        file_writer = FileWriter(self.video.title)

        for segment in segments:
            start = segment['start']
            end = segment['end']
            text = segment['text']

            if self.is_probable_code(text):
                # Save to database
                CodeSegment.objects.create(
                    video=self.video,
                    code=text.strip(),
                    start_time=start,
                    end_time=end
                )
                # Save to file
                file_writer.write(text.strip(), start, end)

        file_writer.close()
