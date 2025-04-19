import os

class FileWriter:
    def __init__(self, video_title):
        self.filename = f"media/{video_title.replace(' ', '_')}_code.txt"
        self.file = open(self.filename, 'w', encoding='utf-8')

    def write(self, code, start, end):
        self.file.write(f"--- {start}s to {end}s ---\n")
        self.file.write(code + "\n\n")

    def close(self):
        self.file.close()
