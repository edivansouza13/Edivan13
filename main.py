from flask import Flask, request, jsonify, send_file
from moviepy.editor import VideoFileClip, concatenate_videoclips
import whisper
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

model = whisper.load_model("base")

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    video_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{video_id}.mp4")
    file.save(input_path)

    result = model.transcribe(input_path)
    segments = result['segments']
    hook_words = ["como", "descubra", "segredo", "você sabia", "nunca mais"]
    selected = [s for s in segments if any(h in s['text'].lower() for h in hook_words)]

    if not selected:
        return jsonify({'error': 'Nenhum trecho com gancho encontrado'}), 400

    clips = []
    total_duration = 0
    video = VideoFileClip(input_path)

    for seg in selected:
        start = max(0, seg['start'] - 0.5)
        end = min(video.duration, seg['end'] + 0.5)
        duration = end - start
        if total_duration + duration > 90:
            break
        clip = video.subclip(start, end)
        clips.append(clip)
        total_duration += duration

    final_clip = concatenate_videoclips(clips)
    output_path = os.path.join(PROCESSED_FOLDER, f"{video_id}_short.mp4")
    final_clip.write_videofile(output_path, codec="libx264")

    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
